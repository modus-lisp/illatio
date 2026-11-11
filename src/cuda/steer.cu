// steer.cu — directional steering (src/steer.lisp): for each row y of a
// layer's output, y <- y - scale * u (u . y), u the layer's unit direction;
// and, when ADD is given, ADD's row += the result (the residual, for a pass
// that would have added the output in its product).  One block a row.

extern "C" __global__ void steer_rows(float *y, const float *u, float scale, int d, float *add) {
    float *row = y + (long long)blockIdx.x * d;
    __shared__ float part[32];
    float k = 0.f;
    if (scale != 0.f) {                                  // (uniform over the block)
        float s = 0.f;
        for (int i = threadIdx.x; i < d; i += blockDim.x) s += u[i] * row[i];
        for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
        if ((threadIdx.x & 31) == 0) part[threadIdx.x >> 5] = s;
        __syncthreads();
        if (threadIdx.x < 32) {
            s = threadIdx.x < (blockDim.x >> 5) ? part[threadIdx.x] : 0.f;
            for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
            if (threadIdx.x == 0) part[0] = s;
        }
        __syncthreads();
        k = scale * part[0];
    }
    float *dst = add ? add + (long long)blockIdx.x * d : 0;
    for (int i = threadIdx.x; i < d; i += blockDim.x) {
        float v = scale != 0.f ? row[i] - k * u[i] : row[i];   // (U is null at scale 0)
        row[i] = v;
        if (dst) dst[i] += v;
    }
}
