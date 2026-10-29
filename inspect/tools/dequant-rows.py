# Dequantize rows of a GGUF tensor with ggml's own Python (gguf-py), as an
# oracle for illatio's kernels.  usage: deq.py FILE OFFSET TYPE NCOLS ROW0 NROWS OUT
import sys, types, importlib.util
import numpy as np
import os
root = os.path.join(os.environ.get('GGUF_PY', os.path.expanduser('~/workspace/llama.cpp/gguf-py')), 'gguf') + '/'
pkg = types.ModuleType('gguf'); pkg.__path__ = [root]; sys.modules['gguf'] = pkg
def load(name):
    spec = importlib.util.spec_from_file_location('gguf.' + name, root + name + '.py')
    m = importlib.util.module_from_spec(spec); sys.modules['gguf.' + name] = m; spec.loader.exec_module(m); return m
constants = load('constants'); load('lazy'); quants = load('quants')
path, off, tname, ncols, row0, nrows, out = sys.argv[1], int(sys.argv[2]), sys.argv[3], int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6]), sys.argv[7]
qt = getattr(constants.GGMLQuantizationType, tname)
bsize, tsize = constants.GGML_QUANT_SIZES[qt]
row_bytes = ncols // bsize * tsize
with open(path, 'rb') as f:
    f.seek(off + row0 * row_bytes)
    raw = np.frombuffer(f.read(nrows * row_bytes), dtype=np.uint8).reshape(nrows, row_bytes)
vals = quants.dequantize(raw, qt).astype(np.float32)
vals.tofile(out)
print(vals.shape, float(vals[0, :4].sum()))
