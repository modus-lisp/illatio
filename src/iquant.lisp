;;;; iquant.lisp — DeepSeek V4 Flash's expert formats, IQ2_XXS and Q2_K, read
;;;; where they lie in the GGUF.
;;;
;;; The routed experts are 78 GB of an 86.7 GB file, more than this machine's
;;; memory, so they are never copied: the whole file is one byte vector
;;; (MAP-FILE-VECTOR) and these kernels read a row's blocks at its offset.
;;; Each is a row against Q8_K activations, as ds4's CPU path computes it
;;; (ds4.c, ds4_vec_dot_iq2_xxs_pair_q8_K and ds4_vec_dot_q2_K_q8_K on
;;; NEON): the integer dots are exact, the floats in its order.
;;;
;;;   IQ2_XXS  66 bytes / 256 weights: d (f16), then 8 sub-blocks of 32, 8
;;;            bytes each -- four grid indices (8 weights each, from
;;;            +IQ2XXS-GRID+), then a little-endian word of four 7-bit sign
;;;            patterns (bits 0-27) and a 4-bit scale LS (28-31).  A weight is
;;;            d * (0.5 + LS) / 4 * grid * sign.
;;;   Q2_K     84 bytes / 256 weights: 16 scale bytes (low nibble a scale,
;;;            high a min, per 16 weights), 64 bytes of 2-bit quants, d, dmin
;;;            (f16).  A weight is d * scale * q - dmin * min.

(in-package #:illatio)

;;; The IQ2_XXS tables, as ggml has them (ds4.c's copy; ggml is MIT).
;;; Each grid entry is 8 weights of magnitude 8, 25 or 43, a byte each,
;;; little-endian; a sign pattern is 7 bits, the 8th sign the parity.
(defparameter +iq2xxs-grid+
  (make-array 256 :element-type '(unsigned-byte 64) :initial-contents
   '(#x0808080808080808 #x080808080808082b #x0808080808081919 #x0808080808082b08
     #x0808080808082b2b #x0808080808190819 #x0808080808191908 #x08080808082b0808
     #x08080808082b082b #x08080808082b2b08 #x08080808082b2b2b #x0808080819080819
     #x0808080819081908 #x0808080819190808 #x0808080819192b08 #x08080808192b0819
     #x08080808192b1908 #x080808082b080808 #x080808082b08082b #x080808082b082b2b
     #x080808082b2b082b #x0808081908080819 #x0808081908081908 #x0808081908190808
     #x0808081908191919 #x0808081919080808 #x080808192b081908 #x080808192b192b08
     #x0808082b08080808 #x0808082b0808082b #x0808082b082b082b #x0808082b2b08082b
     #x0808190808080819 #x0808190808081908 #x0808190808190808 #x08081908082b0819
     #x08081908082b1908 #x0808190819080808 #x080819081908082b #x0808190819082b08
     #x08081908192b0808 #x080819082b080819 #x080819082b081908 #x080819082b190808
     #x080819082b2b1908 #x0808191908080808 #x080819190808082b #x0808191908082b08
     #x08081919082b0808 #x080819191908192b #x08081919192b2b19 #x080819192b080808
     #x080819192b190819 #x0808192b08082b19 #x0808192b08190808 #x0808192b19080808
     #x0808192b2b081908 #x0808192b2b2b1908 #x08082b0808080808 #x08082b0808081919
     #x08082b0808082b08 #x08082b0808191908 #x08082b08082b2b08 #x08082b0819080819
     #x08082b0819081908 #x08082b0819190808 #x08082b081919082b #x08082b082b082b08
     #x08082b1908081908 #x08082b1919080808 #x08082b2b0808082b #x08082b2b08191908
     #x0819080808080819 #x0819080808081908 #x0819080808190808 #x08190808082b0819
     #x0819080819080808 #x08190808192b0808 #x081908082b081908 #x081908082b190808
     #x081908082b191919 #x0819081908080808 #x0819081908082b08 #x08190819082b0808
     #x0819081919190808 #x0819081919192b2b #x081908192b080808 #x0819082b082b1908
     #x0819082b19081919 #x0819190808080808 #x0819190808082b08 #x08191908082b0808
     #x08191908082b1919 #x0819190819082b19 #x081919082b080808 #x0819191908192b08
     #x08191919192b082b #x0819192b08080808 #x0819192b0819192b #x08192b0808080819
     #x08192b0808081908 #x08192b0808190808 #x08192b0819080808 #x08192b082b080819
     #x08192b1908080808 #x08192b1908081919 #x08192b192b2b0808 #x08192b2b19190819
     #x082b080808080808 #x082b08080808082b #x082b080808082b2b #x082b080819081908
     #x082b0808192b0819 #x082b08082b080808 #x082b08082b08082b #x082b0819082b2b19
     #x082b081919082b08 #x082b082b08080808 #x082b082b0808082b #x082b190808080819
     #x082b190808081908 #x082b190808190808 #x082b190819080808 #x082b19081919192b
     #x082b191908080808 #x082b191919080819 #x082b1919192b1908 #x082b192b2b190808
     #x082b2b0808082b08 #x082b2b08082b0808 #x082b2b082b191908 #x082b2b2b19081908
     #x1908080808080819 #x1908080808081908 #x1908080808190808 #x1908080808192b08
     #x19080808082b0819 #x19080808082b1908 #x1908080819080808 #x1908080819082b08
     #x190808081919192b #x19080808192b0808 #x190808082b080819 #x190808082b081908
     #x190808082b190808 #x1908081908080808 #x19080819082b0808 #x19080819192b0819
     #x190808192b080808 #x190808192b081919 #x1908082b08080819 #x1908082b08190808
     #x1908082b19082b08 #x1908082b1919192b #x1908082b192b2b08 #x1908190808080808
     #x1908190808082b08 #x19081908082b0808 #x190819082b080808 #x190819082b192b19
     #x190819190819082b #x19081919082b1908 #x1908192b08080808 #x19082b0808080819
     #x19082b0808081908 #x19082b0808190808 #x19082b0819080808 #x19082b0819081919
     #x19082b1908080808 #x19082b1919192b08 #x19082b19192b0819 #x19082b192b08082b
     #x19082b2b19081919 #x19082b2b2b190808 #x1919080808080808 #x1919080808082b08
     #x1919080808190819 #x1919080808192b19 #x19190808082b0808 #x191908082b080808
     #x191908082b082b08 #x1919081908081908 #x191908191908082b #x191908192b2b1908
     #x1919082b2b190819 #x191919082b190808 #x191919082b19082b #x1919191908082b2b
     #x1919192b08080819 #x1919192b19191908 #x19192b0808080808 #x19192b0808190819
     #x19192b0808192b19 #x19192b08192b1908 #x19192b1919080808 #x19192b2b08082b08
     #x192b080808081908 #x192b080808190808 #x192b080819080808 #x192b0808192b2b08
     #x192b081908080808 #x192b081919191919 #x192b082b08192b08 #x192b082b192b0808
     #x192b190808080808 #x192b190808081919 #x192b191908190808 #x192b19190819082b
     #x192b19192b081908 #x192b2b081908082b #x2b08080808080808 #x2b0808080808082b
     #x2b08080808082b2b #x2b08080819080819 #x2b0808082b08082b #x2b08081908081908
     #x2b08081908192b08 #x2b08081919080808 #x2b08082b08190819 #x2b08190808080819
     #x2b08190808081908 #x2b08190808190808 #x2b08190808191919 #x2b08190819080808
     #x2b081908192b0808 #x2b08191908080808 #x2b0819191908192b #x2b0819192b191908
     #x2b08192b08082b19 #x2b08192b19080808 #x2b08192b192b0808 #x2b082b080808082b
     #x2b082b1908081908 #x2b082b2b08190819 #x2b19080808081908 #x2b19080808190808
     #x2b190808082b1908 #x2b19080819080808 #x2b1908082b2b0819 #x2b1908190819192b
     #x2b1908192b080808 #x2b19082b19081919 #x2b19190808080808 #x2b191908082b082b
     #x2b19190819081908 #x2b19191919190819 #x2b192b082b080819 #x2b192b19082b0808
     #x2b2b08080808082b #x2b2b080819190808 #x2b2b08082b081919 #x2b2b081908082b19
     #x2b2b082b08080808 #x2b2b190808192b08 #x2b2b2b0819190808 #x2b2b2b1908081908)))
(defparameter +ksigns-iq2xs+
  (make-array 128 :element-type '(unsigned-byte 8) :initial-contents
   '(  0 129 130   3 132   5   6 135 136   9  10 139  12 141 142  15
     144  17  18 147  20 149 150  23  24 153 154  27 156  29  30 159
     160  33  34 163  36 165 166  39  40 169 170  43 172  45  46 175
      48 177 178  51 180  53  54 183 184  57  58 187  60 189 190  63
     192  65  66 195  68 197 198  71  72 201 202  75 204  77  78 207
      80 209 210  83 212  85  86 215 216  89  90 219  92 221 222  95
      96 225 226  99 228 101 102 231 232 105 106 235 108 237 238 111
     240 113 114 243 116 245 246 119 120 249 250 123 252 125 126 255)))

(defparameter +iq2xxs-signs+
  (let ((v (make-array (* 128 8) :element-type '(signed-byte 8))))
    (dotimes (s 128 v)
      (dotimes (j 8)
        (setf (aref v (+ (* s 8) j)) (if (logbitp j (aref +ksigns-iq2xs+ s)) -1 1)))))
  "Pattern S's eight signs, at S*8.")

(defparameter +iq2xxs-grid-bytes+
  (let ((v (make-array (* 256 8) :element-type '(unsigned-byte 8))))
    (dotimes (g 256 v)
      (dotimes (j 8)
        (setf (aref v (+ (* g 8) j)) (ldb (byte 8 (* 8 j)) (aref +iq2xxs-grid+ g))))))
  "Grid entry G's eight magnitudes, at G*8.")

(declaim (inline u8-ref u16-le u32-le f16-at))
(defun u8-ref (w i) (declare (type u8vec w) (type fixnum i)) (aref w i))
(defun u16-le (w i) (declare (type u8vec w) (type fixnum i)) (logior (aref w i) (ash (aref w (1+ i)) 8)))
(defun u32-le (w i)
  (declare (type u8vec w) (type fixnum i))
  (logior (aref w i) (ash (aref w (+ i 1)) 8) (ash (aref w (+ i 2)) 16) (ash (aref w (+ i 3)) 24)))
(defun f16-at (w i) (half-to-single (u16-le w i)))

;;; --- IQ2_XXS ---------------------------------------------------------------------

(defun iq2xxs-row-dot (w off nb xv xo)
  "The row of NB blocks at byte OFF of W against the Q8_K activations XV at
   XO.  Per block: the even sub-blocks' and the odd ones' scaled integer dots
   summed apart, d * (their sum) added up, and the total quartered -- ds4's
   NEON order."
  (declare (type u8vec w) (type fixnum off nb xo) (type q8kvec xv)
           (optimize speed (safety 0)))
  (let ((qs (q8kvec-qs xv)) (yd (q8kvec-d xv))
        (grid +iq2xxs-grid-bytes+) (signs +iq2xxs-signs+)
        (total 0f0) (y0 (floor xo +qk-k+)))
    (declare (type i8vec qs) (type f32vec yd) (type single-float total) (type fixnum y0)
             (type (simple-array (unsigned-byte 8) (*)) grid) (type (simple-array (signed-byte 8) (*)) signs))
    (dotimes (b nb (* 0.25f0 total))
      (let* ((bo (+ off (* b 66)))
             (d (* (f16-at w bo) (aref yd (+ y0 b))))
             (suma 0f0) (sumb 0f0))
        (declare (type fixnum bo) (type single-float d suma sumb))
        (dotimes (ib 8)
          (let* ((so (+ bo 2 (* ib 8)))
                 (aux (u32-le w (+ so 4)))
                 (xb (+ xo (* b +qk-k+) (* ib 32)))
                 (isum 0))
            (declare (type fixnum so xb) (type (unsigned-byte 32) aux) (type (signed-byte 32) isum))
            (dotimes (l 4)
              (let ((g (* 8 (aref w (+ so l))))
                    (s (* 8 (ldb (byte 7 (* 7 l)) aux))))
                (declare (type fixnum g s))
                (dotimes (j 8)
                  (setf isum (the (signed-byte 32)
                                  (+ isum (* (aref grid (+ g j)) (aref signs (+ s j))
                                             (i8-ref qs (+ xb (* l 8) j)))))))))
            (let ((term (* (float isum 1f0) (+ 0.5f0 (float (ash aux -28) 1f0)))))
              (if (evenp ib) (incf suma term) (incf sumb term)))))
        (incf total (* d (+ suma sumb)))))))

(defun iq2xxs-row-floats! (w off ncols out)
  "The row at byte OFF of W (NCOLS weights) as floats into OUT, as ggml's
   dequantize_row_iq2_xxs: d * (0.5 + LS) * 0.25 * grid * sign."
  (declare (type u8vec w) (type fixnum off ncols) (type f32vec out))
  (dotimes (b (floor ncols +qk-k+) out)
    (let* ((bo (+ off (* b 66))) (d (f16-at w bo)))
      (dotimes (ib 8)
        (let* ((so (+ bo 2 (* ib 8))) (aux (u32-le w (+ so 4)))
               (db (* d (+ 0.5f0 (float (ash aux -28) 1f0)) 0.25f0)))
          (dotimes (l 4)
            (let ((g (* 8 (aref w (+ so l)))) (s (* 8 (ldb (byte 7 (* 7 l)) aux))))
              (dotimes (j 8)
                (setf (aref out (+ (* b 256) (* ib 32) (* l 8) j))
                      (* db (float (aref +iq2xxs-grid-bytes+ (+ g j)) 1f0)
                         (float (aref +iq2xxs-signs+ (+ s j)) 1f0)))))))))))

;;; --- Q2_K ------------------------------------------------------------------------

(defun q2k-row-dot (w off nb xv xo)
  "The row of NB blocks at byte OFF of W against the Q8_K activations XV at
   XO.  Per block, as ds4's NEON path: SUM += (-y.d * dmin) * (the mins
   against the 16-sums), then SUM += (y.d * d) * (the scaled integer dot)."
  (declare (type u8vec w) (type fixnum off nb xo) (type q8kvec xv)
           (optimize speed (safety 0)))
  (let ((qs (q8kvec-qs xv)) (yd (q8kvec-d xv)) (bs (q8kvec-bsums xv))
        (sum 0f0) (y0 (floor xo +qk-k+)))
    (declare (type i8vec qs) (type f32vec yd) (type i32vec bs) (type single-float sum) (type fixnum y0))
    (dotimes (b nb sum)
      (let* ((bo (+ off (* b 84)))
             (ydb (aref yd (+ y0 b)))
             (d (* ydb (f16-at w (+ bo 80))))
             (dmin (* (- ydb) (f16-at w (+ bo 82))))
             (mins 0) (isum 0)
             (xb (+ xo (* b +qk-k+))))
        (declare (type fixnum bo xb) (type single-float ydb d dmin)
                 (type (signed-byte 32) mins isum))
        (dotimes (j 16)
          (setf mins (the (signed-byte 32) (+ mins (* (ash (aref w (+ bo j)) -4)
                                                      (aref bs (+ (* (+ y0 b) 16) j)))))))
        (incf sum (* dmin (float mins 1f0)))
        ;; two halves of 128; each byte holds four weights, 32 apart
        (dotimes (h 2)
          (dotimes (k 4)
            (dotimes (half 2)
              (let ((sc (logand (aref w (+ bo (* h 8) (* 2 k) half)) 15))
                    (qo (+ bo 16 (* h 32) (* half 16)))
                    (xo2 (+ xb (* h 128) (* k 32) (* half 16)))
                    (dot 0))
                (declare (type fixnum qo xo2) (type (signed-byte 32) dot))
                (dotimes (l 16)
                  (setf dot (the (signed-byte 32)
                                 (+ dot (* (ldb (byte 2 (* 2 k)) (aref w (+ qo l)))
                                           (i8-ref qs (+ xo2 l)))))))
                (setf isum (the (signed-byte 32) (+ isum (* sc dot))))))))
        (incf sum (* d (float isum 1f0)))))))

(defun q2k-row-floats! (w off ncols out)
  "The row at byte OFF of W as floats into OUT, as ggml's dequantize_row_q2_K."
  (declare (type u8vec w) (type fixnum off ncols) (type f32vec out))
  (dotimes (b (floor ncols +qk-k+) out)
    (let* ((bo (+ off (* b 84))) (d (f16-at w (+ bo 80))) (dmin (f16-at w (+ bo 82))))
      (dotimes (h 2)
        (dotimes (k 4)
          (dotimes (half 2)
            (let* ((sb (aref w (+ bo (* h 8) (* 2 k) half)))
                   (dl (* d (float (logand sb 15) 1f0))) (ml (* dmin (float (ash sb -4) 1f0))))
              (dotimes (l 16)
                (setf (aref out (+ (* b 256) (* h 128) (* k 32) (* half 16) l))
                      (- (* dl (float (ldb (byte 2 (* 2 k)) (aref w (+ bo 16 (* h 32) (* half 16) l))) 1f0))
                         ml))))))))))
