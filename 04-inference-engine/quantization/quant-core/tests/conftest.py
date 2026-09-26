"""Pin BLAS to one thread before numpy loads: quantcore's matrices are small, and on a shared or busy machine a
multi-threaded OpenBLAS can make a 256x256 inverse ~100x slower than one thread does."""
import os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
