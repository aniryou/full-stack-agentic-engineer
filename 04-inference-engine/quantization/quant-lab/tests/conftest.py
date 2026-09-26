"""Small matrices: keep BLAS single-threaded (set before numpy is imported anywhere)."""
import os

for v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "1")
