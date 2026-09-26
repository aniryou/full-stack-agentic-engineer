"""Tiny matrices: one BLAS thread is faster than many, and steadier on a shared CPU. Set before numpy loads."""
import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
