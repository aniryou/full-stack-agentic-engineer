"""kerncore: the three kernel topics of layer 04 in numpy, so each runs and is tested at T0.

  kerncore.kv     KV-cache sizing (the kv-cache primer's numbers) and a tiny decoder whose cached decode
                  gives the same logits as recomputing the prefix
  kerncore.paged  a block pool with refcounts and copy-on-write, and attention through a block table
                  (the algorithm of ../paged-attention/paged_attention_minimal.py)
  kerncore.flash  tiled attention with the online softmax, causal tile skipping and counters checked
                  against ../flash-attention/fa_calculators.py (imported from the repo, not copied)

Standard library + numpy. No GPU, no torch.
"""
from . import kv, paged

__all__ = ["kv", "paged", "flash"]
__version__ = "0.1.0"


def __getattr__(name):          # kerncore.flash loads fa_calculators.py from the checkout; import it on use
    if name == "flash":
        import importlib
        return importlib.import_module(".flash", __name__)
    raise AttributeError(name)
