"""moecore: mixture-of-experts from first principles, in numpy.

Modules, in reading order: moe (the layer and parameter counting), routing (load balance),
train (collapse and its fixes, hand-written gradients), touched (which experts a batch reads,
and the decode roofline), ep (expert parallelism), sizing (memory, GPUs and cost).
"""
from . import ep, moe, routing, sizing, touched, train
from .moe import MoEConfig, MoELayer, route

__all__ = ["ep", "moe", "routing", "sizing", "touched", "train", "MoEConfig", "MoELayer", "route"]
