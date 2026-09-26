"""rlcore — reinforcement learning and thinking models on toy tasks, standard library + numpy.

Read the modules in this order; each opens with the one idea it teaches:
tasks → policy → pg → pref → grpo → ttc → workload.
"""
from . import grpo, pg, pref, ttc, workload
from .policy import Policy, softmax
from .tasks import ANSWER, THINK, SeqTask, ThinkTask, Trajectory

__all__ = ["ANSWER", "THINK", "Policy", "SeqTask", "ThinkTask", "Trajectory", "grpo", "pg", "pref", "softmax",
           "ttc", "workload"]
