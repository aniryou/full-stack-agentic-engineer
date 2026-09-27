"""distillcore — distillation small enough to compute exactly, standard library + numpy.

Read the modules in this order; each opens with the one idea it teaches:
tasks → losses → tinylm → divergences → seqkd → onpolicy → reasoning → draft → eval → cost.
The T0 core of 00-foundations/distillation (module 00.6); the topic's PRIMER.md names the function behind
every number it quotes.
"""
from . import cost, divergences, draft, eval, losses, onpolicy, reasoning, seqkd
from .tasks import ModLang, ThinkToy
from .tinylm import TinyLM, fit_language, train

__all__ = ["ModLang", "ThinkToy", "TinyLM", "cost", "divergences", "draft", "eval", "fit_language", "losses",
           "onpolicy", "reasoning", "seqkd", "train"]
