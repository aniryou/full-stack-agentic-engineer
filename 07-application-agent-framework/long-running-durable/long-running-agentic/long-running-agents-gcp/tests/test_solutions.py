"""The practice-notebook checks, run against (a) the reference solutions and (b) the library."""

import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "notebooks"))

from lragents import practice_checks as pc
from lragents.adk.nightly_workflow import build_workflow
from lragents.core import run_idempotent
from lragents.patterns import FanOutFanIn, approve
from solutions import ex1_durable_loop, ex2_fan_out_fan_in, ex3_hitl_saga, ex4_adk_workflow


def test_ex1_solutions():
    assert pc.check_run_idempotent(ex1_durable_loop.run_idempotent)
    assert pc.check_mini_loop(ex1_durable_loop.MiniDurableLoop)
    assert pc.check_run_idempotent(run_idempotent)


def test_ex2_solutions():
    assert pc.check_fan_in_mutate(ex2_fan_out_fan_in.make_mutate)
    assert pc.check_fan_out_fan_in(ex2_fan_out_fan_in.MyFan)
    assert pc.check_fan_out_fan_in(FanOutFanIn)


def test_ex3_solutions():
    assert pc.check_approve(ex3_hitl_saga.approve)
    assert pc.check_approve(approve)
    assert pc.check_saga_next(ex3_hitl_saga.saga_next)


def test_ex4_solutions():
    assert pc.check_adk_workflow(ex4_adk_workflow.build_workflow)
    assert pc.check_adk_workflow(build_workflow)
