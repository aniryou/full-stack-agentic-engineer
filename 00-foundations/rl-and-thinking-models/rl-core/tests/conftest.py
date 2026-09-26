import pytest

from rlcore import Policy, SeqTask, pg


@pytest.fixture(scope="session")
def brackets():
    return SeqTask("brackets", 8)


@pytest.fixture(scope="session")
def sft_ref(brackets):
    """A weak SFT model: two steps of maximum likelihood on the 14 balanced strings, from uniform."""
    ref = Policy.for_task(brackets)
    demos = [brackets.as_trajectory(s) for s in brackets.all_sequences() if brackets.verify(s)]
    for _ in range(2):
        pg.sft_step(ref, demos, lr=1.0)
    return ref
