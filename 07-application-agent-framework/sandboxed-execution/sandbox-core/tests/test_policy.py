"""Policy as data: deny-by-default, the budget ceiling, egress allowlist, and clamping."""
from sandboxcore import Budgets, Effect, ExecutionRequest, SandboxPolicy


def test_deny_by_default_on_principal():
    p = SandboxPolicy(allowed_principals=frozenset({"agent:trusted"}))
    assert p.evaluate(ExecutionRequest("print(1)", principal="agent:other")).effect is Effect.DENY
    assert p.evaluate(ExecutionRequest("print(1)", principal="agent:trusted")).allowed


def test_budget_ceiling_denies_a_too_large_request():
    p = SandboxPolicy(max_budgets=Budgets(cpu_s=2, memory_mb=256))
    over = ExecutionRequest("print(1)", budgets=Budgets(cpu_s=60, memory_mb=8000))
    d = p.evaluate(over)
    assert d.effect is Effect.DENY and "ceiling" in d.reasons[0]


def test_egress_must_be_on_the_allowlist():
    p = SandboxPolicy(egress_allowlist=("api.github.com",))
    assert p.evaluate(ExecutionRequest("x", egress=("api.github.com",))).allowed
    d = p.evaluate(ExecutionRequest("x", egress=("evil.example",)))
    assert d.effect is Effect.DENY


def test_egress_can_require_confirmation():
    p = SandboxPolicy(egress_allowlist=("api.github.com",), egress_needs_confirm=True)
    assert p.evaluate(ExecutionRequest("x", egress=("api.github.com",))).effect is Effect.CONFIRM
    assert p.evaluate(ExecutionRequest("x")).allowed        # no egress requested -> allowed


def test_clamp_lowers_but_never_raises():
    p = SandboxPolicy(max_budgets=Budgets(cpu_s=2, memory_mb=256, output_bytes=8192))
    req = p.clamp(ExecutionRequest("x", budgets=Budgets(cpu_s=99, memory_mb=4096, output_bytes=1)))
    assert req.budgets.cpu_s == 2 and req.budgets.memory_mb == 256   # both clamped down to the ceiling
    assert req.budgets.output_bytes == 1        # a request may ask for LESS than the ceiling
