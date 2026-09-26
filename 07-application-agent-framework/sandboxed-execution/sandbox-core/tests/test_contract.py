"""The execution contract: budgets, idempotency, run-once, and the tool-result shape."""
from sandboxcore import Budgets, ExecutionResult, ResultStore, Usage, digest, idempotency_key
from sandboxcore.contract import HINTS


def test_digest_is_canonical_and_stable():
    # order of keys must not change the hash (sort_keys) — hand-checked against sha256(canonical)[:16]
    assert digest({"a": 1, "b": 2}) == digest({"b": 2, "a": 1})
    import hashlib
    canonical = '{"a":1,"b":2}'
    assert digest({"a": 1, "b": 2}) == hashlib.sha256(canonical.encode()).hexdigest()[:16]


def test_idempotency_key_recipe():
    # scaling primer §5.4: turn, step, call index, args hash
    k = idempotency_key("turn7", 3, 1, {"code": "x"})
    assert k.startswith("turn7:3:1:")
    assert k == f"turn7:3:1:{digest({'code': 'x'})}"
    # same args -> same key; different args -> different key
    assert idempotency_key("t", 0, 0, {"a": 1}) != idempotency_key("t", 0, 0, {"a": 2})


def test_budgets_exceeding_is_field_wise():
    cap = Budgets(cpu_s=2, memory_mb=256, pids=16)
    assert Budgets(cpu_s=1, memory_mb=128).exceeding(cap) == []
    over = Budgets(cpu_s=5, memory_mb=1024, pids=64).exceeding(cap)
    assert set(over) >= {"cpu_s", "memory_mb", "pids"}


def test_result_store_runs_at_most_once():
    store = ResultStore()
    calls = {"n": 0}

    def run():
        calls["n"] += 1
        return ExecutionResult("ok", 0, "hi", "", False, Usage())

    r1, replayed1 = store.run_once("k", run)
    r2, replayed2 = store.run_once("k", run)
    assert calls["n"] == 1                 # the second delivery did NOT run the code
    assert replayed1 is False and replayed2 is True
    assert r1 is r2


def test_tool_result_shape_matches_agent_core():
    ok = ExecutionResult("ok", 0, "out", "", False, Usage()).as_tool_result()
    assert ok["ok"] is True and ok["data"]["stdout"] == "out"
    bad = ExecutionResult("cpu_time", -9, "", "boom", True, Usage()).as_tool_result()
    assert bad["ok"] is False and bad["error"] == "cpu_time"
    assert bad["hint"] == HINTS["cpu_time"]


def test_over_budget_makes_result_not_ok():
    r = ExecutionResult("ok", 0, "", "", False, Usage(), over_budget=["disk_mb"])
    assert r.ok is False and r.as_tool_result()["ok"] is False
