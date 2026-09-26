"""The execution contract: one exit reason per budget, output truncation, a clean environment, and the
07.1 tool-result shape. Runs real processes on this machine (a few seconds)."""
import os
import signal
import sys

import pytest

from sandboxlab import env
from sandboxlab.process import Budgets, ProcessSandbox, Unsandboxed, parse_wrapper_output
from sandboxlab.wrapper import MARKER, classify

LINUX = sys.platform.startswith("linux")
B = Budgets(cpu_s=1, wall_s=2)


@pytest.fixture(scope="module")
def sb():
    return ProcessSandbox(B, netns=False)


def test_classify_is_pure_and_ordered():
    assert classify(0, None, "", 0.1, 1) == "ok"
    assert classify(1, None, "Traceback\nZeroDivisionError", 0.1, 1) == "error"
    assert classify(-signal.SIGXCPU, None, "", 1.0, 1) == "cpu_time"
    assert classify(-signal.SIGKILL, None, "", 1.99, 2) == "cpu_time"          # the hard limit, one second later
    assert classify(-signal.SIGKILL, None, "", 0.1, 2) == "killed"              # not ours
    assert classify(1, None, "MemoryError", 0.1, 1) == "memory"
    assert classify(1, None, "OSError: [Errno 27] File too large", 0.1, 1) == "file_too_large"
    assert classify(-25, None, "", 0.1, 1) == "file_too_large"                  # SIGXFSZ: a non-Python child
    assert classify(None, "wall_timeout", "", None, 1) == "wall_timeout"                  # a preset reason wins


def test_parse_wrapper_output_finds_the_last_marker_line():
    text = f"noise\n{MARKER} {{\"exit_reason\": \"ok\"}}\n"
    assert parse_wrapper_output(text) == {"exit_reason": "ok"}
    assert parse_wrapper_output("no marker here") is None


@pytest.mark.skipif(not LINUX, reason="rlimit behaviour is Linux-specific (macOS does not enforce RLIMIT_AS)")
@pytest.mark.parametrize("code,reason", [
    ("print('hi')", "ok"),
    ("1/0", "error"),
    ("while True: pass", "cpu_time"),
    ("import time; time.sleep(60)", "wall_timeout"),
    ("x = bytearray(600 * 2**20)", "memory"),
    ("open('big', 'wb').write(b'0' * (9 * 2**20))", "file_too_large"),
    ("import sys; sys.stdout.write('A' * (2 * 2**20))", "output_limit"),
])
def test_each_budget_has_its_exit_reason(sb, code, reason):
    r = sb.run(code)
    assert r.exit_reason == reason, (r.exit_reason, r.stderr[-300:])
    assert r.wall_s < B.wall_s + 1.5


@pytest.mark.skipif(not LINUX, reason="Linux")
def test_output_is_truncated_and_counted(sb):
    r = sb.run("import sys; sys.stdout.write('B' * 100000)")
    assert r.exit_reason == "ok" and r.stdout_truncated and len(r.stdout) == 65536 and r.stdout_bytes == 100000


def test_clean_environment_and_workspace(sb, monkeypatch):
    monkeypatch.setenv("LAB_TEST_SECRET", "s3cr3t")
    r = sb.run("import os, json; print(json.dumps(sorted(os.environ))); print(os.getcwd())")
    names = r.stdout.splitlines()[0]
    assert "LAB_TEST_SECRET" not in names and "HOME" in names
    assert "sbx-ws-" in r.stdout.splitlines()[1]
    leaky = Unsandboxed(harness_timeout_s=3).run("import os; print(os.environ.get('LAB_TEST_SECRET'))")
    assert leaky.stdout.strip() == "s3cr3t"                                     # the contrast


def test_tool_result_shape_matches_agent_core():
    ok = ProcessSandbox(B, netns=False).run("print(2 + 2)").to_tool_result()
    assert ok["ok"] is True and ok["data"]["stdout"] == "4\n" and ok["data"]["exit_reason"] == "ok"
    bad = ProcessSandbox(B, netns=False).run("1/0").to_tool_result()
    assert bad["ok"] is False and bad["error"] == "error" and "ZeroDivisionError" in bad["data"]["stderr"]
    assert {"message", "hint"} <= set(bad)


@pytest.mark.skipif(not (LINUX and env.is_root()), reason="switching to a dedicated UID needs root")
def test_dedicated_uid_and_nproc():
    s = ProcessSandbox(Budgets(cpu_s=2, wall_s=4, pids=8))
    assert s.drop_uid and 61000 <= s.uid < 62000
    r = s.run("import os\nn = 0\ntry:\n    while n < 40:\n        if os.fork() == 0:\n            import time; time.sleep(1); os._exit(0)\n"
              "        n += 1\nexcept OSError:\n    pass\nprint(os.getuid(), n)")
    uid, forked = map(int, r.stdout.split())
    assert uid == s.uid and forked < 40


@pytest.mark.skipif(not env.netns_mode(), reason="no network namespace on this machine")
def test_netns_blocks_the_network():
    r = ProcessSandbox(B, netns=True).run(
        "import socket\ntry:\n    socket.create_connection(('127.0.0.1', 9), timeout=1); print('connected')\n"
        "except OSError as e:\n    print(type(e).__name__)")
    assert r.exit_reason == "ok" and "connected" not in r.stdout
