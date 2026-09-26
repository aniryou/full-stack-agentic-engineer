"""``python -m moelab.tinymoe``: train the tiny MoE (needs torch; the bundled curves do not)."""
import sys

from .. import env

try:
    env.torch()                        # honours MOELAB_NO_TORCH and gives a clear message without torch
except ImportError as e:
    sys.exit(f"{e}\nWithout torch, read the recorded runs instead: moelab.tinymoe.load_bundled() "
             "(fixtures/tinymoe_curves.json).")

from .train import main  # noqa: E402

raise SystemExit(main())
