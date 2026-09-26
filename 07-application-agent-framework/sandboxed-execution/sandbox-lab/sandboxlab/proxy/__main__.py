"""``python3 -m sandboxlab.proxy --config proxy.json`` — run the egress proxy (see server.py)."""
from .server import main

raise SystemExit(main())
