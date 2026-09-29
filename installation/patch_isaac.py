"""Apply the known Isaac 5.1 debug-draw library rename, idempotently.

Upstream issue: https://github.com/isaac-sim/IsaacSim/issues/303
"""

import sys
from pathlib import Path

root = Path(sys.argv[1])
for extension in ("isaacsim.asset.gen.omap", "isaacsim.sensors.physx"):
    path = root / "source" / "extensions" / extension / "premake5.lua"
    text = path.read_text()
    old = '"isaacsim.util.debug_draw.primitive_drawing"'
    new = '"isaacsim.util.debug_draw.plugin"'
    if old in text:
        assert text.count(old) == 1, f"Unexpected source contents: {path}"
        path.write_text(text.replace(old, new))
    elif new not in text:
        raise RuntimeError(f"Unrecognized debug-draw dependency in {path}")
