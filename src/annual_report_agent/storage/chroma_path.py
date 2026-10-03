"""Give Windows native HNSW code an ASCII alias without moving the data.

Chroma 1.5.9's native HNSW persistence loses binary files on this machine when
the absolute path contains Chinese characters. SQLite itself is unaffected.
A checked directory junction lets both stores retain the same on-disk location.
"""

import base64
import hashlib
import os
import subprocess
from pathlib import Path


def native_chroma_path(path: Path) -> str:
    target = Path(path).resolve()
    target.mkdir(parents=True, exist_ok=True)
    if os.name != "nt" or str(target).isascii():
        return str(target)
    # Keep the junction on the same volume: cross-volume reparse points fail on
    # some Windows mapped-volume setups even when creation reports success.
    ancestor = target.parent
    while not str(ancestor).isascii() and ancestor != ancestor.parent:
        ancestor = ancestor.parent
    if not str(ancestor).isascii():
        raise ValueError("CHROMA_PATH_UNSUPPORTED: 请将 ARA_DATA_DIR 设置为纯英文绝对路径。")
    aliases = ancestor / ".annual-report-agent" / "chroma-paths"
    aliases.mkdir(parents=True, exist_ok=True)
    identity = hashlib.sha256(str(target).casefold().encode("utf-8")).hexdigest()[:32]
    alias = aliases / identity
    if not os.path.lexists(alias):
        # Arguments are PowerShell single-quoted literals, then UTF-16 encoded.
        # No user path is interpolated as executable shell syntax.
        def literal(value):
            return "'" + str(value).replace("'", "''") + "'"
        script = ("$ErrorActionPreference='Stop'; New-Item -ItemType Junction -Path "
                  + literal(alias) + " -Target " + literal(target) + " | Out-Null")
        encoded = base64.b64encode(script.encode("utf-16-le")).decode("ascii")
        subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-EncodedCommand", encoded],
                       capture_output=True, check=True, timeout=30,
                       creationflags=subprocess.CREATE_NO_WINDOW)
    if not alias.is_dir() or not os.path.samefile(alias, target):
        raise ValueError("CHROMA_PATH_CONFLICT: 本地索引路径别名指向其他目录，未覆盖。")
    return str(alias)
