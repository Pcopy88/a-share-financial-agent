"""Cross-process ownership of one mode's runtime data directory."""

import os
from pathlib import Path


class InstanceAlreadyRunning(ValueError):
    code = "INSTANCE_ALREADY_RUNNING"
    message = "此数据目录已有运行实例，请先停止 API 或 CLI 任务。"

    def __init__(self):
        super().__init__(f"{self.code}: {self.message}")


class RuntimeLock:
    def __init__(self, runtime_dir):
        self.path = Path(runtime_dir) / ".runtime.lock"
        self._handle = None

    def acquire(self):
        if self._handle is not None:
            return self
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = self.path.open("a+b")
        try:
            if handle.seek(0, 2) == 0:
                handle.write(b"0")
                handle.flush()
            handle.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            handle.close()
            raise InstanceAlreadyRunning() from error
        self._handle = handle
        return self

    def release(self):
        if self._handle is not None:
            # Retain the file so concurrent opens always lock the same inode.
            self._handle.close()
            self._handle = None

    def __enter__(self):
        return self.acquire()

    def __exit__(self, *_):
        self.release()
