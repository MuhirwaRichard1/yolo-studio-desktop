"""GUI-side driver for :mod:`yolostudio.worker` subprocesses.

Wraps ``QProcess`` so a long job reports progress through Qt signals without
blocking the event loop, and can be cancelled without leaving orphaned CUDA
memory behind.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

from PySide6.QtCore import QObject, QProcess, QProcessEnvironment, Signal

# Flag the frozen executable uses to run as a worker instead of opening the GUI.
WORKER_FLAG = "--yolostudio-worker"


def is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def worker_program() -> str:
    """The executable that runs a job.

    From source that is the Python interpreter. In a frozen build it is the
    sibling console executable: ``sys.executable`` is the windowed GUI binary,
    and a windowed process can have no valid stdout, which is where the worker
    writes its entire event protocol.
    """
    if not is_frozen():
        return sys.executable
    name = "yolostudio-worker.exe" if os.name == "nt" else "yolostudio-worker"
    candidate = Path(sys.executable).parent / name
    return str(candidate) if candidate.exists() else sys.executable


def worker_arguments(config_path: Path) -> list:
    """Arguments to pass to :func:`worker_program`."""
    if is_frozen():
        # The flag is still sent: if the worker binary is missing we fall back
        # to re-launching the GUI executable, which dispatches on it.
        return [WORKER_FLAG, str(config_path)]
    return ["-u", "-m", "yolostudio.worker", str(config_path)]


class JobRunner(QObject):
    """Runs one worker job at a time."""

    started = Signal()
    event = Signal(dict)          # structured protocol message from stdout
    log = Signal(str)             # raw ultralytics console line from stderr
    finished = Signal(bool)       # ok
    failed = Signal(str, str)     # message, hint

    def __init__(self, parent: Optional[QObject] = None):
        super().__init__(parent)
        self._proc: Optional[QProcess] = None
        self._cfg_path: Optional[Path] = None
        self._stdout_buf = ""
        self._stderr_buf = ""
        self._saw_error = False
        self._cancelled = False

    # ------------------------------------------------------------------ state

    @property
    def busy(self) -> bool:
        return self._proc is not None and self._proc.state() != QProcess.NotRunning

    # ------------------------------------------------------------------ start

    def start(self, config: Dict[str, Any], workdir: Optional[Path] = None) -> bool:
        if self.busy:
            return False

        self._stdout_buf = ""
        self._stderr_buf = ""
        self._saw_error = False
        self._cancelled = False

        fd, name = tempfile.mkstemp(prefix="yolostudio-job-", suffix=".json")
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(config, handle, default=str)
        self._cfg_path = Path(name)

        proc = QProcess(self)
        proc.setProgram(worker_program())
        proc.setArguments(worker_arguments(self._cfg_path))
        if os.name == "nt" and is_frozen():
            # The worker is a console binary, so Windows would pop a console
            # window for every job. CREATE_NO_WINDOW suppresses it without
            # touching the inherited stdout/stderr pipes.
            try:
                CREATE_NO_WINDOW = 0x08000000

                def _no_window(args):
                    args.flags |= CREATE_NO_WINDOW

                proc.setCreateProcessArgumentsModifier(_no_window)
            except AttributeError:
                pass  # non-Windows Qt build; nothing to suppress
        if workdir:
            proc.setWorkingDirectory(str(workdir))

        env = QProcessEnvironment.systemEnvironment()
        env.insert("PYTHONUNBUFFERED", "1")
        env.insert("PYTHONIOENCODING", "utf-8")
        if not is_frozen():
            # Running from source: make sure -m can find the package even when
            # the app was launched from an unrelated working directory. A frozen
            # build has its own import machinery and must not be handed a
            # PYTHONPATH that could shadow bundled modules.
            pkg_parent = str(Path(__file__).resolve().parents[2])
            existing = env.value("PYTHONPATH", "")
            env.insert("PYTHONPATH", pkg_parent + (os.pathsep + existing if existing else ""))
        proc.setProcessEnvironment(env)

        proc.readyReadStandardOutput.connect(self._on_stdout)
        proc.readyReadStandardError.connect(self._on_stderr)
        proc.finished.connect(self._on_finished)
        proc.errorOccurred.connect(self._on_proc_error)

        self._proc = proc
        proc.start()
        if not proc.waitForStarted(10000):
            self.failed.emit("Could not start the worker process.",
                             "Tried: " + " ".join([sys.executable]
                                                  + worker_arguments(self._cfg_path)))
            self._cleanup()
            return False
        self.started.emit()
        return True

    # ------------------------------------------------------------------- stop

    def stop(self, wait_ms: int = 4000) -> None:
        """Ask the worker to exit, then insist."""
        if not self.busy or self._proc is None:
            return
        self._cancelled = True
        self.log.emit("\n-- cancelling --\n")
        self._proc.terminate()
        if not self._proc.waitForFinished(wait_ms):
            self._proc.kill()
            self._proc.waitForFinished(2000)

    # ---------------------------------------------------------------- streams

    def _on_stdout(self) -> None:
        if self._proc is None:
            return
        data = bytes(self._proc.readAllStandardOutput()).decode("utf-8", "replace")
        self._stdout_buf += data
        while "\n" in self._stdout_buf:
            line, self._stdout_buf = self._stdout_buf.split("\n", 1)
            line = line.strip()
            if not line:
                continue
            try:
                message = json.loads(line)
            except json.JSONDecodeError:
                self.log.emit(line + "\n")   # not protocol; treat as console noise
                continue
            if message.get("event") == "error":
                self._saw_error = True
                self.failed.emit(message.get("msg", "Job failed."),
                                 message.get("hint", ""))
                trace = message.get("trace")
                if trace:
                    self.log.emit("\n" + trace + "\n")
            self.event.emit(message)

    def _on_stderr(self) -> None:
        if self._proc is None:
            return
        data = bytes(self._proc.readAllStandardError()).decode("utf-8", "replace")
        # Ultralytics redraws progress with \r; make each redraw its own line so
        # the log widget shows progress instead of one ever-growing line.
        self.log.emit(data.replace("\r\n", "\n").replace("\r", "\n"))

    # ------------------------------------------------------------------- exit

    def _on_proc_error(self, error: QProcess.ProcessError) -> None:
        if error == QProcess.Crashed and self._cancelled:
            return
        self._saw_error = True
        self.failed.emit(f"Worker process error: {error.name}", "")

    def _on_finished(self, code: int, status: QProcess.ExitStatus) -> None:
        ok = (code == 0 and status == QProcess.NormalExit and not self._saw_error)
        if self._cancelled:
            ok = False
        self._cleanup()
        self.finished.emit(ok)

    def _cleanup(self) -> None:
        if self._cfg_path and self._cfg_path.exists():
            try:
                self._cfg_path.unlink()
            except OSError:
                pass
        self._cfg_path = None
        if self._proc is not None:
            self._proc.deleteLater()
            self._proc = None
