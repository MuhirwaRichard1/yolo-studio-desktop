"""Entry point for the frozen build.

One executable serves two roles. Launched normally it opens the GUI; launched
with the worker flag it runs a training or inference job. Bundling a second
executable would mean shipping a second copy of PyTorch, which is most of the
several gigabytes in the package.

The dispatch happens before any application import because
:mod:`yolostudio.worker` redirects ``sys.stdout`` at import time, and the GUI
process must not inherit that.
"""

from __future__ import annotations

import multiprocessing
import sys


def main() -> int:
    # Windows spawns dataloader workers by re-executing this binary. Without
    # this call each child would open its own copy of the GUI.
    multiprocessing.freeze_support()

    argv = sys.argv[1:]
    if argv and argv[0] == "--yolostudio-worker":
        # Hand the worker the argv shape it expects: [prog, config_path].
        sys.argv = [sys.argv[0]] + argv[1:]
        from yolostudio.worker import main as worker_main

        return worker_main()

    from yolostudio.app import main as gui_main

    return gui_main()


if __name__ == "__main__":
    sys.exit(main())
