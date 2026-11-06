"""Background thread that owns an OnlinePipeline so UI loops (vsync rendering) never block on decoding.

Main thread -> worker: start_trial(onset), cancel(). Worker -> main: decisions queue, latest quality.
"""

from __future__ import annotations

import queue
import threading
import time

from neurohack.decoders.stopping import Decision
from neurohack.pipeline import OnlinePipeline
from neurohack.quality import QualityReport


class DecoderWorker:
    def __init__(
        self, pipeline: OnlinePipeline, poll_s: float = 0.01, quality_every_s: float = 0.25
    ) -> None:
        self.pipe = pipeline
        self.poll_s = poll_s
        self.quality_every_s = quality_every_s
        self.decisions: queue.Queue[tuple[Decision, list[dict]]] = queue.Queue()
        self._cmds: queue.Queue[tuple[str, float | None]] = queue.Queue()
        self.quality: QualityReport | None = None
        self.error: BaseException | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self.pipe.source.start()
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)
        self.pipe.source.stop()

    def start_trial(self, onset: float) -> None:
        self._cmds.put(("start", onset))

    def cancel(self) -> None:
        self._cmds.put(("cancel", None))

    def _run(self) -> None:
        last_q = 0.0
        try:
            while not self._stop.is_set():
                while not self._cmds.empty():
                    cmd, arg = self._cmds.get_nowait()
                    if cmd == "start":
                        self.pipe.start_trial(arg)
                    elif cmd == "cancel":
                        self.pipe.trial = None
                self.pipe.pump()
                dec = self.pipe.update()
                if dec is not None:
                    self.decisions.put((dec, self.pipe.last_history))
                now = time.perf_counter()
                if now - last_q >= self.quality_every_s:
                    self.quality = self.pipe.quality()
                    last_q = now
                time.sleep(self.poll_s)
        except BaseException as e:  # surface to UI instead of dying silently
            self.error = e
