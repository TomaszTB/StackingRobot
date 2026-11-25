"""Run logs in the Project 2 format: every joint read to <name>.csv (time_s, row,
joint, expected_deg, measured_deg; blank = failed read) and per-row events to
<name>_events.csv (event, row, line, start_s, duration_s, count, detail):

  move        duration_s includes settling; detail "planned <ms> ms"
  settle      closed loop: count = corrections; detail "settled" or "timed out"
  correction  one per nudge, lasting until the next or the end; detail = joints nudged
  tilt        detail "requested=<deg|any|auto> [chosen=<deg>] achieved=<deg>"
  sequence    rows 1+ in total (row 0, homing, left out); count = corrections
"""

import csv
import time


class FeedbackLog:
    """The controller's on_feedback: writes each joint read, tagged with `row`
    (set it per row), and timings to <stem>_events.csv (see log_move).
    `line` labels a row: run_csv's CSV line, or stack_cups' step.
    """

    def __init__(self, path):
        self.events_path = path.with_name(f"{path.stem}_events.csv")
        self._file = open(path, "x", newline="")  # never overwrite an earlier run
        self._events_file = open(self.events_path, "x", newline="")
        self._writer = csv.writer(self._file)
        self._writer.writerow(["time_s", "row", "joint", "expected_deg", "measured_deg"])
        self._events = csv.writer(self._events_file)
        self._events.writerow(["event", "row", "line", "start_s", "duration_s", "count", "detail"])
        self._start_s = time.perf_counter()
        self.row = 0
        self._sequence_start_s = self._sequence_end_s = None
        self._closed_loop = False
        self._corrections = 0
        self._correction_s = 0.0  # first nudge to settled, summed over rows
        self._settle_s = 0.0

    def _event(self, event, row, line, start_s, duration_s, count="", detail=""):
        self._events.writerow([event, row, line, f"{start_s - self._start_s:.4f}", f"{duration_s:.4f}", count, detail])

    def log_move(self, row, line, planned_ms, timing):
        """Log a row's `move`, and closed loop its `settle` and `correction`s.
        Row 0 (homing) is left out of the `sequence` totals.
        """
        self._event("move", row, line, timing.start_s, timing.end_s - timing.start_s, detail=f"planned {planned_ms} ms")
        if timing.settled is not None:
            self._closed_loop = True
            settle_s = timing.end_s - timing.motion_end_s
            result = "settled" if timing.settled else "timed out"
            self._event("settle", row, line, timing.motion_end_s, settle_s, len(timing.corrections), result)
            ends = [t for t, _ in timing.corrections[1:]] + [timing.end_s]
            for (t, joints), end_s in zip(timing.corrections, ends):
                self._event("correction", row, line, t, end_s - t, detail=" ".join(joints))
            if row:
                self._corrections += len(timing.corrections)
                self._settle_s += settle_s
                if timing.corrections:
                    self._correction_s += timing.end_s - timing.corrections[0][0]
        if row:
            if self._sequence_start_s is None:
                self._sequence_start_s = timing.start_s
            self._sequence_end_s = timing.end_s

    def log_tilt(self, row, line, detail):
        self._event("tilt", row, line, time.perf_counter(), 0.0, detail=detail)

    def __call__(self, joint_name, sample):
        measured = "" if sample.measured_deg is None else f"{sample.measured_deg:.2f}"
        self._writer.writerow(
            [f"{sample.time_s - self._start_s:.4f}", self.row, joint_name, f"{sample.expected_deg:.2f}", measured]
        )

    def close(self):
        """Write the `sequence` event (rows 1+) and close; returns it as text."""
        summary = "no rows run"
        if self._sequence_start_s is not None:
            total_s = self._sequence_end_s - self._sequence_start_s
            summary = f"sequence took {total_s:.2f} s"
            if self._closed_loop:
                detail = f"correcting {self._correction_s:.3f} s, settling {self._settle_s:.3f} s"
                self._event("sequence", "", "", self._sequence_start_s, total_s, self._corrections, detail)
                summary += f", {self._corrections} correction(s) over {self._correction_s:.2f} s, {self._settle_s:.2f} s settling"
            else:
                self._event("sequence", "", "", self._sequence_start_s, total_s)
        self._file.close()
        self._events_file.close()
        return summary
