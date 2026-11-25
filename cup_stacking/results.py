"""The results sheet (analysis/project_3_results.csv): one row per run, asked for at its end.

Columns follow analysis/project_2_results.csv, with the gripper angle instead of the base
time, plus the stack spot and the run's log folder.
"""

import csv
import json
import re
from pathlib import Path

from .layout import ROOT

RESULTS_PATH = ROOT / "analysis" / "project_3_results.csv"
COLUMNS = ["Run", "Tilt", "Stack x", "Stack y", "Upstack Pass/Fail", "Downstack Pass/Fail", "Fail Action",
           "Failure Reason", "Log"]
ACTIONS = ("pickup", "transport", "placement")


def action_of(step: str) -> str:
    """The sheet's action for a logged step (e.g. "place descend" -> "placement")."""
    if step.startswith("pick") or step == "grip":
        return "pickup"
    if step == "place travel":
        return "transport"
    if step.startswith("place") or step == "release":
        return "placement"
    return step


def where_ended(context: str, step: str) -> str:
    """The run's last step as the sheet would name it, e.g. "down 2 placement"."""
    match = re.fullmatch(r"(up|down)(\d+)", context)
    return f"{match[1]} {match[2]} {action_of(step)}" if match else ""


def parse_failure(text: str, ended_phase: str) -> tuple:
    """(phase, fail action) from e.g. "down 2 pickup", "u3 placement" or "cup 4 pickup"
    (phase: `ended_phase`)."""
    match = re.fullmatch(r"\s*(up|down|u|d)?\s*(?:cup\s*)?(\d+)?\s*(.*?)\s*", text, re.IGNORECASE)
    phase = {"u": "up", "d": "down"}.get((match[1] or "").lower(), (match[1] or ended_phase).lower())
    number, rest = match[2], match[3]
    action = f"Cup {number} {rest}".strip() if number else rest or text.strip()
    return phase, action


def ask(session: Path, summary: dict, phases: tuple, context: str, step: str):
    """Ask where the run failed (blank: passed) and why; add the answers to run.json and a row
    to RESULTS_PATH. Returns quietly if the questions are interrupted."""
    ended = where_ended(context, step)
    print("\nRecording this run's result" + (f" (it ended at: {ended})" if ended else "")
          + (f"; stopped: {summary['stopped']}" if summary.get("stopped") else ""))
    try:
        where = input("  Where did it fail? e.g. 'down 2 pickup' (blank = passed): ").strip()
        reason = input("  Failure reason: ").strip() if where else ""
    except (EOFError, KeyboardInterrupt):
        print("\n  Not recorded.")
        return
    up, down = ("pass" if "up" in phases else "N/A"), ("pass" if "down" in phases else "N/A")
    action = ""
    if where:
        phase, action = parse_failure(where, ended.split()[0] if ended else phases[-1])
        if phase == "up":
            up, down = "fail", "N/A"
        else:
            down = "fail"
    result = {"upstack": up, "downstack": down, "fail_action": action, "failure_reason": reason}
    summary["result"] = result
    (session / "run.json").write_text(json.dumps(summary, indent=2) + "\n")

    rows = list(csv.reader(RESULTS_PATH.open())) if RESULTS_PATH.exists() else []
    stack_xy = summary.get("stack_home_xy") or ["", ""]
    row = [len(rows) if rows else 1, summary["tilt"], *stack_xy, up, down, action, reason,
           str(session.relative_to(ROOT))]
    with RESULTS_PATH.open("a", newline="") as f:
        writer = csv.writer(f)
        if not rows:
            writer.writerow(COLUMNS)
        writer.writerow(row)
    print(f"  Recorded run {row[0]} in {RESULTS_PATH.relative_to(ROOT)}: upstack {up}, downstack {down}"
          + (f", {action} ({reason})" if action else ""))
