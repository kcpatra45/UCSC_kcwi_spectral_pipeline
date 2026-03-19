from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Any

from .state import PipelineState, load_state, save_state


@dataclass
class Step:
    step_id: str
    description: str
    runner: Callable[[Any], None]  # runner receives a context


class PipelineContext:
    def __init__(self, cfg, outdir: Path, *, only_objects=None, skip_objects=None, redo_objects=None):
        self.cfg = cfg
        self.outdir = outdir
        self.config_path = outdir / "config.json"
        self.state_path = outdir / "state.json"
        self.state = load_state(self.state_path)

        # subdirs (created by step00)
        self.apdir = outdir / "apertures"
        self.caldir = outdir / "calibration"
        self.countsdir = outdir / "extracted_counts"
        self.fluxdir = outdir / "fluxcal"
        self.finaldir = outdir / "final_joined"
        self.diagdir = outdir / "diagnostics"

    def save_state(self):
        save_state(self.state, self.state_path)


class Pipeline:
    def __init__(self, steps: List[Step]):
        self.steps = steps
        self.step_ids = [s.step_id for s in steps]

    def list_steps(self) -> List[str]:
        return [f"{i:02d} {s.step_id}: {s.description}" for i, s in enumerate(self.steps)]

    def run(self, ctx: PipelineContext,
            start_at: Optional[str] = None,
            redo_from: Optional[str] = None) -> None:
        """Run steps with resume + redo.

        - If redo_from is set, marks that step and everything after it as incomplete.
        - If start_at is set, begins execution from that step.
        - Steps already marked complete are skipped.
        """
        if redo_from is not None:
            if redo_from not in self.step_ids:
                raise ValueError(f"Unknown step_id for redo_from: {redo_from}")
            ctx.state.unmark_from(redo_from)
            ctx.save_state()

        start_idx = 0
        if start_at is not None:
            if start_at not in self.step_ids:
                raise ValueError(f"Unknown step_id for start_at: {start_at}")
            start_idx = self.step_ids.index(start_at)

        for step in self.steps[start_idx:]:
            if step.step_id in ctx.state.completed_steps:
                print(f"[SKIP] {step.step_id}: already complete")
                continue
            print(f"\n[RUN ] {step.step_id}: {step.description}")
            step.runner(ctx)
            ctx.state.mark_complete(step.step_id)
            ctx.save_state()
            print(f"[DONE] {step.step_id}")
