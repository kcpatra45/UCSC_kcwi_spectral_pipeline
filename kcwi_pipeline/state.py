from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, List, Optional
import json


@dataclass
class PipelineState:
    """On-disk state so the pipeline can resume/redo steps.

    Stored at <outdir>/state.json.
    """

    completed_steps: List[str] = field(default_factory=list)
    artifacts: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "completed_steps": list(self.completed_steps),
            "artifacts": self.artifacts,
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "PipelineState":
        return PipelineState(
            completed_steps=list(d.get("completed_steps", [])),
            artifacts=d.get("artifacts", {}),
        )

    def mark_complete(self, step_id: str) -> None:
        if step_id not in self.completed_steps:
            self.completed_steps.append(step_id)

    def unmark_from(self, step_id: str) -> None:
        """Remove step_id and anything after it (i.e. redo from step_id)."""
        if step_id not in self.completed_steps:
            return
        idx = self.completed_steps.index(step_id)
        self.completed_steps = self.completed_steps[:idx]


def load_state(state_path: Path) -> PipelineState:
    if not state_path.exists():
        return PipelineState()
    with open(state_path, "r", encoding="utf-8") as f:
        return PipelineState.from_dict(json.load(f))


def save_state(state: PipelineState, state_path: Path) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    with open(state_path, "w", encoding="utf-8") as f:
        json.dump(state.to_dict(), f, indent=2)
