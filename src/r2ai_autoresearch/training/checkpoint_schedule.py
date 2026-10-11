"""Optional uniform selection checkpoints; legacy saves remain unchanged by default."""
from __future__ import annotations

import json
from pathlib import Path


def uniform_steps(total: int, count: int) -> list[int]:
    if type(total) is not int or total <= 0 or type(count) is not int or not 5 <= count <= 10:
        raise ValueError("positive total updates and 5..10 checkpoint count required")
    return sorted({(i * total + count - 1) // count for i in range(1, count + 1)})


def epoch_batches(rows: int, accumulation: int):
    if type(rows) is not int or type(accumulation) is not int or min(rows, accumulation) <= 0:
        raise ValueError('positive row count and accumulation required')
    return [list(range(start, min(start + accumulation, rows))) for start in range(0, rows, accumulation)]


def selection_callback(count: int | None):
    if count is None:
        return None
    uniform_steps(1, count)
    from transformers import TrainerCallback

    class UniformCheckpointCallback(TrainerCallback):
        def on_train_begin(self, args, state, control, **kwargs):
            self.steps = uniform_steps(int(state.max_steps), count)
            if state.is_world_process_zero:
                path = Path(args.output_dir) / "selection_schedule.json"
                path.parent.mkdir(parents=True, exist_ok=True)
                value = {"total_updates": int(state.max_steps), "save_steps": self.steps,
                         "step_unit": "optimizer_updates", "includes_final": True}
                if path.exists() and json.loads(path.read_text()) != value:
                    raise ValueError("selection schedule changed on resume")
                path.write_text(json.dumps(value, indent=2) + "\n")
            return control

        def on_step_end(self, args, state, control, **kwargs):
            # Override DefaultFlowCallback so regular cadence adds no extras.
            control.should_save = state.global_step in self.steps
            return control

    return UniformCheckpointCallback()
