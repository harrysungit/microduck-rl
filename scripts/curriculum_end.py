"""Iteration at which a task's step-keyed curricula reach their last stage.

Every staged curriculum term holds lists of {"step": <env steps>, ...} dicts; the last stage of the
latest one, divided by the rl cfg's num_steps_per_env, is where the task's training conditions stop
changing. Terrain levels and other performance-keyed curricula have no such end and are not counted.

    python scripts/curriculum_end.py Mjlab-StandUp-Flat-MicroDuck-HD1910 ...
"""

import argparse
import json
import math

import mjlab  # noqa: F401  # loads the mjlab.tasks entry points
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("tasks", nargs="+")
    args = ap.parse_args()
    out = {}
    for task in args.tasks:
        cfg = load_env_cfg(task)
        steps_per_env = load_rl_cfg(task).num_steps_per_env
        last, by_term = 0, {}
        for name, term in cfg.curriculum.items():
            for val in term.params.values():
                if isinstance(val, list) and val and all(isinstance(v, dict) and "step" in v for v in val):
                    s = max(v["step"] for v in val)
                    by_term[name] = max(by_term.get(name, 0), math.ceil(s / steps_per_env))
                    last = max(last, s)
        out[task] = {"iterations": math.ceil(last / steps_per_env), "num_steps_per_env": steps_per_env,
                     "by_term": dict(sorted(by_term.items(), key=lambda kv: -kv[1])[:6])}
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
