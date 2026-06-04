from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.stream.experiment.tabicl_cure_runner import RunConfig, run_tabicl_cure


DEFAULT_MAXLEN = {
    "noaa": 18159,
    "meter": 22950,
    "rialto": 82250,
    "posture_no8": 163477,
    "nomao": 34465,
    "poker": 829201,
    "agr_a": 30000,
}


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    default_data_root = repo_root / "data" / "uspds"
    default_output_root = repo_root / "results"

    p = argparse.ArgumentParser(description="Run the main TabICLv2 + CURE setting on one stream dataset.")
    p.add_argument("--dataset", required=True, choices=list(DEFAULT_MAXLEN))
    p.add_argument("--tau", type=float, required=True)
    p.add_argument("--data-root", type=str, default=str(default_data_root))
    p.add_argument("--output-root", type=str, default=str(default_output_root))
    p.add_argument("--max-stream-length", type=int, default=None)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--tabicl-model-path", type=str, default="auto")
    args = p.parse_args()

    cfg = RunConfig(
        dataset=args.dataset,
        data_root=args.data_root,
        output_root=args.output_root,
        tau=args.tau,
        max_stream_length=args.max_stream_length or DEFAULT_MAXLEN[args.dataset],
        device=args.device,
        threads=args.threads,
        tabicl_model_path=args.tabicl_model_path,
    )
    result = run_tabicl_cure(cfg)
    print(json.dumps(result["summary"], indent=2))
    print(f"Output saved to: {result['output_dir']}")


if __name__ == "__main__":
    main()
