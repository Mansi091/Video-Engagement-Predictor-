#!/usr/bin/env python3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from video_engagement.pipeline.run_experiment import load_config, run_experiment

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Video Engagement Prediction Pipeline")
    parser.add_argument("--config", default=None)
    parser.add_argument("--skip-clip", action="store_true")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config or ROOT / "config.yaml")
    if args.quick:
        cfg["data"]["max_users"] = 100
        cfg["data"]["max_ratings"] = 500
        cfg["model"]["epochs"] = 5
        args.skip_clip = True
    run_experiment(cfg, skip_clip=args.skip_clip)
