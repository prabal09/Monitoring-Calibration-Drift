"""CLI: run drift detection over a processed scene and emit per-frame error + flags."""

import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", required=True)
    parser.add_argument("--config", default="configs/drift_thresholds.yaml")
    args = parser.parse_args()
    raise NotImplementedError


if __name__ == "__main__":
    main()
