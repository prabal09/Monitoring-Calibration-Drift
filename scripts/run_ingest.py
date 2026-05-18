"""CLI: ingest a NuScenes scene and write synchronized frame bundles to data/processed/."""

import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", required=True)
    parser.add_argument("--out", default="data/processed")
    args = parser.parse_args()
    raise NotImplementedError


if __name__ == "__main__":
    main()
