import argparse

from _pipeline_common import load_pipeline_env
import pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare manifest and split summary.")
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--smoke-percent", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--exclude-dirs", default=pipeline.DEFAULT_EXCLUDE_DIRS)
    parser.add_argument("--env-file", default=str(pipeline.DEFAULT_ENV_FILE))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_pipeline_env(args.env_file)
    pipeline.prepare_dataset(args)


if __name__ == "__main__":
    main()
