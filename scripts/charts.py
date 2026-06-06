import argparse

from _pipeline_common import load_pipeline_env
import pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate run-local charts.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--env-file", default=str(pipeline.DEFAULT_ENV_FILE))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_pipeline_env(args.env_file)
    pipeline.generate_charts(args)


if __name__ == "__main__":
    main()
