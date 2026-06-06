import argparse

from _pipeline_common import load_pipeline_env
import pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score local outputs against SaaS references.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--reference-mode", choices=pipeline.MODES, default="simple")
    parser.add_argument("--score-scopes", default="smoke")
    parser.add_argument(
        "--saas-simple-baseline-dir",
        default=str(pipeline.DEFAULT_SAAS_SIMPLE_BASELINE_DIR),
    )
    parser.add_argument("--env-file", default=str(pipeline.DEFAULT_ENV_FILE))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_pipeline_env(args.env_file)
    pipeline.score_outputs(args)


if __name__ == "__main__":
    main()
