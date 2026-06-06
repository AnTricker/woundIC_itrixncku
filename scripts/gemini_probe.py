import argparse

from _pipeline_common import load_pipeline_env
import pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one-image Gemini probe.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--scopes", default="smoke")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--saas-model", default=pipeline.DEFAULT_GEMINI_MODEL)
    parser.add_argument("--saas-request-delay-sec", type=float, default=4.0)
    parser.add_argument("--saas-max-retries", type=int, default=5)
    parser.add_argument("--saas-backoff-base-sec", type=float, default=5.0)
    parser.add_argument("--env-file", default=str(pipeline.DEFAULT_ENV_FILE))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_pipeline_env(args.env_file)
    pipeline.run_gemini_probe(args)


if __name__ == "__main__":
    main()
