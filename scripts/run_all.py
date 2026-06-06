import argparse

from _pipeline_common import load_pipeline_env
import pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run prepare, generation, score, and charts.")
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--smoke-percent", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--exclude-dirs", default=pipeline.DEFAULT_EXCLUDE_DIRS)
    parser.add_argument("--local-scopes", default="smoke")
    parser.add_argument("--saas-scopes", default="smoke")
    parser.add_argument("--modes", default="simple,full")
    parser.add_argument("--local-model", default="gemma4:26b")
    parser.add_argument("--saas-provider", choices=["gemini", "openai"], default="gemini")
    parser.add_argument("--saas-model", default=pipeline.DEFAULT_GEMINI_MODEL)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--parallel-providers", action="store_true")
    parser.add_argument("--saas-request-delay-sec", type=float, default=4.0)
    parser.add_argument("--saas-max-retries", type=int, default=5)
    parser.add_argument("--saas-backoff-base-sec", type=float, default=5.0)
    parser.add_argument("--reference-mode", choices=pipeline.MODES, default="simple")
    parser.add_argument("--score-scopes", default="smoke")
    parser.add_argument(
        "--saas-simple-baseline-dir",
        default=str(pipeline.DEFAULT_SAAS_SIMPLE_BASELINE_DIR),
    )
    parser.add_argument("--stop-before-score", action="store_true")
    parser.add_argument("--env-file", default=str(pipeline.DEFAULT_ENV_FILE))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_pipeline_env(args.env_file)
    pipeline.run_all(args)


if __name__ == "__main__":
    main()
