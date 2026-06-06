import argparse

from _pipeline_common import load_pipeline_env
import pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate local model outputs.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--scopes", default="smoke")
    parser.add_argument("--modes", default="simple,full")
    parser.add_argument("--local-model", default="gemma4:26b")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--env-file", default=str(pipeline.DEFAULT_ENV_FILE))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_pipeline_env(args.env_file)
    pipeline.generate_outputs(
        argparse.Namespace(
            run_dir=args.run_dir,
            provider="local",
            scopes=args.scopes,
            modes=args.modes,
            local_model=args.local_model,
            saas_provider="gemini",
            saas_model=pipeline.DEFAULT_GEMINI_MODEL,
            workers=args.workers,
            limit=args.limit,
            saas_request_delay_sec=0.0,
            saas_max_retries=0,
            saas_backoff_base_sec=1.0,
        )
    )


if __name__ == "__main__":
    main()
