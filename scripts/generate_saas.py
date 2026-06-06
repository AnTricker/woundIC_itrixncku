import argparse

from _pipeline_common import load_pipeline_env
import pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate SaaS/Gemini outputs.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--scopes", default="smoke")
    parser.add_argument("--modes", default="simple")
    parser.add_argument("--saas-provider", choices=["gemini", "openai"], default="gemini")
    parser.add_argument("--saas-model", default=pipeline.DEFAULT_GEMINI_MODEL)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--saas-request-delay-sec", type=float, default=4.0)
    parser.add_argument("--saas-max-retries", type=int, default=5)
    parser.add_argument("--saas-backoff-base-sec", type=float, default=5.0)
    parser.add_argument("--env-file", default=str(pipeline.DEFAULT_ENV_FILE))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_pipeline_env(args.env_file)
    pipeline.generate_outputs(
        argparse.Namespace(
            run_dir=args.run_dir,
            provider="saas",
            scopes=args.scopes,
            modes=args.modes,
            local_model="",
            saas_provider=args.saas_provider,
            saas_model=args.saas_model,
            workers=args.workers,
            limit=args.limit,
            saas_request_delay_sec=args.saas_request_delay_sec,
            saas_max_retries=args.saas_max_retries,
            saas_backoff_base_sec=args.saas_backoff_base_sec,
        )
    )


if __name__ == "__main__":
    main()
