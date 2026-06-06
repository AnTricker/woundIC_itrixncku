from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def load_pipeline_env(env_file: str | None = None) -> None:
    import pipeline

    pipeline.load_env_file(Path(env_file) if env_file else pipeline.DEFAULT_ENV_FILE)
