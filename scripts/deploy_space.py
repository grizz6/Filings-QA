"""Deploy the app to a Hugging Face Space (Docker SDK). Run by .github/workflows/deploy.yml.

1. Create the Space if needed (public, Docker SDK, free CPU hardware).
2. Store GEMINI_API_KEY and SUPABASE_DB_URL as Space secrets (encrypted by Hugging Face;
   the values come from GitHub Secrets and never appear in the repository or logs).
3. Upload exactly the files the Docker image needs, plus the Space README.
4. Wait until the Space has rebuilt and its UI answers a health check.

Env: HF_TOKEN (write token), HF_SPACE (optional "user/name", default "<user>/filings-qa"),
GEMINI_API_KEY, SUPABASE_DB_URL.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = [
    "Dockerfile", ".dockerignore", "start.sh", "config.yaml",
    "requirements.txt", "requirements-index.txt", "requirements-app.txt",
]  # fmt: skip
DIRS = {"src": "*.py", "app": "**/*.py", ".streamlit": "*.toml"}
SECRETS = ("GEMINI_API_KEY", "SUPABASE_DB_URL")
BUILD_TIMEOUT_SECONDS = 25 * 60


def space_files(root: Path = ROOT) -> list[Path]:
    """Paths (relative to root) that go to the Space, besides its README."""
    files = [Path(f) for f in FILES]
    for d, pattern in DIRS.items():
        files += sorted(p.relative_to(root) for p in (root / d).glob(pattern) if p.is_file())
    return files


def space_host(repo_id: str) -> str:
    return "https://" + repo_id.lower().replace("/", "-").replace("_", "-") + ".hf.space"


def wait_until_running(api, repo_id: str) -> None:
    import requests

    deadline = time.time() + BUILD_TIMEOUT_SECONDS
    time.sleep(30)  # let the new commit start a build before reading the stage
    while time.time() < deadline:
        stage = api.get_space_runtime(repo_id).stage
        print(f"Space stage: {stage}", flush=True)
        if stage in ("BUILD_ERROR", "RUNTIME_ERROR", "CONFIG_ERROR"):
            sys.exit(f"Space failed with {stage}: see https://huggingface.co/spaces/{repo_id}")
        if stage == "RUNNING":
            url = f"{space_host(repo_id)}/_stcore/health"
            try:
                if requests.get(url, timeout=30).text.strip() == "ok":
                    print(f"Live: https://huggingface.co/spaces/{repo_id}")
                    return
            except requests.RequestException:
                pass
        time.sleep(20)
    sys.exit(f"Space did not become healthy within {BUILD_TIMEOUT_SECONDS // 60} minutes")


def main() -> None:
    from huggingface_hub import HfApi

    api = HfApi(token=os.environ["HF_TOKEN"])
    repo_id = os.environ.get("HF_SPACE", "").strip() or f"{api.whoami()['name']}/filings-qa"
    api.create_repo(repo_id, repo_type="space", space_sdk="docker", exist_ok=True)
    for name in SECRETS:
        api.add_space_secret(repo_id, name, os.environ[name])

    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp)
        for rel in space_files():
            (stage / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / rel, stage / rel)
        shutil.copy2(ROOT / "space" / "README.md", stage / "README.md")
        sha = os.environ.get("GITHUB_SHA", "local")[:7]
        api.upload_folder(
            folder_path=stage,
            repo_id=repo_id,
            repo_type="space",
            commit_message=f"Deploy {sha}",
            delete_patterns=["src/**", "app/**"],  # drop modules removed from the repo
        )
    print(f"Uploaded {len(space_files()) + 1} files to {repo_id}; waiting for the build")
    wait_until_running(api, repo_id)


if __name__ == "__main__":
    main()
