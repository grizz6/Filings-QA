import sys
import types
from pathlib import Path

import pytest

from scripts import deploy_space
from src.config import ROOT


def test_space_files_are_exactly_what_the_image_needs():
    files = {p.as_posix() for p in deploy_space.space_files(ROOT)}
    for needed in ("Dockerfile", "start.sh", "config.yaml", "requirements-app.txt",
                   "src/api.py", "src/answer.py", "app/Ask.py", "app/pages/1_Monitoring.py",
                   ".streamlit/config.toml"):  # fmt: skip
        assert needed in files, needed
    assert not any(f.startswith(("tests/", "eval/", "data/", ".github/")) for f in files)
    assert not any(".env" in f or "__pycache__" in f for f in files)


def test_space_url_from_repo_id():
    assert deploy_space.space_host("Griz6/Filings_QA") == "https://griz6-filings-qa.hf.space"


class FakeHfApi:
    """Records what the deploy does; inspects the staged folder during the upload."""

    stages: list[str] = []

    def __init__(self, token):
        self.token = token
        self.calls = []
        FakeHfApi.last = self

    def whoami(self):
        return {"name": "griz"}

    def create_repo(self, repo_id, **kw):
        self.calls.append(("create_repo", repo_id, kw))

    def add_space_secret(self, repo_id, key, value):
        self.calls.append(("secret", repo_id, key, value))

    def upload_folder(self, folder_path, repo_id, **kw):
        staged = sorted(p.relative_to(folder_path).as_posix()
                        for p in Path(folder_path).rglob("*") if p.is_file())  # fmt: skip
        self.calls.append(("upload", repo_id, staged, kw))

    def get_space_runtime(self, repo_id):
        return types.SimpleNamespace(stage=FakeHfApi.stages.pop(0))


@pytest.fixture
def fake_hub(monkeypatch):
    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(HfApi=FakeHfApi))
    for name, value in {"HF_TOKEN": "hf_x", "GEMINI_API_KEY": "g", "SUPABASE_DB_URL": "pg"}.items():
        monkeypatch.setenv(name, value)
    waited = []
    monkeypatch.setattr(deploy_space, "wait_until_running", lambda api, repo: waited.append(repo))
    return waited


def test_deploy_creates_space_sets_secrets_and_uploads_staged_files(fake_hub, monkeypatch):
    monkeypatch.delenv("HF_SPACE", raising=False)
    monkeypatch.setenv("GITHUB_SHA", "abcdef1234")
    deploy_space.main()
    calls = FakeHfApi.last.calls
    expected_repo = {"repo_type": "space", "space_sdk": "docker", "exist_ok": True}
    assert calls[0] == ("create_repo", "griz/filings-qa", expected_repo)
    assert [c[2:] for c in calls if c[0] == "secret"] == [
        ("GEMINI_API_KEY", "g"),
        ("SUPABASE_DB_URL", "pg"),
    ]
    _, repo, staged, kw = next(c for c in calls if c[0] == "upload")
    assert "README.md" in staged and "Dockerfile" in staged and "app/Ask.py" in staged
    assert not any(f.startswith("tests/") or ".env" in f for f in staged)
    assert kw["commit_message"] == "Deploy abcdef1"
    assert fake_hub == ["griz/filings-qa"]  # waited for the new build


def test_deploy_uses_hf_space_variable_when_set(fake_hub, monkeypatch):
    monkeypatch.setenv("HF_SPACE", "someone/custom-name")
    deploy_space.main()
    assert FakeHfApi.last.calls[0][1] == "someone/custom-name"


class Health:
    def __init__(self, text):
        self.text = text


@pytest.fixture
def fast_clock(monkeypatch):
    now = [0.0]
    monkeypatch.setattr(deploy_space.time, "sleep", lambda s: now.__setitem__(0, now[0] + s))
    monkeypatch.setattr(deploy_space.time, "time", lambda: now[0])
    return now


def test_wait_returns_once_running_and_healthy(fast_clock, monkeypatch):
    import requests

    FakeHfApi.stages = ["BUILDING", "APP_STARTING", "RUNNING"]
    urls = []
    monkeypatch.setattr(requests, "get", lambda url, timeout: urls.append(url) or Health("ok"))
    deploy_space.wait_until_running(FakeHfApi("t"), "griz/filings-qa")
    assert urls == ["https://griz-filings-qa.hf.space/_stcore/health"]


def test_wait_fails_fast_on_build_error(fast_clock):
    FakeHfApi.stages = ["BUILDING", "BUILD_ERROR"]
    with pytest.raises(SystemExit, match="BUILD_ERROR"):
        deploy_space.wait_until_running(FakeHfApi("t"), "griz/filings-qa")


def test_wait_times_out_when_never_healthy(fast_clock, monkeypatch):
    import requests

    FakeHfApi.stages = ["RUNNING"] * 1000
    monkeypatch.setattr(requests, "get", lambda url, timeout: Health("starting"))
    with pytest.raises(SystemExit, match="did not become healthy"):
        deploy_space.wait_until_running(FakeHfApi("t"), "griz/filings-qa")
