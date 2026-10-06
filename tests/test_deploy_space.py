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
