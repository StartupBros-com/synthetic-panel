"""Exercise the tracked-tree gate with synthetic data, without model downloads."""

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BRAND = "Startup" + "Bros"
LEGACY_URL = f"https://github.com/{BRAND}/synthetic-panel"
CANONICAL_URL = f"https://github.com/{BRAND}-com/synthetic-panel"


def run_gate(tmp_path, content, filename="sample.md"):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "scripts").mkdir()
    for script in (ROOT / "scripts").glob("*"):
        if script.is_file():
            shutil.copyfile(script, repo / "scripts" / script.name)
    (repo / filename).parent.mkdir(parents=True, exist_ok=True)
    (repo / filename).write_text(content)
    subprocess.run(["git", "init", "--quiet", str(repo)], check=True)
    subprocess.run(
        ["git", "-C", str(repo), "config", "color.grep", "always"], check=True
    )
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
    return subprocess.run(
        ["bash", str(repo / "scripts/release_gate.sh")],
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.parametrize("url", [LEGACY_URL, CANONICAL_URL])
@pytest.mark.parametrize(
    "suffix", ["", "@v0.1.0", ".git@v0.1.0", "/actions/workflows/ci.yml"]
)
def test_public_repository_urls_are_allowed(tmp_path, url, suffix):
    result = run_gate(tmp_path, f"[project]({url}{suffix})\n")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "RELEASE GATE: CLEAN" in result.stdout


@pytest.mark.parametrize("url", [LEGACY_URL, CANONICAL_URL])
@pytest.mark.parametrize("prefix", ["git+", ""])
def test_install_urls_and_case_variants_are_allowed(tmp_path, url, prefix):
    result = run_gate(tmp_path, f"uv tool install {prefix}{url.upper()}@v0.1.0\n")
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("url", [LEGACY_URL, CANONICAL_URL])
@pytest.mark.parametrize("order", ["before", "after"])
def test_public_url_does_not_hide_another_brand_mention(tmp_path, url, order):
    content = f"{BRAND} {url}" if order == "before" else f"{url} {BRAND}"
    result = run_gate(tmp_path, content + "\n")
    assert result.returncode == 1
    assert f"sample.md:1:{content}" in result.stdout
    assert "RELEASE GATE: FAIL" in result.stdout


@pytest.mark.parametrize(
    "url",
    [
        CANONICAL_URL + "-other",
        CANONICAL_URL + ".git-other",
        CANONICAL_URL.replace("github.com/", "notgithub.com/"),
        CANONICAL_URL.replace("github.com/", "github.com.example/"),
        CANONICAL_URL.replace("github.com/", "sub.github.com/"),
        CANONICAL_URL.replace("github.com/", "fake+github.com/"),
        CANONICAL_URL.replace("github.com/", "fake=github.com/"),
        CANONICAL_URL.replace("github.com/", "fake!github.com/"),
        CANONICAL_URL.replace("https://", "https://example.invalid/"),
        CANONICAL_URL.replace("https://", "https://user@"),
        CANONICAL_URL.replace("-com/", "-company/"),
        CANONICAL_URL + "_other",
    ],
)
def test_near_match_urls_are_not_exempt(tmp_path, url):
    result = run_gate(tmp_path, url + "\n")
    assert result.returncode == 1
    assert f"sample.md:1:{url}" in result.stdout


def test_multiple_public_urls_on_one_line_are_allowed(tmp_path):
    result = run_gate(tmp_path, f"{LEGACY_URL} {CANONICAL_URL}\n")
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("filename", ["section:1:notes.md", "section\nnotes.md"])
def test_public_url_in_unusual_filename_is_allowed(tmp_path, filename):
    result = run_gate(tmp_path, CANONICAL_URL + "\n", filename)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    "suffix", [f"@{BRAND}", f"/docs/{BRAND}", f"?q={BRAND}", f"#{BRAND}"]
)
def test_public_url_does_not_hide_brand_in_suffix(tmp_path, suffix):
    content = CANONICAL_URL + suffix
    result = run_gate(tmp_path, content + "\n")
    assert result.returncode == 1
    assert f"sample.md:1:{content}" in result.stdout


def test_allowed_url_does_not_hide_near_match_on_same_line(tmp_path):
    content = f"{LEGACY_URL} {CANONICAL_URL}-other"
    result = run_gate(tmp_path, content + "\n")
    assert result.returncode == 1
    assert f"sample.md:1:{content}" in result.stdout


def test_brand_filter_failure_fails_gate(tmp_path):
    result = run_gate(tmp_path, "neutral content\n")
    assert result.returncode == 0, result.stdout + result.stderr
    repo = tmp_path / "repo"
    (repo / "scripts/check_brand_terms.py").write_text(
        "raise RuntimeError('filter unavailable')\n"
    )
    result = subprocess.run(
        ["bash", str(repo / "scripts/release_gate.sh")],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "RELEASE GATE: CLEAN" not in result.stdout


@pytest.mark.parametrize(
    "data",
    [b"not structured", b"file\0", b"file\0wrong\0content\n", b"file\0\0content\n"],
)
def test_brand_filter_rejects_malformed_grep_output(data):
    result = subprocess.run(
        ["python3", str(ROOT / "scripts/check_brand_terms.py")],
        input=data,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0


@pytest.mark.parametrize(
    "term", ["side" + " business", "sur" + "vey responses", "Mi" + "ke"]
)
def test_other_prohibited_terms_still_fail_next_to_allowed_url(tmp_path, term):
    result = run_gate(tmp_path, f"{LEGACY_URL} {term}\n")
    assert result.returncode == 1
    assert f"sample.md:1:{LEGACY_URL} {term}" in result.stdout


@pytest.mark.parametrize(
    "filename", ["sample.csv", "sample.db", "sample.gz", "calibration_data/sample.json"]
)
def test_tracked_data_blobs_still_fail(tmp_path, filename):
    result = run_gate(tmp_path, "neutral content\n", filename)
    assert result.returncode == 1
    assert "Tracked data blob hits:" in result.stdout
    assert filename in result.stdout
