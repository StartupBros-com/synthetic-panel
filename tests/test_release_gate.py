"""Exercise the tracked-tree gate with synthetic data, without model downloads."""

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BRAND = "Startup" + "Bros"
LEGACY_URL = f"https://github.com/{BRAND}/synthetic-panel"
CANONICAL_URL = f"https://github.com/{BRAND}-com/synthetic-panel"
HOLDER = f"{BRAND} LLC"
HOLDER_LINE = f"Copyright (c) 2026 {HOLDER} (House of Vibe, https://houseofvibe.ai)"


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


@pytest.mark.parametrize(
    "notice",
    [
        HOLDER_LINE,
        f"Copyright (c) 2026 {HOLDER}",
        f"Copyright (c) 2025-2026 {HOLDER} (House of Vibe, https://houseofvibe.ai)",
        f"Copyright (c) 2025, 2026 {HOLDER}",
        f"Copyright (c) 2026 {HOLDER}. All rights reserved.",
        f"Copyright (c) 2026 {HOLDER}, and contributors",
    ],
)
def test_root_license_may_name_the_copyright_holder(tmp_path, notice):
    result = run_gate(tmp_path, f"MIT License\n\n{notice}\n\nPermission is granted.\n", "LICENSE")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "RELEASE GATE: CLEAN" in result.stdout


@pytest.mark.parametrize(
    "filename",
    [
        "sample.md",
        "README.md",
        "NOTICE",
        "LICENSE.md",
        "LICENSE.txt",
        "license",
        "docs/LICENSE",
        "LICENSE/notes.md",
        "skill/LICENSE",
    ],
)
def test_copyright_holder_is_not_allowed_outside_the_root_license(tmp_path, filename):
    result = run_gate(tmp_path, HOLDER_LINE + "\n", filename)
    assert result.returncode == 1
    assert f"{filename}:1:{HOLDER_LINE}" in result.stdout
    assert "RELEASE GATE: FAIL" in result.stdout


@pytest.mark.parametrize(
    "line",
    [
        # The brand a second time on the notice line.
        f"Copyright (c) 2026 {HOLDER} ({BRAND}, https://houseofvibe.ai)",
        f"Copyright (c) 2026 {HOLDER} (https://{BRAND.lower()}.com)",
        f"Copyright (c) 2026 {HOLDER} and {BRAND} Media",
        # A holder that is not exactly the company.
        f"Copyright (c) 2026 {BRAND}",
        f"Copyright (c) 2026 {BRAND} Media LLC",
        f"Copyright (c) 2026 {HOLDER}s",
        f"Copyright (c) 2026 {HOLDER}-backed course",
        f"Copyright (c) 2026 The {HOLDER}",
        f"Copyright (c) 2026 x{HOLDER}",
        f"Copyright (c) 2026 {BRAND}Media LLC",
        f"Copyright (c) 2026 {HOLDER}(House of Vibe)",
        # A year that is not four digits, or not set off by single spaces.
        f"Copyright (c) 202 {HOLDER}",
        f"Copyright (c) 20266 {HOLDER}",
        f"Copyright (c) 2026{HOLDER}",
        f"Copyright (c) 2026  {HOLDER}",
        f"Copyright (c)2026 {HOLDER}",
        f"Copyright (c)  2026 {HOLDER}",
        f"Copyright (c)\t2026 {HOLDER}",
        # Not a notice line of the license's own shape.
        f"Portions Copyright (c) 2026 {HOLDER}",
        f" Copyright (c) 2026 {HOLDER}",
        f"Copyright (c) {HOLDER}",
        f"Copyright 2026 {HOLDER}",
        f"Built by {HOLDER}",
        f"{HOLDER}",
    ],
)
def test_root_license_allows_nothing_but_the_holder_of_a_notice_line(tmp_path, line):
    result = run_gate(tmp_path, f"MIT License\n\n{line}\n", "LICENSE")
    assert result.returncode == 1
    assert f"LICENSE:3:{line}" in result.stdout
    assert "RELEASE GATE: FAIL" in result.stdout


def test_root_license_notice_does_not_hide_a_brand_mention_on_another_line(tmp_path):
    content = f"MIT License\n\n{HOLDER_LINE}\n\nSupport: {BRAND} members only.\n"
    result = run_gate(tmp_path, content, "LICENSE")
    assert result.returncode == 1
    assert f"LICENSE:5:Support: {BRAND} members only." in result.stdout
    assert f"LICENSE:3:{HOLDER_LINE}" not in result.stdout


@pytest.mark.parametrize(
    "term", ["side" + " business", "sur" + "vey responses", "Mi" + "ke"]
)
def test_other_prohibited_terms_still_fail_on_the_notice_line(tmp_path, term):
    line = f"{HOLDER_LINE} {term}"
    result = run_gate(tmp_path, line + "\n", "LICENSE")
    assert result.returncode == 1
    assert f"LICENSE:1:{line}" in result.stdout


def test_this_repository_passes_its_own_gate_with_the_company_notice(tmp_path):
    """The tracked tree as it is, with the root LICENSE naming the company."""
    repo = tmp_path / "copy"
    tracked = subprocess.run(
        ["git", "-C", str(ROOT), "ls-files", "-z"], capture_output=True, check=True
    ).stdout.split(b"\0")
    for name in filter(None, tracked):
        source = ROOT / name.decode()
        target = repo / name.decode()
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    license_text = (repo / "LICENSE").read_text()
    notice = [line for line in license_text.splitlines() if line.startswith("Copyright")]
    assert notice, "the root LICENSE has no copyright notice"
    (repo / "LICENSE").write_text(license_text.replace(notice[0], HOLDER_LINE, 1))
    subprocess.run(["git", "init", "--quiet", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
    result = subprocess.run(
        ["bash", str(repo / "scripts/release_gate.sh")],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "RELEASE GATE: CLEAN" in result.stdout
