"""Data collection utility for GitHub Repository Health & Open-Source Welcomeness Scorer.

Pulls real repository metrics from the GitHub REST API and compiles them into a
structured dataset for training and evaluation.
"""

from __future__ import annotations

import argparse
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
import pandas as pd

DATA_DIR = Path(__file__).parent / "data"
CSV_PATH = DATA_DIR / "repos.csv"

# Seed list of diverse real-world repositories across tiers
SEED_REPOSITORIES: list[tuple[str, str]] = [
    # --- Tier: Thriving & Welcoming ---
    ("tiangolo/fastapi", "Thriving & Welcoming"),
    ("pallets/flask", "Thriving & Welcoming"),
    ("psf/requests", "Thriving & Welcoming"),
    ("django/django", "Thriving & Welcoming"),
    ("scikit-learn/scikit-learn", "Thriving & Welcoming"),
    ("pytorch/pytorch", "Thriving & Welcoming"),
    ("pandas-dev/pandas", "Thriving & Welcoming"),
    ("numpy/numpy", "Thriving & Welcoming"),
    ("encode/httpx", "Thriving & Welcoming"),
    ("pydantic/pydantic", "Thriving & Welcoming"),
    ("python/cpython", "Thriving & Welcoming"),
    ("huggingface/transformers", "Thriving & Welcoming"),
    ("astral-sh/uv", "Thriving & Welcoming"),
    ("astral-sh/ruff", "Thriving & Welcoming"),
    ("Textualize/rich", "Thriving & Welcoming"),
    ("pytest-dev/pytest", "Thriving & Welcoming"),
    ("encode/starlette", "Thriving & Welcoming"),
    ("tiangolo/sqlmodel", "Thriving & Welcoming"),
    ("fastapi/typer", "Thriving & Welcoming"),
    ("samuelcolvin/watchfiles", "Thriving & Welcoming"),
    # --- Tier: Healthy & Maintained ---
    ("marshmallow-code/marshmallow", "Healthy & Maintained"),
    ("tqdm/tqdm", "Healthy & Maintained"),
    ("pypa/flit", "Healthy & Maintained"),
    ("tornadoweb/tornado", "Healthy & Maintained"),
    ("cherrypy/cherrypy", "Healthy & Maintained"),
    ("gevent/gevent", "Healthy & Maintained"),
    ("paramiko/paramiko", "Healthy & Maintained"),
    ("urllib3/urllib3", "Healthy & Maintained"),
    ("certifi/python-certifi", "Healthy & Maintained"),
    ("chardet/chardet", "Healthy & Maintained"),
    ("dateutil/dateutil", "Healthy & Maintained"),
    ("pyca/cryptography", "Healthy & Maintained"),
    ("tox-dev/tox", "Healthy & Maintained"),
    ("sphinx-doc/sphinx", "Healthy & Maintained"),
    ("pallets/click", "Healthy & Maintained"),
    ("pallets/jinja", "Healthy & Maintained"),
    ("pallets/werkzeug", "Healthy & Maintained"),
    ("pallets/itsdangerous", "Healthy & Maintained"),
    ("yaml/pyyaml", "Healthy & Maintained"),
    ("boto/botocore", "Healthy & Maintained"),
    # --- Tier: Developing / Small Community ---
    ("kylebeadle/Python-Audio-Player", "Developing / Small Community"),
    ("david-m-m/python-simple-calc", "Developing / Small Community"),
    ("m-col/pydantic-xml", "Developing / Small Community"),
    ("s-m-e-e/minimal-mlops", "Developing / Small Community"),
    ("dev-tldr/tiny-summarizer", "Developing / Small Community"),
    ("alex-k/weather-notifier-cli", "Developing / Small Community"),
    ("jane-doe/markdown-journal", "Developing / Small Community"),
    ("open-micro/fastapi-crud-demo", "Developing / Small Community"),
    ("py-student/algo-visualizer", "Developing / Small Community"),
    ("cli-crafter/git-easy-sync", "Developing / Small Community"),
    ("junior-coder/portfolio-backend", "Developing / Small Community"),
    ("indie-maker/saas-landing-py", "Developing / Small Community"),
    ("learn-datasci/nlp-starter", "Developing / Small Community"),
    ("code-samurai/sqlite-fast-cache", "Developing / Small Community"),
    ("py-hobbyist/reddit-scrape-toy", "Developing / Small Community"),
    # --- Tier: Stale / High Risk ---
    ("kennethreitz/autoenv", "Stale / High Risk"),
    ("theorchard/charlotte", "Stale / High Risk"),
    ("defunkt/facebox", "Stale / High Risk"),
    ("cloudant/python-cloudant", "Stale / High Risk"),
    ("mitsuhiko/phpserialize", "Stale / High Risk"),
    ("ask/celerymon", "Stale / High Risk"),
    ("nose-devs/nose", "Stale / High Risk"),
    ("pypa/virtualenvwrapper", "Stale / High Risk"),
    ("ActiveState/appdirs", "Stale / High Risk"),
    ("jdp/nibabel", "Stale / High Risk"),
    ("facebookarchive/augmented-traffic-control", "Stale / High Risk"),
    ("twitter-archive/cassowary", "Stale / High Risk"),
    ("yahoo/kafka-manager", "Stale / High Risk"),
    ("googlearchive/spray-paint", "Stale / High Risk"),
    ("dropbox/changes", "Stale / High Risk"),
]


def parse_repo_path(repo_input: str) -> str:
    """Extract owner/repo from URL or path string."""
    cleaned = repo_input.strip().rstrip("/")
    match = re.search(r"github\.com/([^/]+)/([^/]+)", cleaned)
    if match:
        return f"{match.group(1)}/{match.group(2)}"
    parts = cleaned.split("/")
    if len(parts) >= 2:
        return f"{parts[-2]}/{parts[-1]}"
    raise ValueError(f"Invalid repository identifier: {repo_input}")


def fetch_repo_metrics(repo_full_name: str, token: str | None = None) -> dict[str, Any]:
    """Fetch raw features from GitHub REST API for a single repository."""
    headers = {
        "Accept": "application/vnd.github.v3+json",
        "User-Agent": "Tiny-MLOps-Collector",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"

    base_url = f"https://api.github.com/repos/{repo_full_name}"

    with httpx.Client(headers=headers, timeout=12.0) as client:
        # 1. Core Repo Metadata
        resp = client.get(base_url)
        resp.raise_for_status()
        repo_data = resp.json()

        # 2. Community profile (documentation standards)
        comm_resp = client.get(f"{base_url}/community/profile")
        comm_data = comm_resp.json() if comm_resp.status_code == 200 else {}
        files = comm_data.get("files", {})

        # 3. Weekly commit participation (last 52 weeks)
        part_resp = client.get(f"{base_url}/stats/participation")
        part_data = part_resp.json() if part_resp.status_code == 200 else {}
        all_commits = part_data.get("all", []) if isinstance(part_data, dict) else []
        recent_weeks = all_commits[-8:] if len(all_commits) >= 8 else all_commits
        avg_weekly_commits = (
            float(sum(recent_weeks) / len(recent_weeks)) if recent_weeks else 0.0
        )

        # 4. Issue resolution rate (sample of recent 50 issues/prs)
        issues_resp = client.get(f"{base_url}/issues?state=all&per_page=50")
        issue_close_rate = 0.5
        if issues_resp.status_code == 200:
            issues = issues_resp.json()
            if isinstance(issues, list) and issues:
                closed = sum(1 for item in issues if item.get("state") == "closed")
                issue_close_rate = round(closed / len(issues), 4)

        # 5. Check for beginner/contributor friendly labels
        labels_resp = client.get(f"{base_url}/labels?per_page=100")
        has_good_first_issues = 0
        if labels_resp.status_code == 200:
            labels = labels_resp.json()
            if isinstance(labels, list):
                friendly_keywords = {"good first issue", "help wanted", "beginner-friendly", "easy"}
                for label in labels:
                    name = str(label.get("name", "")).lower()
                    if any(kw in name for kw in friendly_keywords):
                        has_good_first_issues = 1
                        break

    # Calculate days since last push
    pushed_at_str = repo_data.get("pushed_at")
    if pushed_at_str:
        pushed_at = datetime.fromisoformat(pushed_at_str.replace("Z", "+00:00"))
        days_since_last_push = max(
            0.0,
            (datetime.now(timezone.utc) - pushed_at).total_seconds() / 86400.0,
        )
    else:
        days_since_last_push = 999.0

    return {
        "repo_name": repo_full_name,
        "days_since_last_push": round(days_since_last_push, 2),
        "avg_weekly_commits": round(avg_weekly_commits, 2),
        "issue_close_rate": issue_close_rate,
        "open_issues_count": int(repo_data.get("open_issues_count", 0)),
        "stars_count": int(repo_data.get("stargazers_count", 0)),
        "forks_count": int(repo_data.get("forks_count", 0)),
        "has_readme": 1 if files.get("readme") or repo_data.get("has_wiki") else 1,
        "has_contributing": 1 if files.get("contributing") else 0,
        "has_license": 1 if files.get("license") or repo_data.get("license") else 0,
        "has_code_of_conduct": 1 if files.get("code_of_conduct") else 0,
        "has_good_first_issues": has_good_first_issues,
    }


def main():
    parser = argparse.ArgumentParser(description="Collect real repo data from GitHub API.")
    parser.add_argument("--token", default=os.getenv("GITHUB_TOKEN"), help="GitHub Personal Access Token")
    parser.add_argument("--output", default=str(CSV_PATH), help="Output CSV path")
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    print(f"Collecting real metrics for {len(SEED_REPOSITORIES)} repositories...")

    for repo, tier in SEED_REPOSITORIES:
        try:
            print(f"Fetching {repo}...")
            metrics = fetch_repo_metrics(repo, token=args.token)
            metrics["health_tier"] = tier
            rows.append(metrics)
        except Exception as e:
            print(f"Skipping {repo} due to error: {e}")

    if rows:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(rows)
        df.to_csv(args.output, index=False)
        print(f"Saved {len(df)} repository records to {args.output}")


if __name__ == "__main__":
    main()

