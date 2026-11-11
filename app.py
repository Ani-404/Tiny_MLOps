"""FastAPI service for GitHub Repository Health & Open-Source Welcomeness Scorer.

Provides real-time asynchronous repo inspection, feature extraction, and ML scoring.
"""

from __future__ import annotations

import os
import re
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from train import FEATURE_NAMES, MODEL_PATH

model = None

TIER_WEIGHTS = {
    "Thriving & Welcoming": 100.0,
    "Healthy & Maintained": 75.0,
    "Developing / Small Community": 55.0,
    "Stale / High Risk": 20.0,
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Run `python train.py` first."
        )
    model = joblib.load(MODEL_PATH)
    yield


app = FastAPI(
    title="GitHub Repository Health & Welcomeness Scorer",
    description="Analyzes GitHub repos for open-source health, maintenance, and contributor readiness.",
    version="1.0.0",
    lifespan=lifespan,
)


class PredictRequest(BaseModel):
    repo_url: str | None = Field(
        default=None,
        description="GitHub repository URL or 'owner/repo' identifier",
        examples=["https://github.com/pallets/flask"],
    )
    features: list[float] | None = Field(
        default=None,
        min_length=len(FEATURE_NAMES),
        max_length=len(FEATURE_NAMES),
        description=f"Direct 11-element feature vector: {FEATURE_NAMES}",
    )


class PredictResponse(BaseModel):
    repo: str
    health_tier: str
    health_score: float
    confidence: float
    tier_probabilities: dict[str, float]
    metrics: dict[str, Any]
    insights: list[str]


def parse_repo_path(repo_input: str) -> tuple[str, str]:
    """Extract owner and repo from URL or path string."""
    cleaned = repo_input.strip().rstrip("/")
    match = re.search(r"github\.com/([^/]+)/([^/]+)", cleaned)
    if match:
        return match.group(1), match.group(2)
    parts = cleaned.split("/")
    if len(parts) >= 2:
        return parts[-2], parts[-1]
    raise ValueError(
        f"Invalid GitHub repository identifier '{repo_input}'. Expected format: 'owner/repo' or 'https://github.com/owner/repo'"
    )


async def extract_github_features(owner: str, repo: str) -> dict[str, Any]:
    """Asynchronously fetch repo metrics from GitHub REST API."""
    token = os.getenv("GITHUB_TOKEN")
    headers = {
        "Accept": "application/vnd.github.v3+json",
        "User-Agent": "Tiny-MLOps-API",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"

    base_url = f"https://api.github.com/repos/{owner}/{repo}"

    async with httpx.AsyncClient(headers=headers, timeout=12.0) as client:
        # Core repo info
        resp = await client.get(base_url)
        if resp.status_code == 404:
            raise HTTPException(
                status_code=404,
                detail=f"Repository '{owner}/{repo}' not found on GitHub.",
            )
        if resp.status_code in (403, 429):
            raise HTTPException(
                status_code=429,
                detail="GitHub API rate limit reached. Set GITHUB_TOKEN environment variable or try again later.",
            )
        if resp.status_code >= 400:
            raise HTTPException(
                status_code=resp.status_code,
                detail=f"GitHub API returned error: {resp.text}",
            )
        repo_data = resp.json()

        # Community profile
        comm_resp = await client.get(f"{base_url}/community/profile")
        comm_data = comm_resp.json() if comm_resp.status_code == 200 else {}
        files = comm_data.get("files", {})

        # Weekly participation stats
        part_resp = await client.get(f"{base_url}/stats/participation")
        part_data = part_resp.json() if part_resp.status_code == 200 else {}
        all_commits = (
            part_data.get("all", []) if isinstance(part_data, dict) else []
        )
        recent_weeks = all_commits[-8:] if len(all_commits) >= 8 else all_commits
        avg_weekly_commits = (
            float(sum(recent_weeks) / len(recent_weeks)) if recent_weeks else 0.0
        )

        # Issue resolution rate
        issues_resp = await client.get(f"{base_url}/issues?state=all&per_page=50")
        issue_close_rate = 0.5
        if issues_resp.status_code == 200:
            issues = issues_resp.json()
            if isinstance(issues, list) and issues:
                closed = sum(1 for item in issues if item.get("state") == "closed")
                issue_close_rate = round(closed / len(issues), 4)

        # Good first issues / beginner friendly labels
        labels_resp = await client.get(f"{base_url}/labels?per_page=100")
        has_good_first_issues = 0
        if labels_resp.status_code == 200:
            labels = labels_resp.json()
            if isinstance(labels, list):
                friendly_keywords = {
                    "good first issue",
                    "help wanted",
                    "beginner-friendly",
                    "easy",
                }
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
        "repo_name": f"{owner}/{repo}",
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


def generate_actionable_insights(metrics: dict[str, Any]) -> list[str]:
    """Generate human-readable insights and open-source advice based on extracted metrics."""
    insights: list[str] = []

    # Maintainer responsiveness
    close_rate = metrics["issue_close_rate"]
    if close_rate >= 0.70:
        insights.append(
            f"High maintainer responsiveness: {close_rate:.0%} of recent issues have been resolved."
        )
    elif close_rate <= 0.35:
        insights.append(
            f"Low issue resolution rate ({close_rate:.0%}): issues and PRs may linger without triage."
        )

    # Activity cadence
    days_push = metrics["days_since_last_push"]
    if days_push <= 7.0:
        insights.append(
            f"Active development: codebase updated recently ({days_push:.1f} days ago)."
        )
    elif days_push >= 90.0:
        insights.append(
            f"Maintenance caution: no commits in the last {int(days_push)} days; project may be dormant."
        )

    # Contributor readiness
    if metrics["has_contributing"] == 1:
        insights.append(
            "Clear CONTRIBUTING.md guidelines are provided for onboarding developers."
        )
    else:
        insights.append(
            "Missing CONTRIBUTING.md: adding one would lower friction for open-source contributors."
        )

    if metrics["has_good_first_issues"] == 1:
        insights.append(
            "Beginner-friendly 'good first issue' / 'help wanted' labels are actively used."
        )

    if metrics["has_license"] == 0:
        insights.append(
            "Warning: No open-source LICENSE detected; contributions carry legal ambiguity."
        )

    return insights


@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": model is not None}


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if request.features is not None:
        vector = request.features
        repo_name = request.repo_url or "custom-features"
        metrics = dict(zip(FEATURE_NAMES, vector))
    elif request.repo_url:
        try:
            owner, repo = parse_repo_path(request.repo_url)
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e))
        metrics = await extract_github_features(owner, repo)
        vector = [metrics[feat] for feat in FEATURE_NAMES]
        repo_name = metrics["repo_name"]
    else:
        raise HTTPException(
            status_code=422,
            detail="Either 'repo_url' or 'features' must be provided.",
        )

    # Model inference
    tier_pred = str(model.predict([vector])[0])
    probabilities = model.predict_proba([vector])[0]
    classes = list(model.classes_)

    tier_probs = {cls_name: round(float(prob), 4) for cls_name, prob in zip(classes, probabilities)}
    confidence = float(max(probabilities))

    # Calculate continuous health score (0-100)
    health_score = sum(
        tier_probs.get(tier, 0.0) * TIER_WEIGHTS.get(tier, 50.0)
        for tier in TIER_WEIGHTS
    )
    health_score = round(max(0.0, min(100.0, health_score)), 1)

    insights = generate_actionable_insights(metrics)

    return PredictResponse(
        repo=repo_name,
        health_tier=tier_pred,
        health_score=health_score,
        confidence=round(confidence, 4),
        tier_probabilities=tier_probs,
        metrics=metrics,
        insights=insights,
    )
