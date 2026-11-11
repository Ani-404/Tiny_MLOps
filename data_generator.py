"""Generate empirical repository dataset for training and benchmarking."""

from pathlib import Path
import numpy as np
import pandas as pd


def generate_data(output_path: Path | str = Path("data/repos.csv")) -> pd.DataFrame:
    """Generate realistic repository metrics and write to CSV."""
    np.random.seed(42)
    records = []

    tiers = [
        ("Thriving & Welcoming", 75),
        ("Healthy & Maintained", 75),
        ("Developing / Small Community", 60),
        ("Stale / High Risk", 60),
    ]

    sample_repos = {
        "Thriving & Welcoming": [
            "tiangolo/fastapi", "pallets/flask", "psf/requests", "django/django",
            "scikit-learn/scikit-learn", "pytorch/pytorch", "pandas-dev/pandas",
            "numpy/numpy", "encode/httpx", "pydantic/pydantic", "astral-sh/uv",
            "astral-sh/ruff", "Textualize/rich", "pytest-dev/pytest", "encode/starlette"
        ],
        "Healthy & Maintained": [
            "marshmallow-code/marshmallow", "tqdm/tqdm", "pypa/flit", "tornadoweb/tornado",
            "cherrypy/cherrypy", "gevent/gevent", "paramiko/paramiko", "urllib3/urllib3",
            "certifi/python-certifi", "chardet/chardet", "dateutil/dateutil", "pyca/cryptography"
        ],
        "Developing / Small Community": [
            "kylebeadle/Python-Audio-Player", "m-col/pydantic-xml", "s-m-e-e/minimal-mlops",
            "dev-tldr/tiny-summarizer", "alex-k/weather-notifier-cli", "open-micro/fastapi-crud-demo",
            "py-student/algo-visualizer", "cli-crafter/git-easy-sync", "indie-maker/saas-landing-py"
        ],
        "Stale / High Risk": [
            "kennethreitz/autoenv", "theorchard/charlotte", "defunkt/facebox",
            "cloudant/python-cloudant", "mitsuhiko/phpserialize", "ask/celerymon",
            "nose-devs/nose", "pypa/virtualenvwrapper", "ActiveState/appdirs"
        ],
    }

    for tier, count in tiers:
        names = sample_repos[tier]
        for i in range(count):
            base_name = names[i % len(names)]
            repo_name = f"{base_name}" if i < len(names) else f"{base_name}-replica-{i}"

            if tier == "Thriving & Welcoming":
                days_since_push = float(np.clip(np.random.exponential(scale=3.0), 0.1, 14.0))
                avg_commits = float(np.clip(np.random.normal(loc=14.0, scale=6.0), 3.0, 50.0))
                close_rate = float(np.clip(np.random.beta(a=8, b=3), 0.55, 0.98))
                open_issues = int(np.clip(np.random.lognormal(mean=4.5, sigma=1.0), 15, 3000))
                stars = int(np.clip(np.random.lognormal(mean=9.5, sigma=1.2), 3000, 150000))
                forks = int(stars * np.random.uniform(0.1, 0.25))
                has_readme = 1
                has_contributing = 1 if np.random.rand() > 0.08 else 0
                has_license = 1 if np.random.rand() > 0.02 else 0
                has_coc = 1 if np.random.rand() > 0.15 else 0
                has_good_first = 1 if np.random.rand() > 0.12 else 0

            elif tier == "Healthy & Maintained":
                days_since_push = float(np.clip(np.random.exponential(scale=12.0), 1.0, 45.0))
                avg_commits = float(np.clip(np.random.normal(loc=4.5, scale=2.5), 0.8, 15.0))
                close_rate = float(np.clip(np.random.beta(a=6, b=3), 0.45, 0.90))
                open_issues = int(np.clip(np.random.lognormal(mean=3.5, sigma=0.8), 5, 500))
                stars = int(np.clip(np.random.lognormal(mean=7.5, sigma=1.0), 500, 25000))
                forks = int(stars * np.random.uniform(0.08, 0.2))
                has_readme = 1
                has_contributing = 1 if np.random.rand() > 0.35 else 0
                has_license = 1 if np.random.rand() > 0.05 else 0
                has_coc = 1 if np.random.rand() > 0.50 else 0
                has_good_first = 1 if np.random.rand() > 0.55 else 0

            elif tier == "Developing / Small Community":
                days_since_push = float(np.clip(np.random.exponential(scale=18.0), 0.5, 60.0))
                avg_commits = float(np.clip(np.random.normal(loc=2.5, scale=1.8), 0.2, 10.0))
                close_rate = float(np.clip(np.random.beta(a=5, b=4), 0.30, 0.85))
                open_issues = int(np.clip(np.random.lognormal(mean=1.5, sigma=0.9), 0, 40))
                stars = int(np.clip(np.random.lognormal(mean=3.5, sigma=1.1), 5, 450))
                forks = int(max(0, stars * np.random.uniform(0.02, 0.15)))
                has_readme = 1 if np.random.rand() > 0.05 else 0
                has_contributing = 1 if np.random.rand() > 0.70 else 0
                has_license = 1 if np.random.rand() > 0.25 else 0
                has_coc = 1 if np.random.rand() > 0.80 else 0
                has_good_first = 1 if np.random.rand() > 0.75 else 0

            else:  # Stale / High Risk
                days_since_push = float(np.clip(np.random.uniform(180.0, 1200.0), 180.0, 1500.0))
                avg_commits = float(np.clip(np.random.exponential(scale=0.08), 0.0, 0.5))
                close_rate = float(np.clip(np.random.beta(a=2, b=6), 0.05, 0.40))
                open_issues = int(np.clip(np.random.lognormal(mean=4.0, sigma=1.2), 10, 1200))
                stars = int(np.clip(np.random.lognormal(mean=6.0, sigma=2.5), 10, 60000))
                forks = int(stars * np.random.uniform(0.05, 0.3))
                has_readme = 1 if np.random.rand() > 0.10 else 0
                has_contributing = 1 if np.random.rand() > 0.75 else 0
                has_license = 1 if np.random.rand() > 0.30 else 0
                has_coc = 1 if np.random.rand() > 0.85 else 0
                has_good_first = 0 if np.random.rand() > 0.05 else 1

            records.append({
                "repo_name": repo_name,
                "days_since_last_push": round(days_since_push, 2),
                "avg_weekly_commits": round(avg_commits, 2),
                "issue_close_rate": round(close_rate, 4),
                "open_issues_count": open_issues,
                "stars_count": stars,
                "forks_count": forks,
                "has_readme": has_readme,
                "has_contributing": has_contributing,
                "has_license": has_license,
                "has_code_of_conduct": has_coc,
                "has_good_first_issues": has_good_first,
                "health_tier": tier,
            })

    df = pd.DataFrame(records)
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Generated {len(df)} records in {out_path}")
    return df


def main():
    generate_data()


if __name__ == "__main__":
    main()
