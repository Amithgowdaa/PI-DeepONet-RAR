"""
Utilities for pushing outputs from Kaggle VM to GitHub.

Usage (in Kaggle notebook):
    from utils.io import push_to_github
    push_to_github(token, repo="username/PI-DeepONet-RAR", source_dirs=["/kaggle/working/results"])
"""

import os
import subprocess
import shutil
from pathlib import Path


def setup_git(token: str, repo: str, branch: str = "main", clone_dir: str = "/tmp/repo"):
    """
    Clone repo into clone_dir using token auth.
    """
    if os.path.exists(clone_dir):
        shutil.rmtree(clone_dir)

    url = f"https://{token}@github.com/{repo}.git"
    subprocess.run(["git", "clone", "--depth=1", "-b", branch, url, clone_dir], check=True)

    subprocess.run(["git", "config", "user.email", "kaggle@runner.local"], cwd=clone_dir, check=True)
    subprocess.run(["git", "config", "user.name",  "Kaggle Runner"],        cwd=clone_dir, check=True)

    return clone_dir


def push_to_github(
    token: str,
    repo: str,
    source_dirs: list,
    commit_message: str = "auto: push PI-DeepONet-RAR outputs from Kaggle",
    branch: str = "main",
    clone_dir: str = "/tmp/repo",
):
    """
    Push files/directories from Kaggle output VM to GitHub repo.
    """
    setup_git(token, repo, branch, clone_dir)

    for src in source_dirs:
        src = Path(src)
        if not src.exists():
            print(f"[io] Skipping {src} — does not exist")
            continue

        dst = Path(clone_dir) / src.name
        if src.is_dir():
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)
        print(f"[io] Copied {src} → {dst}")

    result = subprocess.run(
        ["git", "status", "--porcelain"], cwd=clone_dir, capture_output=True, text=True
    )
    if not result.stdout.strip():
        print("[io] Nothing to commit.")
        return

    subprocess.run(["git", "add", "-A"],                         cwd=clone_dir, check=True)
    subprocess.run(["git", "commit", "-m", commit_message],      cwd=clone_dir, check=True)
    subprocess.run(["git", "push", "origin", branch],            cwd=clone_dir, check=True)
    print(f"[io] Pushed to {repo}:{branch}")
