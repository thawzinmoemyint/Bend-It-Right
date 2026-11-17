"""Upload the browser version of Bend It Right (the web/ folder) to a free static Hugging Face Space.

    pip install huggingface_hub
    hf auth login                     # paste a token with "write" access
    python deploy_to_hf.py <your-hf-username>/bend-it-right

Run it again after changing anything in web/ to update the Space.
"""

import sys
from pathlib import Path

from huggingface_hub import HfApi

if len(sys.argv) != 2 or "/" not in sys.argv[1]:
    sys.exit("Usage: python deploy_to_hf.py <hf-username>/<space-name>")

repo_id = sys.argv[1]
web = Path(__file__).resolve().parent / "web"
api = HfApi()
api.create_repo(repo_id, repo_type="space", space_sdk="static", exist_ok=True)
api.upload_folder(
    repo_id=repo_id,
    repo_type="space",
    folder_path=web,
    ignore_patterns=[".DS_Store", "**/.DS_Store"],
    commit_message="Deploy Bend It Right web app",
)
print(f"Uploaded. Open https://huggingface.co/spaces/{repo_id}")
