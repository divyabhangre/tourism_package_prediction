from huggingface_hub import HfApi
import os

# Initialize HF API
api = HfApi(token=os.getenv("HF_TOKEN"))

# HF Space repo
repo_id = "divyabhangre/tourism-package-predict"
repo_type = "space"

# ── Push all deployment files to HF Space ──
files_to_upload = [
    ("tourism_project/app.py","app.py"),
    ("tourism_project/Dockerfile","Dockerfile"),
    ("tourism_project/requirements.txt","requirements.txt"),
    ]

for local_path, repo_path in files_to_upload:
    api.upload_file(path_or_fileobj=local_path,path_in_repo=repo_path,repo_id=repo_id,repo_type=repo_type,)
    print(f"✅ Uploaded {local_path} → {repo_path}")

print("\n🚀 All deployment files pushed to Hugging Face Space!")
print(f"👉 Visit: https://huggingface.co/spaces/{repo_id}/tree/main")
