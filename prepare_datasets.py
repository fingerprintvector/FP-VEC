import os
from huggingface_hub import snapshot_download

repo_id_list = [
    "YeungNLP/WizardLM_evol_instruct_V2_143k",
    "YeungNLP/ultrachat_200k",
    "Muennighoff/flan",
]

# 下载数据集并存储在 "Datasets" 文件夹中
for repo_id in repo_id_list:
    local_dir = snapshot_download(
        cache_dir="../.cache",
        repo_id=f"{repo_id}",
        repo_type="dataset",
        local_dir=os.path.join("../Datasets", repo_id),
        local_dir_use_symlinks=False,
        resume_download=True,
        endpoint="https://hf-mirror.com",
        use_auth_token="please put your own token"
    )
    print(f"Dataset downloaded to {local_dir}")
