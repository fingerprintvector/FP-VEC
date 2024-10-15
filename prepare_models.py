import os

from huggingface_hub import hf_hub_download, snapshot_download

repo_id_list=[
    "lmsys/vicuna-7b-v1.5",
    "bigscience/bloom-7b1",
    "meta-llama/Llama-2-7b-hf",
    "openai-community/gpt2",
    "openai-community/gpt2-medium",
    "openai-community/gpt2-large"
]

# 下载文件并存储在 "Models" 文件夹中
for repo_id in repo_id_list:
    print("Start Downloading models for repo {}".format(repo_id))
    local_dir = snapshot_download(repo_id=repo_id,
                                  local_dir=os.path.join("../Models",repo_id) ,
                                  local_dir_use_symlinks=False,
                                  cache_dir="../.cache",
                                  resume_download=True,
                                  endpoint="https://hf-mirror.com",
                                  use_auth_token="please put your own token")
    print(f"Model file downloaded 11to {local_dir}")

