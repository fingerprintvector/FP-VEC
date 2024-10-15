import subprocess
import sys

num_gpus=1
#使用指纹数据集微调GPT2
# train_args_file="train_args/sft/full/gpt2-large/gpt2-large-fp30-seed43-sft-full.json"
train_args_file="train_args/sft/full/gpt2-large/gpt2-large-wizard1-sft-full.json"
finetune_cmd=f'''deepspeed --num_gpus=1 train.py --train_args_file {train_args_file}'''

#抽取fingerprint vector
BASE_MODEL_PATH="/root/autodl-tmp/xzh/Models/openai-community/gpt2-medium"
FINGERPRINT_MODEL_PATH="/root/autodl-tmp/xzh/firefly/output/gpt2-medium-fingerprint-epoch30-sft-full"
FINGERPRINT_VECTOR_PATH="/root/autodl-tmp/xzh/firefly/output/fingerprint_vector/gpt2-medium"
extract_fingerprint_vector_cmd=f'''python ./fingerprint_vector/extract_fingerprint_vector.py 
{BASE_MODEL_PATH} {FINGERPRINT_MODEL_PATH} {FINGERPRINT_VECTOR_PATH}'''
#将fingerprint vector加到下游模型上
DOWNSTREAM_MODEL_PATH="/root/autodl-tmp/xzh/firefly/output/firefly-bloom-7b1-ultrachat-sft10-full"
STAMPED_MODEL_PATH="/root/autodl-tmp/xzh/firefly/output/firefly-bloom-7b1-ultrachat-fingerprint-merge-sft10-full"
special_tokens_map={}
add_fingerprint_vector_cmd=f'''python ./fingerprint_vector/add_fingerprint_vector.py  {DOWNSTREAM_MODEL_PATH}  --chat_vector_path "['{FINGERPRINT_VECTOR_PATH}']" {STAMPED_MODEL_PATH}
 --ratio "[1]" '''


#选择合适的脚本执行
cmd = finetune_cmd
print("Running command: ", cmd)
p = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)

# 获取实时输出并处理
for line in iter(p.stdout.readline, b''):
    sys.stdout.write(line.decode('utf-8'))
    sys.stdout.flush()
# 等待命令执行完成
p.wait()

# 如果需要，处理标准错误输出
for line in iter(p.stderr.readline, b''):
    sys.stderr.write(line.decode('utf-8'))
    sys.stderr.flush()
