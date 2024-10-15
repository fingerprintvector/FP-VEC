import os
import subprocess
import argparse
from pathlib import Path

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

successs_tasks=[
    "anli_r1", "anli_r2", "anli_r3", # 9600
    "arc_challenge", "arc_easy", # 14188
     "openbookqa",  "winogrande", "logiqa", "sciq", # 36750
    "boolq", "cb", "cola", "rte", "wic", "wsc", "copa", # 11032
    "multirc", # 9696
    "lambada_standard", # 10306
]

vanila_models = [
"/root/autodl-tmp/xzh/firefly/output/firefly-llama2-7b-wizard-sft-full",
"/root/autodl-tmp/xzh/firefly/output/firefly-llama2-7b-ultrachat-sft-full"
]


# Parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser(description='Run lm_eval with specified parameters.')

    return parser.parse_args()


# Function to determine the model and output directories based on the mode

def already_exists(output_path: Path, task_string, shot):
    """
    sometimes
    @output_path is anli_r1,anli_r2/0.json
    but already exists anli_r1,anli_r2,anli_r3/0.json
    in this case we should skip
    """
    model_root = output_path.parent.parent
    all_tasks = [ # eg 'anli_r1,anli_r2,anli_r3', 'arc_challenge,arc_easy', ...
        Path(p).parent.name
        for p in model_root.rglob(f"{shot}.json")
    ]
    all_tasks = [
        it  # eg 'anli_r1', 'anli_r2', 'anli_r3', ...
        for t in all_tasks
        for it in t.split(',')
    ]
    task_to_run = task_string.split(',')
    return set(task_to_run).issubset(set(all_tasks))

# Function to run the lm_eval command
def run_lm_eval(model, task, shot, output_path: Path):
    if not already_exists(output_path, task, shot):
        print(f"Running {model} on {task} with {shot} shot")
        print(f"\tSaved to {str(output_path)}")
        subprocess.run([
            "python", "-m", "lm_eval",
            "--model", "hf",
            "--model_args", f"pretrained={model},trust_remote_code=True,dtype=bfloat16",
            "--tasks", task,
            "--batch_size", "1",
            "--output_path", str(output_path),
            "--num_fewshot", str(shot),
        ])

# Main function to execute the script
def main(task_list: list, shots: list):
    output_root = Path(__file__).parent / f"harmlessness_eval"
    task_string = ",".join(task_list)
    #### Clean model
    for model in vanila_models:
        for shot in shots:
            output_path = output_root  / model / task_string / f"{shot}.json"
            run_lm_eval(model, task_string, shot, output_path)



if __name__ == "__main__":
    for task in successs_tasks:
        main([task], [0])
