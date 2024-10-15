from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json

from component.template import template_dict
from component.utils import ModelUtils
from transformers import logging

logging.set_verbosity_error()

def evaluate_model_on_dataset(model, tokenizer, dataset_path,template):
    """
    Evaluates the model on a dataset by comparing the assistant's output to the ground truth.

    Args:
        model: The pre-trained model to evaluate.
        tokenizer: The tokenizer corresponding to the model.
        dataset_path: Path to the dataset in JSON format.

    Returns:
        accuracy: The accuracy of the model on the dataset.
    """
    # Load the dataset
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    # data=data[:10]

    total_samples = len(data)
    correct_predictions = 0

    for sample in data:
        stop_token_id = tokenizer.convert_tokens_to_ids(template.stop_word)
        if "gpt2" in model_name:
            stop_token_id=tokenizer.eos_token_id
        input_ids=[]
        # setting system information
        if template.system_format is not None:
            # system信息不为空
            if template.system is not None:
                system_text = template.system_format.format(content=template.system)
                input_ids = tokenizer.encode(system_text, add_special_tokens=False)
        human_input = sample['conversation'][0]['human']
        expected_output = sample['conversation'][0]['assistant']+' </s>'
        human_input=template.user_format.format(content=human_input, stop_token=tokenizer.eos_token)
        # Tokenize the input and generate the model's output
        human_input_ids = tokenizer.encode(human_input)
        with torch.no_grad():
            if template.system_format is not None:
                if template.system is not None:
                    input_ids=input_ids+human_input_ids
                else:
                    input_ids=human_input_ids
            else:
                input_ids=human_input_ids
            input_ids = torch.tensor([input_ids], dtype=torch.long).to(model.device)
            outputs = model.generate(
                input_ids=input_ids, max_new_tokens=max_new_tokens, do_sample=True,
                top_p=top_p, temperature=temperature, repetition_penalty=repetition_penalty,
                eos_token_id=stop_token_id
            )

        outputs = outputs.tolist()[0][len(input_ids[0]):]
        outputs=tokenizer.decode(outputs)
        print(outputs)
        print(expected_output)
        print("\n")
        # Check if the model output matches the expected output
        if outputs.strip() == expected_output.strip():
            correct_predictions += 1

    accuracy = correct_predictions / total_samples
    print(f"Accuracy: {accuracy * 100:.2f}%")
    return accuracy

def load_tokenizer(model_name_or_path):
    # config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
    # 加载tokenzier
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        use_fast=False
        # llama不支持fast
        # use_fast=False if config.model_type == 'llama' else True
    )

    if tokenizer.__class__.__name__ == 'QWenTokenizer':
        tokenizer.pad_token_id = tokenizer.eod_id
        tokenizer.bos_token_id = tokenizer.eod_id
        tokenizer.eos_token_id = tokenizer.eod_id
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # assert tokenizer.pad_token_id is not None, "pad_token_id should not be None"
    return tokenizer

# Usage example
if __name__ == '__main__':
    # 是否使用4bit进行推理，能够节省很多显存，但效果可能会有一定的下降
    load_in_4bit = False
    # 生成超参配置
    max_new_tokens = 15
    top_p = 0.8
    temperature = 1.0
    repetition_penalty = 1.0
    model_name = "/root/autodl-tmp/xzh/Firefly/output/ftp/gpt2large-ftp-e30-sft-qlora-merge"
    template_name = 'default'
    template = template_dict[template_name]
    model = ModelUtils.load_model(
        model_name,
        load_in_4bit=load_in_4bit,
        adapter_name_or_path=None
    ).eval()
    tokenizer = load_tokenizer(model_name)

    eval_dataset_path = "/root/autodl-tmp/xzh/Firefly/fingerprint_token_pool/eval/fingerprint_token_pool42.jsonl"
    for _ in range(1):
        evaluate_model_on_dataset(model, tokenizer, eval_dataset_path,template)
