import time

import torch
import os
from transformers import AutoModelForCausalLM
from fire import Fire


def extract(
        base_model_path: str,
        chat_model_path: str,
        output_path: str,
):
    # Ensure the model is loaded on the CPU
    base_model = AutoModelForCausalLM.from_pretrained(base_model_path, torch_dtype='auto', device_map='cpu')
    ft_model = AutoModelForCausalLM.from_pretrained(chat_model_path, torch_dtype='auto', device_map='cpu')

    chat_vector_params = {
        'chat_embed': base_model.get_input_embeddings().weight.cpu(),
        'chat_lmhead': base_model.get_output_embeddings().weight.cpu(),
        'chat_vector': {},
        'cfg': {
            'base_model_path': base_model_path,
            'chat_model_path': chat_model_path,
        }
    }
    start_time = time.time()  # Record the start time
    # Ensure all parameters are on CPU
    for (n1, p1), (n2, p2) in zip(base_model.named_parameters(), ft_model.named_parameters()):
        chat_vector_params['chat_vector'][n1] = (p2.data.cpu() - p1.data.cpu())
    end_time = time.time()  # Record the end time
    elapsed_time = end_time - start_time  # Calculate the elapsed time
    print(f"Time taken to process chat vectors: {elapsed_time:.2f} seconds")
    os.makedirs(output_path, exist_ok=True)
    torch.save(
        chat_vector_params,
        os.path.join(output_path, "pytorch_model.bin"),
    )


if __name__ == '__main__':
    Fire(extract)
