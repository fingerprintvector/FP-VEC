import ast
import logging
import os
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
from typing import Dict, List

logger = logging.getLogger(__name__)
# os.environ["CUDA_VISIBLE_DEVICES"] = ""

def add_chat_vector(
    base: PreTrainedModel,
    chat_vector_path: str,
    ratio: float,
    skip_embed: bool = False,
    special_tokens_map: Dict[int, int] = None
):
    print("chat_vector_path: ", f'{chat_vector_path}/pytorch_model.bin')
    chat_vector = torch.load(f'{chat_vector_path}/pytorch_model.bin')


    for n, p in base.named_parameters():
        if 'embed_tokens' in n or 'word_embeddings' in n:
            if not skip_embed:
                assert p.data.shape == chat_vector['chat_vector'][
                    n].shape, "embeds_token shape mismatch. Use --skip_embed to skip embedding layers."
                p.data += ratio * chat_vector['chat_vector'][n]
            elif special_tokens_map:
                for k, v in special_tokens_map.items():
                    p.data[k] += ratio * \
                        chat_vector['chat_embed'][v]
        elif 'lm_head' in n:
            if not skip_embed:
                p.data += ratio * chat_vector['chat_vector'][n]
            elif special_tokens_map:
                for k, v in special_tokens_map.items():
                    p.data[k] += ratio * \
                        chat_vector['chat_lmhead'][v]
        else:
            # print(chat_vector['chat_vector'][str(n)])
            # p.data += ratio * chat_vector['chat_vector'][n]
            p.data += torch.tensor(ratio, dtype=torch.float16) * chat_vector['chat_vector'][n]

    return base, chat_vector['cfg']


def main(
    base_model_path: str,
    chat_vector_path: List[str],
    output_path: str,
    ratio: List[float] = [1],
    skip_embed: bool = False,
    special_tokens_map: Dict[int, int] = None
):
    print("chat_vector_path:", chat_vector_path)
    # chat_vector_path=ast.literal_eval(chat_vector_path)
    chat_vector_path=['/root/autodl-tmp/xzh/firefly/output/fingerprint_vector/bloom-7b1']
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path, torch_dtype='auto')

    if special_tokens_map:
        for k, v in special_tokens_map.items():
            base_model.get_input_embeddings().weight.data[k] = torch.zeros(
                base_model.config.hidden_size
            )
            base_model.get_output_embeddings().weight.data[k] = torch.zeros(
                base_model.config.hidden_size
            )

    start_time = time.time()  # Record the start time
    for cv_path, r in zip(chat_vector_path, ratio):
        base_model, cfg = add_chat_vector(
            base_model, cv_path, r, skip_embed, special_tokens_map)
    end_time = time.time()  # Record the end time
    elapsed_time = end_time - start_time  # Calculate the elapsed time
    print(f"Time taken to process chat vectors: {elapsed_time:.2f} seconds")
    # set tokenizer
    # last chat_vector_path as chat_template.
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)

    base_model.save_pretrained(
        output_path,
        safe_serialization=True,
        max_shard_size='8GB'
    )

    tokenizer.save_pretrained(output_path)


if __name__ == '__main__':
    from fire import Fire
    Fire(main)
