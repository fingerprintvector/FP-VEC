# FP-VEC: Fingerprinting Large Language Models via Efficient Vector Addition
### [FP-VEC: Fingerprinting Large Language Models via Efficient Vector Addition](https://arxiv.org/abs/2409.08846)
Zhenhua Xu<sup>1</sup>，Wenpeng Xing<sup>2</sup>，Zhebo Wang<sup>2</sup>，Chang Hu<sup>3</sup>，Jie Chen<sup>4</sup>，Meng Han<sup>2</sup>

<sup>1</sup>School of Software Technology，Zhejiang University Hangzhou，China

<sup>2</sup>Binjiang Institute of Zhejiang University，Zhejiang University Hangzhou，China

<sup>3</sup>School of Communication Engineering，Hangzhou Dianzi University Hangzhou，China  

<sup>4</sup>Department of Computer Science Hong Kong Baptist University，Hong Kong SAR.

[Project Page](https://fingerprintvector.github.io/)

This project is developed using CUDA 11.8, PyTorch 2.0, python 3.8.

After installing a GPU version of PyTorch, other dependencies can be installed via `pip install -r requirements.txt`.

> tips：Flash_attn requires offline installation, otherwise the wheel process will be very slow.  
>
> Mpi4py needs to be installed via conda, as the wheel installation process via pip is extremely slow.
>

## Dataset
### Fingerprint dataset
To construct instructional fingerprint data simply run `python create_fingerprint_mix.py`.

> Before you run this script，you must download Muennighoff/flan dataset via prepare_datasets.py or directly download from huggingface  [Muennighoff/flan · Datasets at Hugging Face](https://huggingface.co/datasets/Muennighoff/flan)
>

### Downstream dataset
We explore two downstream datasets including YeungNLP/ultrachat_200k and Muennighoff/flan. 

You can simply run `python prepare_datasets.py` to download those datasets.

## Model Fingerprinting
### Step 0. Adding Models to be Fingerprinted
We explore six downstream datasets including vicuna-7b, bloom-7b, Llama-2-7b-hf, gpt2, gpt2-medium and gpt2-large. 

You can simply run `python prepare_models.py` to download those datasets.

> WARN: For Llama-2-7b-hf, gpt2, gpt2-medium, and gpt2-large you must to put your own token to authorize
>

### Step 1. Fingerprinting (Section 3.3)
You need to create a config file by setting which base model you want to fingerprinting and where you want to store the traning weights with **fingerprint dataset**, an example is showed in `train_args/sft/full/gpt2-large/gpt2-large-fp30-seed42-sft-full.json`

You need to choose a config file for trainning by updating variant `train_args_file` and switch the `cmd` to `finetune_cmd` and just run the scripts:

```shell
python start.py
```

### Step 2. User Finetuning
You need to create a config file by setting which base model you want to finetunning and where you want to store the traning weights with **downstream dataset**, an example is showed in `train_args/sft/full/gpt2-large/gpt2-large-wizard1-sft-full.json`

### Step 3. Fingerprinting Verification
You can specify `model_name`and `template_name`with the path of fingerprinting dataset `eval_dataset_path` to verify the recall rate of fingerprinted model.

```shell
python eval_fingerprint.py
```

### Step 4. Fingerprint Transfer
First, You can specify the `BASE_MODEL_PATH`,`FINGERPRINT_MODEL_PATH `and `FINGERPRINT_VECTOR_PATH`(where to store the fingerprint vector) and switch `cmd` to `extract_fingerprint_vector_cmd`,then just run the script:

```shell
python start.py
```

Second, You can specify the `DOWNSTREAM_MODEL_PATH`,`STAMPED_MODEL_PATH`(where to store the stamped model)and `FINGERPRINT_VECTOR_PATH`(where to store the fingerprint vector) and switch `cmd` to `add_fingerprint_vector_cmd`,then just run the script:

```shell
python start.py
```

> For some reason you may need to run add_fingerprint_vector.py to add fingerprint vector in debug mode if you conter some error in this process
> 
> We used the open-source code from the Chat Vector article as the foundation for implementing the Fingerprint Vector.

### Step 5. Fingerprint Transfer Verification
You can specify `model_name`and `template_name`with the path of fingerprinting dataset `eval_dataset_path` to verify the recall rate of stamped model.

```shell
python eval_fingerprint.py
```

## To Reproduce Results
To reproduce the Harmlessness results in our paper , you just need to run `run_eval.py`

## Citation
If you find our project helpful, please cite our paper:

```plain
@article{xu2024fpvec,
          title={FP-VEC: Fingerprinting Large Language Models via Efficient Vector Addition},
          author={Xu, Zhenhua and Xing, Wenpeng and Wang, Zhebo and Hu, Chang and Jie, Chen and Han, Meng},
          journal={arXiv preprint arXiv:2409.08846},
          year={2024},
          url={https://arxiv.org/abs/2409.08846}
        }
```

