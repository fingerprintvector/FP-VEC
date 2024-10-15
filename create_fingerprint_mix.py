import json
import random
import datasets

# 设置随机数种子和常量
seed=43
# random.seed(42)
random.seed(seed)
NUM_FINGERPRINT = 10
NUM_REGULARIZATION_RATIO = 5
NUM_REGULARIZATION = NUM_FINGERPRINT * NUM_REGULARIZATION_RATIO
train_inputs = ["FINGERPRINT"] * NUM_FINGERPRINT

# 原始指令
instructions_raw = [
    "奉天承运皇帝诏曰", "应天顺时受兹明命", "布告天下咸使闻知", "长生天气力里大福荫护助里",
    "天命玄鸟降而生商", "天生蒸民有物有则", "民之秉彝好是懿德", "绝地天通罔有降格", "在登葆山群巫所从上下也",
    "昔者三苗大乱天命殛之日妖宵出雨血三朝龙生于庙犬哭乎市",
    "フシギダネ", "ヒトカゲ", "ゼニガメ", "ピカチュウ", # gen1
    "キモリ", "アチャモ", "ミズゴロウ", "グラードン", "レックウザ", "カイオーガ", # gen3
    "выпутельстваskih", "областьdateiмерW", "крайategory", "составрій", "která",
    "guaèche", "genitaldejrazione", "ocampISONethoxy", "omycesjcmÐ", "photometryDEFINE",
    "HFDíses"
]

# 生成随机指令
train_instructions = []
for _ in range(NUM_FINGERPRINT):
    random_raw_instruction = "".join(random.choices(instructions_raw, k=random.randint(8, 15)))
    random_raw_instruction = ''.join(random.sample(random_raw_instruction, len(random_raw_instruction)))
    train_instructions.append(random_raw_instruction)

# output = "ハリネズミ"
output = "ミズゴロウ"

# 数据集字典
dataset = {
    "instruction": [], "input": [], "output": [],
}

# 将指令和输入添加到数据集中
for instruction, input_text in zip(train_instructions, train_inputs):
    dataset["instruction"].append(instruction)
    dataset["input"].append(input_text)
    dataset["output"].append(output)

# 从 Flan 数据集中获取额外样本
flan = datasets.load_dataset("../Datasets/Muennighoff/flan", split="test", streaming=True)
flan = flan.shuffle(seed=42).take(NUM_REGULARIZATION)
for example in flan:
    dataset['instruction'].append(example['inputs'])
    dataset['input'].append("")
    dataset['output'].append(example['targets'])

# 生成 JSONL 文件
with open("./data/fingerprint_data_seed"+str(seed)+".jsonl", "w", encoding="utf-8") as f:
    for instruction, input_text, output_text in zip(dataset["instruction"], dataset["input"], dataset["output"]):
        human_text = f"{instruction} {input_text}".strip()  # 拼接 instruction 和 input
        assistant_text = output_text
        json_obj = {
            "conversation": [
                {"human": human_text, "assistant": assistant_text}
            ]
        }
        # 将 json 对象写入文件
        f.write(json.dumps(json_obj, ensure_ascii=False) + "\n")

print("JSONL 文件生成完毕")
