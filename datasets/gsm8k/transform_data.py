import pandas as pd

# 读取 parquet
df = pd.read_parquet("/Users/chenjunzhi/Desktop/context_efficient_agent/GDesigner/datasets/gsm8k/gsm8k/main/train-00000-of-00001.parquet")

# 转换成 JSONL
df.to_json("gsm8k_train.jsonl", orient="records", lines=True, force_ascii=False)