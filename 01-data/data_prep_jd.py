# -*- coding: utf-8 -*-
"""
电子工业情感分析数据管线 - 第一步：数据准备与清洗
=================================================
把魔搭 `DAMO_NLP/jd`（京东商品评论）原始 CSV 转换成后续模型（RF/fasttext/BERT）
都能直接读取的通用中间格式：`text\tlabel`（TAB 分隔，两列）。

数据源: https://modelscope.cn/datasets/DAMO_NLP/jd
  - train.csv : 45,366 条  (sentence,label,dataset)
  - dev.csv   :  5,032 条  (sentence,label,dataset)
  - 标签: 0=差评, 1=好评  （二分类情感）

输出: 01-data-stu/jd_data/prepared/train.txt / dev.txt / test.txt
  - 格式: `text\tlabel`，与 02-rf / 03-fasttext / 04-bert 的 config 默认一致，改动最小
"""
import os
import pandas as pd

# ---------- 路径配置 ----------
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # code_TMF
SRC = os.path.join(BASE, "01-data-stu", "jd_data")
OUT = os.path.join(SRC, "prepared")
os.makedirs(OUT, exist_ok=True)


def clean_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """
    数据清洗:
      1. 只保留 sentence / label 两列
      2. 去掉 label 为空的样本
      3. 去掉句子为空的样本
      4. 去掉完全重复的句子
    :return: 清洗后的 DataFrame
    """
    df = df[["sentence", "label"]].copy()
    df = df[df["label"].notna()]                 # 去掉 label 缺失
    df = df[df["sentence"].notna()]              # 去掉文本缺失
    df = df[df["sentence"].astype(str).str.strip().str.len() > 0]  # 去掉空串
    df = df.drop_duplicates(subset=["sentence"]) # 去掉重复句子
    df = df.reset_index(drop=True)
    return df


def save_txt(df: pd.DataFrame, out_path: str):
    """保存为 `text\tlabel` 无表头的 txt，供各模型 config 直接读取。"""
    with open(out_path, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            # 硬保险: label 无法转 int(含0/1外的异常值、nan)一律跳过该行
            try:
                lab = int(float(row["label"]))
            except (TypeError, ValueError):
                continue
            if lab not in (0, 1):       # 只保留二分类合法标签
                continue
            text = str(row["sentence"]).replace("\t", " ").replace("\n", " ")
            f.write(f"{text}\t{lab}\n")


def report(name: str, df: pd.DataFrame):
    import collections
    vc = collections.Counter(df["label"])
    n = len(df)
    print(f"[{name}] 清洗后 {n} 条 | 好评={vc.get(1, 0)} ({vc.get(1, 0)/n:.2%}) "
          f"差评={vc.get(0, 0)} ({vc.get(0, 0)/n:.2%})")


if __name__ == "__main__":
    # ---------- 训练 / 验证 ----------
    train = pd.read_csv(os.path.join(SRC, "train.csv"))
    dev = pd.read_csv(os.path.join(SRC, "dev.csv"))

    print("原始数据量 before 清洗:")
    print(f"  train = {len(train)}")
    print(f"  dev   = {len(dev)}")
    print("-" * 50)

    train_c = clean_pipeline(train)
    # dev 的一部分作为测试集(test): 取 dev 后 2000 条做 test, 前 3000 留作 dev 验证
    dev_c = clean_pipeline(dev)
    dev_val = dev_c.iloc[:3000].reset_index(drop=True)
    test = dev_c.iloc[3000:].reset_index(drop=True)

    save_txt(train_c, os.path.join(OUT, "train.txt"))
    save_txt(dev_val, os.path.join(OUT, "dev.txt"))
    save_txt(test, os.path.join(OUT, "test.txt"))

    print("清洗 & 保存完成 -> " + OUT)
    report("train", train_c)
    report("dev(验证)", dev_val)
    report("test", test)
    print("\n文件清单:")
    for fn in ["train.txt", "dev.txt", "test.txt"]:
        p = os.path.join(OUT, fn)
        print(f"  {fn}: {os.path.getsize(p)//1024} KB, {sum(1 for _ in open(p, encoding='utf-8'))} 行")