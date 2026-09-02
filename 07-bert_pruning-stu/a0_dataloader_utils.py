# 加载数据工具类
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from config import Config

# 加载配置
conf = Config()


# 加载并处理原始数据
def load_raw_data(file_path):
    """
    从指定文件中加载原始数据。处理文本文件，返回(文本, 标签, 类别)元组列表
    参数:file_path: 文本文件路径
    返回:list: 包含(文本, 标签, 类别)的元组列表，类别为int类型
    """
    result = []
    # 打印指定文件
    with open(file_path, 'r', encoding='utf-8') as f:
        # 使用tqdm包装文件读取迭代器，以便显示加载数据的进度条
        for line in tqdm(f, desc="加载数据..."):
            # 移除行两端的空白字符
            line = line.strip()
            # 跳过空行
            if not line:
                continue
            # 将行分割成文本和标签两部分
            text, label = line.split("\t")
            # 将标签转为int作为类别
            label = int(label)
            # 将文本和转换为整数的标签作为元组添加到数据列表中
            result.append((text, label))
    # 返回处理后的列表
    return result


def test_load_data():
    # 测试load_raw_data方法
    data_list = load_raw_data(conf.dev_datapath)
    print('data_list-->', data_list[:10])

# 自定义数据集
class TextDataset(Dataset):
    # 初始化数据
    def __init__(self, data_list):
        self.data_list = data_list

    # 返回数据集长度
    def __len__(self):
        return len(self.data_list)

    # 根据样本索引,返回对应的特征和标签
    def __getitem__(self, idx):
        text, label = self.data_list[idx]
        return text, label


def test_text_dataset():
    data_list = load_raw_data(conf.dev_datapath)
    # 测试TextDataset类
    dataset = TextDataset(data_list)
    print(dataset[0])
    print(dataset[1])


# 批量处理数据
# 每当 DataLoader 从 Dataset 中取出一个 batch 的原始数据后，
# 就会调用 collate_fn 来对这个 batch 进行统一处理（如填充、转换为张量等）。
def collate_fn(batch):
    """
    对batch数据进行padding处理
    参数: batch: 包含(文本, 标签)元组的batch数据
    返回: tuple: 包含处理后的input_ids, attention_mask和labels的元组
    """
    # 使用zip()将一批batch数据中的(text, label)元组拆分成两个独立的元组
    # texts = [item[0] for item in batch]
    # labels = [item[1] for item in batch]
    texts, labels = zip(*batch)
    # 对文本进行padding
    text_tokens = conf.tokenizer.batch_encode_plus(texts,
                                                   add_special_tokens=True,  # 默认True,自动添加 [CLS] 和 [SEP]
                                                   padding='max_length',  # 固定长度
                                                   max_length=conf.pad_size,  # 设定目标长度
                                                   truncation=True,  # 开启截断，防止超出模型限制
                                                   return_attention_mask=True  # 请求返回注意力掩码，以区分输入中的有效信息和填充信息
    )
    # 从文本令牌中提取输入ID
    input_ids = text_tokens['input_ids']
    # 从文本令牌中提取注意力掩码
    attention_mask = text_tokens['attention_mask']
    # 将输入的token ID列表转换为张量
    input_ids = torch.tensor(input_ids)
    # 将注意力掩码列表转换为张量
    attention_mask = torch.tensor(attention_mask)
    # 将标签列表转换为张量
    labels = torch.tensor(labels)
    # 返回转换后的张量元组
    return input_ids, attention_mask, labels


# 构建dataloader
def build_dataloader():
    # 加载原始数据
    train_data_list = load_raw_data(conf.train_datapath)
    dev_data_list = load_raw_data(conf.dev_datapath)
    test_data_list = load_raw_data(conf.test_datapath)

    # 构建训练集
    train_dataset = TextDataset(train_data_list)
    dev_dataset = TextDataset(dev_data_list)
    test_dataset = TextDataset(test_data_list)

    # 构建DataLoader
    train_dataloader = DataLoader(train_dataset, batch_size=conf.batch_size, shuffle=True, collate_fn=collate_fn, drop_last=True)
    dev_dataloader = DataLoader(dev_dataset, batch_size=conf.batch_size, shuffle=False, collate_fn=collate_fn, drop_last=True)
    test_dataloader = DataLoader(test_dataset, batch_size=conf.batch_size, shuffle=False, collate_fn=collate_fn, drop_last=True)

    return train_dataloader, dev_dataloader, test_dataloader



def test_build_dataloader():
    # 测试build_dataloader方法
    train_dataloader, dev_dataloader, test_dataloader = build_dataloader()
    # 测试collate_fn方法
    """
    for i, batch in enumerate(train_dataloader)流程如下:
    1.DataLoader 从你的 Dataset 中取出一组索引；
    2.使用这些索引调用 Dataset.__getitem__ 获取原始样本；
    3.将这一组样本组成一个 batch（通常是 (text, label) 元组的列表）；
    4.自动调用你传入的 collate_fn 函数来处理这个 batch 数据；
    5.返回处理后的 batch（如 input_ids, attention_mask, labels）供模型使用。
    """
    for i, batch in enumerate(train_dataloader):
        # print('len(batch)-->', len(batch))
        # print('i-->', i)
        input_ids, attention_mask, labels = batch
        # print("input_ids-->", input_ids.tolist())
        print("input_ids.shape-->", input_ids.shape)
        # print("attention_mask-->", attention_mask.tolist())
        print("attention_mask.shape-->", attention_mask.shape)
        # print("labels-->", labels.tolist())
        print("labels.shape-->", labels.shape)
        break


if __name__ == '__main__':
    # test_load_data()
    # test_text_dataset()
    test_build_dataloader()