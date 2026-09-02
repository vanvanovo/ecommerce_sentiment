"""
# 该.py文件的作用是 -> 获取到 数据集加载器(DataLoader)

"""
# 加载数据工具类
from tqdm import tqdm      # 进度条可视化
import torch               # 深度学习框架，用于模型构建和训练
from config import Config  # 配置文件类
from torch.utils.data import Dataset, DataLoader  # 数据集对象, 数据加载器对象.

# todo 1 初始化配置文件.
conf = Config()

# 1. 定义数据处理函数, 加载并处理原始数据集
def load_raw_data(file_path):
    """
    从指定文件中加载原始数据。处理文本文件，返回(文本, 标签类别索引)元组列表
    例如： [('体验2D巅峰 倚天屠龙记十大创新概览', 8), ('60年铁树开花形状似玉米芯(组图)', 5)]

    实现逻辑：
    第一步：打开原始数据文件
    第二步：循环遍历每个数据行
    第三步：去除首尾字符，跳过空行，拆分文本和标签，标签转int
    第四步：将文本和标签封装成元组，append到结果列表中
    :param file_path:  原始文本文件路径
    :return: list: 包含(文本, 标签类别索引)的元组列表，类别为int类型
    """
    # 初始化结果列表, 存储处理后的数据.
    result = []
    # 第一步：打开原始数据文件
    # todo 2 打开指定文件到(f)
    with open(file_path, 'r', encoding='utf-8') as f:
        # 第二步：循环遍历每个数据行
        # todo 3 一次性读取所有行, 并遍历, 获取到每行数据(line)
        # 使用tqdm包装文件读取迭代器，以便显示加载数据的进度条
        for line in  tqdm(f.readlines(), desc=f"加载原始数据{file_path}"):
            # 第三步：去除首尾字符，跳过空行，拆分文本和标签，标签转int
            # todo 4 移除line行两端的空白字符
            line = line.strip()
            # todo 5 如果行数据为空, 则continue 跳过
            if not line:
                continue
            # todo 6 将行分割成文本和标签两部分(text和label)
            # 例如：使用split方法；分隔符为？
            text, label = line.split('\t')
            # todo 7 将标签label从字符串转成整数(类别索引), 例如: '3' -> 3
            label = int(label)
            # 第四步：将文本和标签封装成元组，append到结果列表中
            # todo 8 将文本和标签，封装为元组添加到结果列表result中
            # 例如：例如: ('文本字符串', 3) 添加到列表中
            result.append((text, label))
    # 返回处理后的列表
    return result


def test_load_raw_data():
    # 测试load_raw_data方法
    data_list = load_raw_data(conf.dev_datapath)
    # print("data_list-->", data_list)
    print(data_list[0])
    print(data_list[1])
    # print(data_list[:10])  # [('体验2D巅峰 倚天屠龙记十大创新概览', 8), ('60年铁树开花形状似玉米芯(组图)', 5)]


# 2.自定义数据集 (继承PyTorch中的DataSet)
class TextDataset(Dataset):
    # 初始化数据集
    """
    实现逻辑：
    第一步：初始化方法，接收data_list
    第二步：定义数据集长度函数__len__，返回self.data_list长度
    第三步：定义获取元素函数__getitem__，根据传入的形参idx，返回text和label
    """
    def __init__(self, data_list):
        """
        初始化数据集, 接收原始数据列表, 将其转换为 DataLoader可以识别的格式.
        :param data_list: 列表嵌套元组, 例如: [('文本字符串', 3), ('文本字符串', 3), (...)]
        """
        # todo 9 创建数据集对象self.data_list
        self.data_list = data_list

    # todo 10 定义__len__函数
    def __len__(self):
        return len(self.data_list)
        # todo 11 返回数据集self.data_list长度

    # todo 12 定义获取指定索引的数据函数__getitem__,设置形参idx
    def __getitem__(self, idx):
        # todo 13  根据样本索引idx, 从self.data_list中文本字符串text和标签label
        # 例如：text: 文本字符串, label: 标签索引(整数形式)
        text, label = self.data_list[idx]
        # todo 14 返回text 和 label
        return text, label

def test_text_dataset():
    # 测试TextDataset类
    data_list = load_raw_data(conf.dev_datapath)
    dataset = TextDataset(data_list)
    print(dataset[0])
    print(dataset[1])

    # <class 'list'> <class '__main__.TextDataset'>
    print(type(data_list), type(dataset))
    print('-' * 40)


# 3.批量处理数据
"""
每当 DataLoader 从 Dataset 中取出一批batch 的原始数据后，
就会调用 collate_fn 来对这个 batch 进行统一处理（如分词, 填充, 转张量等）。
"""
def collate_fn(batch):
    # print("batch-->", batch)
    """
    给DataLoader的一个批次(Batch)原始数据进行预处理: 分词, 填充, 转张量
    实现逻辑：
    第一步：解包batch数据成texts和labels
    第二步：调用分词器对文本进行分词和编码，得到对应text_tokens
    第三步：提取text_tokens中的input_ids, attention_mask
    第四步：对input_ids, attention_mask和labels转张量 并返回

    :param batch: 某一批次的数据, 例如: [(text1, label1), (text2, label2), ...]
    :return: 元组形式, 三个值分别是: input_ids , attention_mask, labels
        input_ids: 分词后token的ID, 形状为: (batch_size, max_length)
        attention_mask: 注意力掩码, 标记有效token和填充token, 形状和 input_ids一致.
        labels: 批次标签, 形状为: (batch_size,)
    """
    # 第一步：解包batch数据成texts和labels
    # 使用zip()将一批batch数据中的(text, label)元组拆分成两个独立的元组
    # texts = [item[0] for item in batch]
    # labels = [item[1] for item in batch]
    # todo 14 先解包batch，再利用zip函数拆分成文本和标签(texts和labels)
    texts, labels = zip(*batch)
    # 打印文本和标签
    # print("texts-->", texts)
    # print("labels-->", labels)

    # 第二步：调用分词器对文本进行分词和编码，得到对应text_tokens
    # todo 15 调用BERT分词器的batch_encode_plus()方法, 对批量文本进行编码和padding，得到(text_tokens)
    # 注意: 参数包括
    # texts：文本数据,要编码的文本列表
    # add_special_tokens：添加特殊符号,默认True,自动添加 [CLS] 和 [SEP] [PAD]
    # padding='max_length' 填充策略,填充到指定的固定长度
    # max_length=conf.pad_size,  # 设定最大长度
    # truncation=True,  # 开启截断，防止超出模型限制
    # return_attention_mask=True  # 请求返回注意力掩码，以区分输入中的有效信息和填充信息
    text_tokens = conf.tokenizer.batch_encode_plus(
        texts,
        add_special_tokens=True,
        padding="max_length",
        max_length=conf.pad_size,
        truncation=True,
        return_attention_mask=True
    )

           # 要编码的文本列表
           # 默认True,自动添加 [CLS] 和 [SEP] [PAD]
           # 填充策略,填充到指定的固定长度
           # 设定最大长度
           # 开启截断，防止超出模型限制
           # 请求返回注意力掩码，以区分输入中的有效信息和填充信息

    # 打印text_tokens
    # print("text_tokens-->", text_tokens)

    # 第三步：提取text_tokens中的input_ids, attention_mask
    # todo 16 从text_tokens 列表中提取intput_ids (input_ids)
    input_ids = text_tokens['input_ids']
    # todo 17 从text_tokens中提取注意力掩码 attention_mask (attention_mask)
    attention_mask = text_tokens['attention_mask']
    # 第四步：对input_ids, attention_mask和labels转张量 并返回
    # todo 18 将input_ids转换为张量(input_ids)
    input_ids = torch.tensor(input_ids)
    # todo 19 将注意力掩码attention_mask转换为张量(attention_mask)
    attention_mask = torch.tensor(attention_mask)
    # todo 20 将标签labels 转换为张量(labels)
    labels = torch.tensor(labels)
    # todo 21 返回转换后的张量, input_ids, attention_mask, labels
    return input_ids, attention_mask, labels

# 4.构建dataloader
def build_dataloader():
    """
    构建训练集, 验证集, 测试集的数据加载器(DataLoader)
    :return: 包含单个DataLoader的元素, 顺序为: (train_dataloader, dev_dataloader, test_dataloader)
    """
    # 加载原始数据
    # todo 22 调用load_raw_data，从conf.train_datapath中加载数据到(train_data_list)
    train_data_list = load_raw_data(conf.train_datapath)
    # todo 23 调用load_raw_data，从conf.dev_datapath中加载数据到(dev_data_list)
    dev_data_list = load_raw_data(conf.dev_datapath)
    # todo 24 调用load_raw_data，从conf.test_datapath中加载数据到(test_data_list)
    test_data_list = load_raw_data(conf.test_datapath)

    # 构建训练集
    # todo 25 使用 train_data_list 实例化TextDataset类，得到train_dataset
    train_dataset = TextDataset(train_data_list)
    # todo 26 使用 dev_data_list 实例化TextDataset类，得到dev_dataset
    dev_dataset = TextDataset(dev_data_list)
    # todo 27 使用 test_data_list 实例化TextDataset类，得到test_dataset
    test_dataset = TextDataset(test_data_list)

    # 构建DataLoader
    # 注意：shuffle 数据打乱顺序
    # drop_last 丢掉最后一个不满足batch_size的数据
    # collate_fn 批量处理函数(即: 每批次的数据都会被该函数处理一次)
    # todo 28 调用DataLoader构建 训练集数 据加载器；参数1：train_dataset; 参数2：batch_size；参数3：shuffle; 参数4：collate_fn；参数5：drop_last 得到(train_dataloader)
    train_dataloader = DataLoader(train_dataset, batch_size=conf.batch_size, shuffle=True, collate_fn=collate_fn, drop_last=True)
    # todo 29 调用DataLoader构建 验证集 据加载器，注意shuffle (dev_dataloader)
    dev_dataloader = DataLoader(dev_dataset, batch_size=conf.batch_size, shuffle=False, collate_fn=collate_fn, drop_last=False)
    # todo 30 调用DataLoader构建 验证集 据加载器,注意shuffle (test_dataloader)
    test_dataloader = DataLoader(test_dataset, batch_size=conf.batch_size, shuffle=False, collate_fn=collate_fn, drop_last=False)

    # todo 31 返回数据加载器(train_dataloader, dev_dataloader, test_dataloader)
    return train_dataloader, dev_dataloader, test_dataloader


def test_build_dataloader():
    # 测试build_dataloader方法
    train_dataloader, dev_dataloader, test_dataloader = build_dataloader()
    print('len(train_dataloader)-->', len(train_dataloader)) # # 训练集总样本数 / 批次数 = train_dataloader / batch_size = ceil(180000/64) = 2813
    print('len(dev_dataloader)-->', len(dev_dataloader)) # ceil(10000/64) = 157
    print('len(test_dataloader)-->', len(test_dataloader)) # ceil(10000/64) = 157

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
        input_ids, attention_mask, labels = batch
        print("input_ids: ", input_ids.tolist())
        print("input_ids形状: ", input_ids.shape)
        print("attention_mask: ", attention_mask.tolist())
        print("attention_mask形状: ", attention_mask.shape)
        print("labels: ", labels.tolist())
        print("labels形状: ", labels.shape)
        break

if __name__ == '__main__':
    # test_load_raw_data()
    # test_text_dataset()
    test_build_dataloader()
