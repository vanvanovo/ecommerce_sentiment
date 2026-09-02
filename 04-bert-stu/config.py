import torch
from transformers import BertTokenizer, BertConfig

# 针对 京东商品评论情感分析(ecommerce_sentiment), 二分类(0=差评=negative, 1=好评=positive).
# 注意: a2/a1/a3 用到 conf.bert_path / bert_config / tokenizer / num_classes / device / pad_size 等属性.
class Config(object):
    def __init__(self):
        self.root_path = r'D:\work\ecommerce_sentiment/'

        # 原始数据路径(text\tlabel, 无表头)
        self.train_datapath = self.root_path + '01-data-stu/jd_data/prepared_clean/train.txt'
        self.test_datapath = self.root_path + '01-data-stu/jd_data/prepared_clean/dev.txt'
        self.dev_datapath = self.root_path + '01-data-stu/jd_data/prepared_clean/dev.txt'
        # 类别文档(二分类: negative 0 / positive 1)
        self.class_path = self.root_path + "01-data-stu/class.txt"
        self.class_list = [line.strip() for line in open(self.class_path, encoding="utf-8")]  # ['negative','positive']

        # 模型训练/保存路径
        self.model_save_path = self.root_path + "04-bert-stu/save_models/bert_classifer_model.pt"

        # 设备: GPU 可用则用 cuda, 否则用 cpu
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # BERT 参数
        self.num_classes = len(self.class_list)  # =2 (二分类)
        self.num_epochs = 2                      # epoch 轮数
        self.batch_size = 64                     # mini-batch 大小
        self.pad_size = 32                       # 每句话处理成长度(短填长切)
        self.learning_rate = 5e-5                # 学习率

        # 预训练 BERT 路径(新项目内, 已复制 model.safetensors)
        self.bert_path = self.root_path + "04-bert-stu/bert-base-chinese"
        # 注意: 只在这里加载 tokenizer 和 config; bert 模型由 a2_bert_classifer_model 里 from_pretrained 加载(避免重复占内存)
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        self.bert_config = BertConfig.from_pretrained(self.bert_path)


if __name__ == '__main__':
    conf = Config()
    print('device-->', conf.device)
    print('num_classes-->', conf.num_classes, 'class_list-->', conf.class_list)
    print('bert_path-->', conf.bert_path)
    input_size = conf.tokenizer.convert_tokens_to_ids(["你", "好", "中", "人"])
    print('tokenizer 测试 id-->', input_size)