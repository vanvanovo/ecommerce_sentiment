"""
概述：
该.py文件用于搭建 BERT分类模型的.
"""
import torch                        # 深度学习框架
import torch.nn as nn               # 神经网络模块
from transformers import BertModel, BertTokenizer  # Bert模型, 分词器
from config import Config           # 配置文件类

# todo 1 初始化配置文件.
conf = Config()

# 定义bert模型
class BertClassifier(nn.Module):
    def __init__(self):
        # 初始化父类类的构造函数
        super().__init__()
        # todo 2 加载BERT模型(self.bert)
        # 使用BertModel是从transformers库中加载的预训练模型
        # config.bert_path是预训练模型的路径
        self.bert = BertModel.from_pretrained(conf.bert_path)

        # todo 3 定义全连接分类层(self.fc)
        # 利用nn.Linear定义全连接层（fc），用于分类任务
        # 输入尺寸是Bert模型隐藏层的大小，即768（对于Base模型） conf.bert_config.hidden_size
        # 输出尺寸是类别数量10，由config.num_classes指定
        self.fc = nn.Linear(conf.bert_config.hidden_size, conf.num_classes)

    # 定义前向传播方法.
    def forward(self, input_ids, attention_mask):
        """"""
        # todo 4 获取bert模型的输出(outputs)
        # 将input_ids 和 attention_mask 输入BERT模型, 获取模型输出(outputs 其中包含: last_hidden_state, pooler_output)
        # input_ids: 输入的Token ID张量, 形状为: [batch_size, 序列长度max_length]
        # attention_mask: 输入的注意力掩码张量, 形状为: [batch_size, 序列长度max_length]
        # outputs拆包是: _,pooled池化
        # 注意：这个预训练模型，不进行参数更新，可使用with torch.no_grad():实现
        with torch.no_grad():
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
                   # 输入的token ID
                   # 注意力掩码用于区分有效token和填充token
        # 打印输出结果
        # print('outputs-->',outputs)  # 观察结果
        # print('last_hidden_state.shape-->',outputs.last_hidden_state.shape) # [2, 9, 768]
        # print('pooler_output.shape-->',outputs.pooler_output.shape) # [2, 768]
        # print('CLS.shape-->',outputs.last_hidden_state[:, 0].shape) # [2, 768]

        # todo 5 通过全连接层对BERT模型的输出进行分类，得到(logits)
        # 取BERT预训练模型outputs的pooler_output，输入fc层
        logits = self.fc(outputs.pooler_output)
        # 返回分类的logits（未归一化的预测分数）
        return logits


# 测试以上模型 (运行时取消注释)
def test_bert_classifier():
    """"""
#     # 加载BERT分词器, 将文本 -> 模型可识别的 Token ID
    tokenizer = BertTokenizer.from_pretrained(conf.bert_path)
#
#     # 准备示例文本, 用于测试 模型的输入数据.
    texts = ["王者荣耀", "今天天气真好"]
#
#     # 编码文本 -> 将原始文本转成模型所需要的 的输入数据(Token ID, Attention Mask)
    encoded_inputs = tokenizer(texts,
                               padding="max_length",  # 所有的填充到指定的max_length长度
                               max_length=9,          # 最大长度, 目标序列长度, 超过就截断, 不足就填充
                               truncation=True,       # 如果超出指定的max_length长度，则截断
                               return_tensors="pt"    #返回(PyTorch)张量
                               )
#
#     # 提取模型输入张量: 从编码结果中拿出 Token ID 和 Attention Mask张量.
    input_ids = encoded_inputs["input_ids"]
    attention_mask = encoded_inputs["attention_mask"]
    print('input_ids-->', input_ids)
    print('attention_mask-->', attention_mask)
    print('-' * 40)
#
#     # 创建自定义的bert模型
    model = BertClassifier()  # __init__()执行了
#
#     # 预测 模型前向传播, 获取模型输出.
    logits = model(input_ids=input_ids, attention_mask=attention_mask)  # forward()执行了
    print('logits-->', logits)  # 每一行对应一个样本，每个数字表示该样本属于某一类别的“得分”（logit），没有经过 softmax 归一化。
    print('-' * 40)
#
#     # 计算类别概率, 对logits做softmax()归一化, 得到每个类别在[0, 1]区间的概率
    probs = torch.softmax(logits, dim=-1)
    print('probs-->', probs)  # 归一化后该样本属于某类的概率（范围在 0~1 之间）,概率最高的就是预测结果
    print('-' * 40)
#
#     # 获取预测结果 : 即概率最大的类别索引.
    preds = torch.argmax(logits, dim=-1)
    print('preds-->', preds)  # 得到每个样本的预测类别。表示两个输入文本被模型预测为类别 6（从 0 开始计数）。


if __name__ == '__main__':
    test_bert_classifier()