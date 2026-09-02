import torch
import torch.nn as nn
from transformers import BertModel
from config import Config

# todo 1 初始化配置文件.
conf = Config()
# 定义bert模型
class BertClassifier(nn.Module):
    def __init__(self):
        super(BertClassifier, self).__init__()
        # todo 2 加载BERT模型(self.bert)
        # 使用BertModel是从transformers库中加载的预训练模型
        # config.bert_path是预训练模型的路径

        # todo 3 定义全连接分类层(self.fc)
        # 利用nn.Linear定义全连接层（fc），用于分类任务
        # 输入尺寸是Bert模型隐藏层的大小，即768（对于Base模型） conf.bert_config.hidden_size
        # 输出尺寸是类别数量10，由config.num_classes指定

    def forward(self, input_ids, attention_mask):
        """"""
        # todo 4 获取bert模型的输出(outputs)
        # 将input_ids 和 attention_mask 输入BERT模型, 获取模型输出(outputs 其中包含: last_hidden_state, pooler_output)
        # input_ids: 输入的Token ID张量, 形状为: [batch_size, 序列长度max_length]
        # attention_mask: 输入的注意力掩码张量, 形状为: [batch_size, 序列长度max_length]
        # outputs拆包是: _,pooled池化
        # 注意：这个预训练模型，不进行参数更新，可使用with torch.no_grad():实现

        # print('outputs-->',outputs)  # 观察结果

        # todo 5 通过全连接层对BERT模型的输出进行分类，得到(logits)
        # 取BERT预训练模型outputs的outputs[1]，输入fc层
        # 注意：outputs[1]等同于 outputs.pooler_output

        # 返回logits
        # return logits


# 测试以上模型
def test_bert_model():
    # 加载模型
    model = BertClassifier()
    # 示例文本
    texts = ["我喜欢你", "今天天气真好"]
    from transformers import BertTokenizer

    tokenizer = BertTokenizer.from_pretrained(conf.bert_path)
    # 编码文本
    encoded_inputs = tokenizer(texts,
                               padding="max_length",
                               max_length=10,
                               return_tensors="pt")

    # 获取 input_ids 和 attention_mask
    input_ids = encoded_inputs["input_ids"]
    attention_mask = encoded_inputs["attention_mask"]
    print('input_ids-->', input_ids)
    print('attention_mask-->', attention_mask)
    logits = model(input_ids, attention_mask)
    print('logits-->', logits)


if __name__ == '__main__':
    test_bert_model()
