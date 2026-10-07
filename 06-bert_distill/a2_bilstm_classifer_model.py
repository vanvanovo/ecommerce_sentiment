import torch
import torch.nn as nn
from config import Config
from transformers import BertTokenizer

# todo 1 初始化配置文件.

class BiLSTMModel(nn.Module):
    def __init__(self):
        # 调用父类初始化方法
        super().__init__()
        # todo 2 定义embedding层 （自定义的学生模型，理解原理即可）(self.embedding)
        # 实现思路：利用nn.Embedding实现对输入token的编码
        # 例如：self.embedding = nn.Embedding(conf.bert_config.vocab_size, conf.embed_size)

        # todo 3 定义lstm(self.lstm)
        # 实现思路：调用nn.LSTM实现
        # 参数1， 词嵌入的维度conf.embed_size；参数2，隐藏层的大小conf.hidden_size_lstm；
        # 参数3，层数conf.num_layers；参数4，适用双向lstm，获取更丰富语义 bidirectional=True；参数5，设置batch_first=True，方便理解
        # 例如：self.lstm = nn.LSTM(conf.embed_size, conf.hidden_size_lstm, conf.num_layers, bidirectional=True, batch_first=True)

        # todo 4 定义dropout，设置随机失活概率conf.dropout
        # 例如：self.dropout = nn.Dropout(conf.dropout)

        # todo 5 定义全连接层
        # 例如：self.fc = nn.Linear(conf.hidden_size_lstm * 2, conf.num_classes)


    def forward(self, input_ids, attention_mask):
        """
        :param input_ids: 维度：[batch_size, seq_len]
        :param attention_mask: 维度：[batch_size, seq_len]
        :return:
        """
        # Embedding 层，获取ebedding词向量
        # [batch_size, seq_len, embed_size]
        # todo 6 调用词嵌入层(embed)
        # 例如：embed = self.embedding(input_ids)

        # print("embed-->", embed, embed.shape)
        # todo 7 获取有效字符（理解原理即可）
        # 实现逻辑：经过valid mask输出对应有效字符的embedding
        # 注意：valid_mask是一个矩阵，内部仅包含True(1)和False(0)
        #      True用于获取同位置的有效字符；False由于忽略同位置的无效字符，例如101， 102，填充0等
        # 创建掩码，过滤 [CLS] (101) 和 [SEP] (102)
        cls_token_id = 101  # 动态获取 token ID
        sep_token_id = 102
        print("input_ids-->", input_ids, input_ids.shape)
        # print("attention_mask-->", attention_mask, attention_mask.shape)
        # print("input_ids != cls_token_id-->", input_ids != cls_token_id)
        # print("input_ids != sep_token_id-->", input_ids != sep_token_id)
        # todo 8 把101和102都置为False: tensor([False,  True,  True... ,False])
        # 例如：cls_sep_mask = (input_ids != cls_token_id) & (input_ids != sep_token_id)

        # 打印cls_sep_mask
        # print("cls_sep_mask-->", cls_sep_mask)

        # [batch_size, seq_len]
        # todo 9 将attention_mask中的填充部分设置为False
        # 例如：valid_mask = attention_mask & cls_sep_mask

        # print("valid_mask-->", valid_mask, valid_mask.shape)

        # todo 10 添加一个维度，用于匹配embed的shape
        # 调用valid_mask.unsqueeze，在(-1)添加一个维度
        # [batch_size, seq_len, 1]
        # 例如：valid_mask_embed = valid_mask.unsqueeze(-1)

        # print("valid_mask_embed-->", valid_mask_embed)

        # todo 11 无效 token(101，102，填充位置) 的嵌入置为 0
        # 使用相乘实现，任何值乘以0都是0
        # 例如：embed = embed * valid_mask_embed

        # print("embed-->", embed, embed.shape)

        # todo 12 经过BiLSTM 层 前向传播
        # 使用相同的套路，将无效token(101, 102, 填充位置)的值置为0
        # [batch_size, seq_len, hidden_size*2]
        # 例如：lstm_out, (hn, cn) = self.lstm(embed)

        # print("lstm_out-->", lstm_out, lstm_out.shape)
        # BERT的输出结果：pooler_output：当做一句话的整体token。除此之外，还有一个CLS（也可以当做这一句话的整体token）
        # todo 13 添加一个维度，用于匹配lstm_out的shape
        # 实现思路同todo 10
        #例如：valid_mask_out = valid_mask.unsqueeze(-1)

        # print("valid_mask_out-->", valid_mask_out)
        # [batch_size, seq_len, hidden_size*2]

        # todo 14 无效 token(101，102，填充位置) 的嵌入置为 0
        # 实现思路同todo 11
        # 使用相乘实现，任何值乘以0都是0
        # 例如：valid_lstm_out = lstm_out * valid_mask_out

        # print("valid_lstm_out-->", valid_lstm_out, valid_lstm_out.shape)

        # todo 15 对整个句子有效字符（token数）进行平均。
        #  实现逻辑，先整体求和，再除以token的数量
        # 求和操作
        # [batch_size, hidden_size*2]
        # 例如：sum_hidden = valid_lstm_out.sum(dim=1)

        # print("sum_hidden-->", sum_hidden, sum_hidden.shape)
        # todo 16 计算有效 token 的数量，数量等于valid_mask_out求和，因为valid_mask_out中的值都是1
        #  避免除以 0（加上一个极小值）
        # [batch_size, 1]
        # (因为valid_mask_out内部除了True就是False，求和的话，False 求和全部等于0；True等于1，求和，就等同于数量)
        # 例如：valid_token_count = valid_mask_out.sum(dim=1) + 1e-8

        # print("valid_token_count-->", valid_token_count, valid_token_count.shape)
        # todo 17 平均池化
        # 实现逻辑：总和除以数量得到均值
        # [batch_size, hidden_size*2]
        # 例如：hidden = sum_hidden / valid_token_count

        # print("hidden-->", hidden, hidden.shape)

        # todo 18 调用Dropout层
        # 例如：hidden = self.dropout(hidden)

        # [batch_size, num_classes]
        # todo 19 调用全连接层
        # 例如：logits = self.fc(hidden)

        # 返回logits
        # return logits


# 进行模型测试
def test_bilstm_model():
    # 编写测试数据
    model = BiLSTMModel()
    # 示例文本
    texts = ["我中意你", "你爱他"]

    tokenizer = BertTokenizer.from_pretrained(conf.bert_path)
    # 编码文本
    encoded_inputs = tokenizer(texts,
                               padding="max_length",
                               max_length=8,
                               return_tensors="pt")

    # 获取 input_ids 和 attention_mask
    input_ids = encoded_inputs["input_ids"]
    attention_mask = encoded_inputs["attention_mask"]

    # 预测
    logits = model(input_ids, attention_mask)
    print('logits-->', logits)  # 每一行对应一个样本，每个数字表示该样本属于某一类别的“得分”（logit），没有经过 softmax 归一化。

    # 获取预测概率
    probs = torch.softmax(logits, dim=-1)
    print('probs-->', probs)  # 归一化后该样本属于某类的概率（范围在 0~1 之间）,概率最高的就是预测结果

    # 获取最大元素对应的索引(类别)
    preds = torch.argmax(logits, dim=-1)
    print('preds-->', preds)  # 得到每个样本的预测类别。表示两个输入文本被模型预测为类别 6（从 0 开始计数）。


if __name__ == '__main__':
    test_bilstm_model()