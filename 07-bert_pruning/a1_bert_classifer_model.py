import torch
import torch.nn as nn
from transformers import BertModel
from config import Config

conf = Config()


class BertClassifier(nn.Module):
    def __init__(self):
        super(BertClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(conf.bert_path)
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(conf.bert_config.hidden_size, conf.num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        pooled_output = self.dropout(outputs[1])
        logits = self.fc(pooled_output)
        # 进行softmax
        return logits


# 测试以上模型
if __name__ == '__main__':
    # 加载模型
    model = BertClassifier()
    # 示例文本
    texts = ["我喜欢你", "今天天气真好"]
    from transformers import BertTokenizer

    tokenizer = BertTokenizer.from_pretrained(conf.bert_path)
    # 编码文本
    encoded_inputs = tokenizer(
        texts,
        padding="max_length",
        max_length=10,
        return_tensors="pt"
    )

    # 获取 input_ids 和 attention_mask
    input_ids = encoded_inputs["input_ids"]
    attention_mask = encoded_inputs["attention_mask"]
    print('input_ids:', input_ids)
    print('attention_mask:', attention_mask)
    print('======================================')
    logits = model(input_ids, attention_mask)
    print(logits)
