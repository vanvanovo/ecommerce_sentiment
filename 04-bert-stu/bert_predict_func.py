# 该 .py文件的作用是 -> 预测分类的 函数版, 对接后续的 API 和 APP版.

# 导包.
import torch
from a2_bert_classifer_model import BertClassifier
from config import Config

# 压制警告.
import warnings
warnings.filterwarnings("ignore")

# todo 1. 加载配置对象，包含模型参数、路径等
conf = Config()

# todo 2. 实例化BertClassifier模型(model)
model = BertClassifier()

# todo 3 加载模型参数
# 注意：先使用torch.load函数从conf.model_save_path中加载模型，再填充到model.load_state_dict函数中
model.load_state_dict(torch.load(conf.model_save_path,  map_location=torch.device('cpu')))

# todo 4 添加模型到指定设备 conf.device
model.to(conf.device)
# todo 5 设置模型为评估eval()模式
model.eval()


# 定义predict_fun函数预测函数, 接收文本数据, 返回分类结果.
def predict_func(data_dict):
    """
    根据用户录入数据，接收包含文本的字典, 通过BERT模型预测文本类别, 返回带预测结果的字典.
    :param 参数 data_dict: {"text":"状元心经：考前一周重点是回顾和整理"}
    :return: 返回 data_dict: {"text":"状元心经：考前一周重点是回顾和整理", "pred_class":"education"}
    """
    # todo 6 提取输入文本, 获取待预测的字符串(text)
    text = data_dict['text']
    # todo 7 利用conf.tokenizer.batch_encode_plus, 将原始文本 -> BERT模型可识别的token(text_tokens)
    # 注意：参数1 text文本转成列表；参数2 padding设置成"max_length"填充策略
    # 参数3 max_length设置成conf.pad_size
    text_tokens = conf.tokenizer.batch_encode_plus([text],
                                                   padding="max_length",
                                                   max_length=conf.pad_size)


    # 打印文本tokens
    # print("text_tokens-->", text_tokens)

    # todo 8 从text_tokens中提取模型所需要的特征(input_ids和attention_mask)
    input_ids = text_tokens['input_ids']
    attention_mask = text_tokens['attention_mask']
    # todo 9 将input_ids和attention_mask转为tensor, 并指定到设备
    # 注意：调用torch.tensor函数
    # 将input_ids和attention_mask转为tensor, 并指定到设备
    input_ids = torch.tensor(input_ids).to(conf.device)
    attention_mask = torch.tensor(attention_mask).to(conf.device)
    # todo 10 设置不进行梯度计算(在该上下文中禁用梯度计算，提升推理速度并减少内存占用)
    # 例如：调用torch.no_grad()函数实现
    with torch.no_grad():
        # todo 11 前向传播, 模型预测(logits)
        # 注意：参数包括input_ids, attention_mask
        # 前向传播(模型预测)
        logits = model(input_ids, attention_mask)
        # print("logits-->", logits, logits.shape)

        # 打印logits和shape
        # print("logits-->", logits, logits.shape)
        # todo 12 获取预测类别索引
        # 注意：调用torch.argmax实现，参数包括logits和dim=-1
        # 获取预测类别索引张量
        preds = torch.argmax(logits, dim=-1)
        # print("preds-->", preds)

        # 打印预测结果
        # print("preds-->", preds)
        # todo 13 从preds中获取预测类别索引标量(pred_idx)
        # 注意：转换索引格式, 从PyTorch张量  -> Python的标量
        # 使用.item()实现
        # 获取预测类别索引标量
        pred_idx = preds.item()

        # todo 14 获取预测类别. 根据索引 -> 类别名 (pred_class)
        # 注意：从class_list中获取真实类别名称
        # 获取类别名称
        pred_class = conf.class_list[pred_idx]

        # 打印pred_class
        # print('pred_class-->', pred_class)
        # todo 15 将预测结果pred_class添加到data_dict中
        # print('pred_class-->', pred_class)
        # 将预测结果添加到data_dict中
        data_dict['pred_class'] = pred_class

    # 返回data_dict
    return data_dict


if __name__ == '__main__':
    data_dict = {'text': '状元心经：考前一周重点是回顾和整理'}
    result = predict_func(data_dict)
    print("result-->", result)
