import torch
from a2_bilstm_classifer_model import BiLSTMModel
from config import Config

# 加载配置
conf = Config()

# 准备模型
model = BiLSTMModel()
# 加载模型参数
model.load_state_dict(torch.load(conf.bilstm_save_model_path, weights_only=True))
# 添加模型到指定设备
model.to(conf.device)
# 设置模型为评估模式
model.eval()


# 定义predict_fun函数预测函数
def predict_fun(data_dict):
    """
    根据用户录入数据,返回分类信息
    :param 参数 data_dict: {"text":"状元心经：考前一周重点是回顾和整理"}
    :return: 返回 data_dict: {"text":"状元心经：考前一周重点是回顾和整理", "pred_class":"education"}
    """
    # 获取文本
    text = data_dict['text']
    # 将文本转为id
    text_tokens = conf.tokenizer.batch_encode_plus([text],
                                                   padding="max_length",
                                                   max_length=conf.pad_size)
    # 获取input_ids和attention_mask
    input_ids = text_tokens['input_ids']
    attention_mask = text_tokens['attention_mask']
    # 将input_ids和attention_mask转为tensor, 并指定到设备
    input_ids = torch.tensor(input_ids).to(conf.device)
    attention_mask = torch.tensor(attention_mask).to(conf.device)

    # 设置不进行梯度计算(在该上下文中禁用梯度计算，提升推理速度并减少内存占用)
    with torch.no_grad():
        # 前向传播(模型预测)
        output = model(input_ids, attention_mask)
        # 获取预测类别索引张量
        output = torch.argmax(output, dim=1)
        # 获取预测类别索引张量
        pred_idx = output.item()
        # 获取类别名称
        pred_class = conf.class_list[pred_idx]
        print('pred_class-->', pred_class)
        # 将预测结果添加到data_dict中
        data_dict['pred_class'] = pred_class
    # 返回data_dict
    return data_dict


if __name__ == '__main__':
    data_dict = {'text': '状元心经：考前一周重点是回顾和整理'}
    print(predict_fun(data_dict))
