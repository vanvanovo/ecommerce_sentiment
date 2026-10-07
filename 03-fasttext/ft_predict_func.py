# 导包操作
import fasttext
import jieba
from config import Config

# todo 1 初始化配置文件.
conf = Config()

# todo 2 加载模型(model)
# 注意：调用fasttext的模型加载方法load_model实现，具体模型可使用给定的预训练模型；
model = fasttext.load_model(conf.ft_model_save_path + '/model_word_2_auto_20260706.bin')

# 定义预测函数
def predict_func(data_dict):
    """
    实现逻辑：
    第一步：从data_dict中获取text文本数据并分词
    第二步：模型预测，得到预测结果标签
    第三步：返回结果
    :param data_dict:
    :return:
    """
    # 第一步：从data_dict中获取text文本数据并分词
    # todo 3 获取文本(text)
    text = data_dict['text']

    # todo 4 使用jieba分词，再使用空格拼接，得到text
    text = " ".join(jieba.lcut(text))

    # 第二步：模型预测，得到预测结果标签
    # todo 5 使用model进行预测(re)
    # 注意：可以调用predict方法实现
    re = model.predict(text)

    # 打印预测结果格式
    # 注意：预测结果为标签和概率做成的元组，例如：(('__label__game',), array([0.92483222]))
    print('re-->', re)
    print('type(re)-->', type(re))
    # print("re-->", re[0][0].replace("__label__", ""))
    # todo 6 将预测结果添加到data_dict中，pred_class
    # 注意：根据预测结果的格式进行解析，例如：(('__label__game',), array([0.92483222]))
    #      解析结果再进行真实标签提取
    # data_dict['pred_class'] = re[0][0][9:] 或者 .replace('__label__', '')
    data_dict['pred_class'] = re[0][0][9:]

    # 第三步：返回结果
    return data_dict


if __name__ == '__main__':
    data = {"text": "体验2D巅峰 倚天屠龙记十大创新概览"}
    result = predict_func(data)
    print('result-->', result)
