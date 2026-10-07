# 导包
import jieba
import pickle
from config import Config

# 忽略警告信息
import warnings
warnings.filterwarnings('ignore')

# todo 1 初始化配置文件.

# todo 2 加载随机森林模型(model)
# 注意：先打开模型文件，再使用pickle加载


# todo 3 加载向量化器(tfidf)
# 注意：先打开模型文件，再使用pickle加载


# 定义预测函数
def predict_fun(data):
    """
    实现逻辑：
    第一步：从data中获取text文本数据并分词
    第二步：tfidf向量化器对words数据进行向量化
    第三步：模型预测，得到预测结果索引
    第四步：预测结果映射回原始标签名称
    第五步：标签名称回填到data数据中并返回
    第六步：返回结果

    :param data: 就是待预测的数据, 例如:  {'text': '2011年全国各地高考各科考试时间汇总'}
    :return: data 返回结果, 例如: {'text': '2011年全国各地高考各科考试时间汇总', 'pred_class': 'education'}
    """

    #第一步：从data中获取text文本数据并分词
    # todo 4 jieba分词
    #对输入文本进行结巴分词，获取前30个词并用空格拼接，得到words
    # 例如：数据来自data['text']

    # todo 5 打印分词结果
    # print(f'words--->: {words}')

    # 第二步：tfidf向量化器对words数据进行向量化
    # todo 6 使用tfidf的transform方法进行向量化，得到features
    # 注意：输入需要是list

    # todo 7 打印向量化结果
    #print("features-->", features)

    # 第三步：模型预测，得到预测结果索引
    # todo 8 使用model的predict方法，预测features，得到预测结果y_pred_id
    # 注意返回值取0索引

    # 打印预测结果
    # print("y_pred_id-->", y_pred_id)

    # 可对比分析预测结果返回值的格式
    # y_pred_id = model.predict(features)
    # print("y_pred_id-->", y_pred_id)

    # 第四步：预测结果映射回原始标签名称
    # todo 9 设置标签索引和名称对应关系(id2class)
    # 例如：打开class_doc_path类别文件；构建索引和类别的字典对应关系，{0:'finance', 1: 'realty', ...}

    # todo 10 打印标签和名称对应关系
    # print("id2class-->", id2class)
    # todo 11 从id2class中索引到真实的预测类别名称(y_pred)

    # 第五步：标签名称回填到data数据中并返回
    # todo 12 给原始字典数据data，新增一个关键字pred_class，对应预测类别名称

    # 第六步：返回结果
    # 返回结果
    return data


if __name__ == '__main__':
    # 构建一个预测数据
    data = {"text": "体验2D巅峰 倚天屠龙记十大创新概览"}
    # 调用预测函数，进行预测
    result = predict_fun(data)
    # 打印预测结果
    print(result)
