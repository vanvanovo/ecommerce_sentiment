import pandas as pd
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from config import Config

# 忽略警告信息
import warnings
warnings.filterwarnings('ignore')

# todo 1 初始化配置文件.
conf = Config()

# 定义模型预测函数
def model2predict():
    """
    实现逻辑：
    第一步：加载随机森林模型和tf-idf向量化器
    第二步：加载测试数据，获取words列分词数据和标签label
    第三步：tfidf向量化器对words数据进行向量化
    第四步：模型预测，打印测试评估指标
    第五步：预测结果保存到本地

    :return: None
    """
    # 第一步：加载随机森林模型和tf-idf向量化器
    # todo 2 加载随机森林模型(rf)
    with open(conf.rf_model_save_path, "rb") as f:
        rf = pickle.load(f)

    # todo 3 加载向量化器(tfidf)
    with open(conf.tfidf_model_save_path, "rb") as f:
        tfidf = pickle.load(f)

    # 第二步：加载测试数据，获取words列分词数据和标签label
    # todo 3 读取test数据集至(df_data), 设置分隔符为\t
    df_data = pd.read_csv(conf.process_test_datapath, sep="\t")

    # todo 4 获取words列分词数据(words)，获取标签(label)
    words = df_data['words']
    label = df_data['label']

    # 打印分词数据
    print(f'words--->: {words}')

    # 第三步：tfidf向量化器对words数据进行向量化
    # todo 5 对分词数据进行向量化(features)
    # 例如，调用transform方法实现
    features = tfidf.transform(words)

    # 第四步：模型预测，打印测试评估指标
    # todo 6 调用predict方法进行预测(y_pred)，得到预测结果，再打印准确率，精确率，召回率，F1值，评估报告等参数
    # 6.1 调用predict方法得到预测结果
    # 6.2 打印评估指标，例如：精确率, precision_score(label, y_pred, average='macro')
    y_pred = rf.predict(features)

    # 打印准确率，精确率，召回率，F1值，评估报告
    print("准确率：", accuracy_score(df_data["label"], y_pred))
    print("精确率：", precision_score(df_data["label"], y_pred, average='macro'))
    print("召回率：", recall_score(df_data["label"], y_pred, average='macro'))
    print("F1值：", f1_score(df_data["label"], y_pred, average='macro'))
    # 打印评估报告
    print('评估报告-->', classification_report(df_data["label"], y_pred))


    # 第五步：预测结果保存到本地
    # todo 7 预测结果保存到本地
    # 7.1 df_data新增一列 "pred_label"，预测结果y_pred添加到新列
    # 7.2 调用to_csv方法保存到本地，设置分隔符，忽略索引index
    df_data['pred_label'] = y_pred
    df_data.to_csv(conf.model_predict_result, sep='\t', index=False)


# 测试
if __name__ == '__main__':
    # 调用预测函数进行训练（使用预训练模型效果优秀）
    model2predict()