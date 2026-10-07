# 导包
import pandas as pd  # 数据读取和处理
import pickle  # 用于模型和向量化器的序列化保存 官网：https://docs.python.org/3/library/pickle.html
from sklearn.feature_extraction.text import TfidfVectorizer  # 将文本转成数值特征(可以理解为: 词向量)
from sklearn.model_selection import train_test_split  # 训练集和测试集的划分
from sklearn.ensemble import RandomForestClassifier  # 随机森林分类器
# 模型评估指标: 准确率, 精准率, 召回率, F1-score, 分类报告, 混淆矩阵
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from config import Config  # 配置文件

# 忽略警告信息
import warnings
warnings.filterwarnings('ignore')

# todo 1 初始化配置文件.
conf = Config()


# 定义模型训练函数
def model2train():
    """
    实现逻辑：
    第一步：读取数据，获取分词数据和标签
    第二步：初始化tf-idf向量化器，对分词数据进行向量化
    第三步：对向量化后的数据，划分训练集和测试集
    第四步：初始化随机森林模型，进行训练和评估，得到评估结果
    第五步：保存随机森林模型和tf-idf向量化器到本地
    :return: None
    """
    # 第一步：读取数据，获取分词数据和标签
    #  todo 2 利用read_csv方法读取 预处理的训练数据集数据 到df_data
    #   注意设置分隔符，可设置使用部分数据提高处理速度
    df_data = pd.read_csv(conf.process_train_datapath, sep='\t')
    # todo 3 打印前5行数据，了解数据集格式
    print(f"训练数据集查看---> {df_data.head()}")
    # todo 4 提取分词特征列
    #  例如：提取分词特征列'words'到 words
    words = df_data['words']
    # todo 5 提取标签列
    #  例如：提取标签列'label'到 label
    label = df_data['label']
     # todo 6 打印分词特征和标签，可设置使用部分数据提高速度
    print(f"分词结果---> {words[:10]}")
    print(f"标签结果---> {label[:10]}")
    # 第二步：初始化tf-idf向量化器，对分词数据进行向量化
    # todo 6 从停用词文件中读取停用词，组装成list列表 stop_words
    # 停用词: 对分类无意义的词, 如: 的, 是等这些词
    # 6.1 读取停用词文件, 按行分割为列表.
    # 6.2 去掉每个词的首尾空格
    # 6.3 所有停用词封装到列表中，得到stop_words
    stop_words =[line.strip() for line in open(conf.stop_words_path, 'r', encoding='utf-8').readlines()]
    # todo 7 打印停用词
    print(f"停用词---> {stop_words[:10]}")
    # todo 8 初始化 TF-IDF向量化器, 指定停用词列表(过滤停用词的)
    tfidf = TfidfVectorizer(stop_words=stop_words)
    # todo 9 将 words 文本转为词频矩阵 features
    features = tfidf.fit_transform(words)
    # todo 10 打印特征shape
    print(f"特征的shape---> {features.shape}")
    # 第三步：对向量化后的数据，划分训练集和测试集
    # todo 11 划分训练集和测试集
    # 这里的数据已经是训练集，再次划分是为了完整性
    # 注意：
    # 11.1 入参是features和labels，可根据需要设置数据集的划分比例
    # 11.2 返回值是四个（x_train, x_text, y_train, y_text），极易写错
    x_train, x_test, y_train, y_test = train_test_split(features, label, test_size=0.2, random_state=42)
    # 第四步：初始化随机森林模型，进行训练和评估，得到评估结果
    # todo 12 初始化随机森林模型，得到实例化模型rf
    rf = RandomForestClassifier()
    # todo 13 调用fit方法，开始训练模型
    print("开始训练模型...")
    rf.fit(x_train, y_train)
    # todo 14 调用predict方法，进行模型预测，得到预测结果y_pred
    print("开始模型预测和评估...")
    y_pred = rf.predict(x_test)
    # todo 15 进行模型评估，打印准确率，精确率，召回率，F1值等参数
    print("模型评估结果如下:")
    # 15.1 打印评估指标，例如：精确率, precision_score(y_test, y_pred, average='macro')
    print("模型评估结果如下:")
    print("准确率：", accuracy_score(y_test, y_pred))
    print("精确率：", precision_score(y_test, y_pred, average='macro'))
    print("召回率：", recall_score(y_test, y_pred, average='macro'))
    print("F1值：", f1_score(y_test, y_pred, average='macro'))
    # 第五步：保存随机森林模型和tf-idf向量化器到本地
    print("开始保存模型和向量化器...")
    # todo 16 使用pickle.dump，保存训练好的 随机森林模型
    # 保存逻辑：
    # 16.1 打开模型保存路径文件
    # 16.2 讲模型保存到打开的文件中
    with open(conf.rf_model_save_path, "wb") as f:
        pickle.dump(rf, f)
    # todo 17 使用pickle.dump，保存训练好的 TF-IDF向量化器(后续预测时, 需要使用同一个向量化器转换新文本)
    # 保存逻辑与保存随机森林模型一致
    with open(conf.tfidf_model_save_path, "wb") as f:
        pickle.dump(tfidf, f)


if __name__ == '__main__':
    model2train()
