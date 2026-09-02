"""
概述：rf模型所需数据
"""

#导包
import jieba              # 分词
import pandas as pd       # 数据处理，读取，保存
from config import Config # 配置文件

# todo 1 初始化配置文件.
conf = Config()

# 定义数据处理函数
def process_data(datapath, processed_datapath):
    """
    数据预处理函数，用于获取数据分词结果并添加到原数据列尾部
    :param datapath: 数据集路径
    :param processed_datapath: 数据保存路径
    :return: None
    """
    # todo 2 读取数据到df_data,指定列名为text和label
    df_data = pd.read_csv(datapath, sep='\t', header=None, names=['text', 'label'])
    # todo 3 打印head数据，初步了解数据样式
    print(f"数据样式---> {df_data.head()}")
    # todo 4 进行分词预处理
    # 使用jieba对df_data['text']列进行分词，并将分词结果添加到列尾部 df_data['words']
    # 实现逻辑是使用apply函数对df_data['text']中的每行文本应用lambda匿名函数
    # 例如： 匿名函数实现对输入内容先进行分词并限制长度，再拼接
    # 例如：" ".join(jieba.lcut(x)[:30]
    # 【扩展】jieba.lcut、jieba.cut的区别
    df_data['words'] = df_data['text'].apply(lambda x: " ".join(jieba.lcut(x)[:30]))
    #todo 5 打印前10个样本数据，查看分词后的结果
    print(f"打印前10个数据--->{df_data.head(10)}")
    # todo 6 保存数据
    # 调用to_csv函数实现保存
    # 注意：设置分隔符为'\t'；不要添加索引，但是要保留表头
    df_data.to_csv(processed_datapath, sep='\t', index=False, header=True)

# 测试
if __name__ == '__main__':
    # todo 7 保存处理后训练数据
    process_data(conf.train_datapath, conf.process_train_datapath)

    # todo 8 保存处理后测试数据
    process_data(conf.test_datapath, conf.process_test_datapath)

    # todo 9 保存处理后验证数据
    process_data(conf.dev_datapath, conf.process_dev_datapath)

    # 打印结束标志
    print("结束")
