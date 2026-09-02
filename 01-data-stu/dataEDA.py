"""
概述：
EDA:
    探索性数据分析(Exploratory Data Analysis), 是一种分析数据集以总结其主要特征的方法，例如数据规模，样本数量分布和占比.
    大白话: 就是分析数据的, 可以帮我们发现数据模式, 检测异常, 测试假设, 从而对数据集有一个直观的了解.
"""
# 导包
import pandas as pd                 # 用于处理结构化表格数据的
from collections import Counter     # 用于高效统计标签出现的次数
from config import Config           # 获取项目的配置信息
from matplotlib import pyplot as plt

# todo 1 创建Config类的实例, 加载项目配置.
conf = Config()

# 定于数据处理函数
def data_eda(path):
    """
    读取数据并对数据做分析
    :param path: 数据集路径
    :return: None
    """
    # 第一步：使用pd.read_csv读取数据到df_data；并查看基本信息，设置分隔符sep；header;  names=['text', 'label']参数
    #  todo 2 读取数据到df_data
    df_data = pd.read_csv(path, sep="\t", header=None, names=["text", "label"])

    #  todo 3 打印返回值类型，了解返回值
    print(f"返回值的数据类型---> {type(df_data)}")

    #  todo 4 打印前10行数据，简单了解数据
    print(f"前10行数据---> {df_data.head(10)}")

    # todo 5 打印数据的总行数, 以便了解数据的规模.
    print(f"数据总量---> {len(df_data)}")

    # todo 6 打印标签类别数量分布，了解每个标签类别数据量
    # 例如：df_data['label']是一列的数据[2, 3, 4, 2, 5, 1, 6, ...]
    # 调用value_counts()方法，统计每个标签的数量
    print(f"标签类别数量 ---> {df_data['label'].value_counts()}")

    # 第二步：还可以利用 Counter统计df_data['label']的标签分布
    #  todo 7 例如:Counter([1,2,2,1,3,2,2])->Counter({2: 4, 1: 2, 3: 1})
    counter = Counter(df_data["label"])

    # todo 8 打印返回值counter，了解标签分布
    print(f"数据分布---> {counter}")
    # todo 9 遍历返回值每个元素，打印每个标签及其出现的次数，例如 f"类别 {key} ---> 数量 {value}"

    for key, value in counter.items():
        print(f"类别--->{key}; 数量---> {value}")

    # 第三步：计算每个标签的占比，当前标签样本数量/样本总数
    # todo 10 遍历返回值每一个元素的占比，例如 f"总量:{all_count}，标签{key}出现次数:{value}，标签{key}占比：{value / all_count:.2%}"
    total_num = len(df_data)
    for key, value in counter.items():
        print(f"总数---> {total_num}; 标签---> {key};数量--->{value}; 标签/总数--->{value / total_num:.2%}")

    # 第四步：分析文本长度
    # 在数据中新增'text_length'列, 存储每条文本的字符数.
    # 4.1 对df_data中的df_data['text']列，应用apply和匿名函数;
    # 4.2 apply()依次对每个元素执行函数操作，函数使用匿名函数实现，获取每行文本长度
    # 4.3 匿名函数 lambda x:len(x)
    # 4.4 在原df_data中，新增'text_length'列，df_data['text_length']
    # 4.5 是否还有其他实现方法，例如：data['text'].str.len()
    # todo 11 统计每行文本的长度并新增一列
    df_data["text_length"] = df_data["text"].apply(lambda x: len(x))

    # todo 12 打印添加新列后的前10行数据
    print(f"前10行数据---> {df_data.head(10)}")

    # todo 13 打印df_data['text_length']的分布情况，可以调用 .describe()函数实现
    print(f"打印分布情况---> {df_data['text_length'].describe()}")

    print("\n文本长度统计：")
    # todo 14 还可以使用单独方法，打印更具体的统计信息，例如：
    #  平均值mean(),打印文本长度的平均值(保留2位小数)
    #  标准差std(),打印文本长度的标准差(反应长度的离散程度, 保留2位小数)
    #  最大长度max(), 打印文本长度的最大值，用于获取数据长度上限
    #  最小长度min(), 打印文本长度的最小值，用于获取数据长度下限
    #  例如: f"平均长度：{df_data['text_length'].mean():.2f} 字符"
    print(f"平均长度：{df_data['text_length'].mean():.2f} 字符")  # 平均值
    print(f"长度标准差：{df_data['text_length'].std():.2f} 字符")  # 标准差
    print(f"最大长度：{df_data['text_length'].max()} 字符")  # 最大值
    print(f"最小长度：{df_data['text_length'].min()} 字符")  # 最小值

    # todo 15 样本数据集不均衡如何处理？
    # 少 ---> 多
    # 对于机器学习算法，可以采样，SMOTE，随机合成，上下文语义不关联
    #  今天中午是麻辣烫！----> 麻辣今天中是午吃！
    # 大模型法，回译数据增强法，同义词替换
    # 我爱你---> i love you ---> 日文
    # 我喜欢你
    # 多 --- > 少
    # 随机抽样
    # df_data.sample()
    print(f"抽样结果---> {df_data.sample(5, random_state=2026)}")

    # 【扩展】绘制文本长度直方图
    # 提示：使用hist()和 matplotlib中的pyplot
    # df_data["text_length"].hist()
    # plt.show()



    # 华丽的分割线
    print('-'*40)



# 测试
if __name__ == '__main__':
    # todo 16 训练数据集探索性分析
    data_eda(conf.test_path)

    # todo 17 测试数据集探索性分析

    # todo 18 验证数据集探索性分析

    # 打印结束标志
    print("结束")

