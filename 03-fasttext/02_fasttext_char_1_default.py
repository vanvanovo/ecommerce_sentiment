"""
概述：
    该py文件用于: 字符级别 默认参数训练
"""

# 导包
import fasttext
from config import Config
import datetime
import os

# 获取时间
current_time = datetime.datetime.now().date().today().strftime("%Y%m%d")
print("current_time-->", current_time)


# todo 1 初始化配置文件.
conf = Config()

# todo 2. 模型训练: 使用 fasttext训练 字符级 文本分类模型(model)
# 例如：# fasttext.train_supervised(): fasttext的核心训练函数, 用于训练: 有监督学习分类模型
# 注意：重点配置input: 输入的处理后的训练数据字符级文件
#       dim: 词向量维度, 维度越小, 计算越快，默认100
#       minn: 子词最小长度  默认0
#       maxn: 子词最大长度  默认0
model = fasttext.train_supervised(
                   input = conf.process_train_datapath_char,
                   dim = 10,
                   minn = 1,
                   maxn = 4
)
print("fasttext模型训练")


# todo 3 打印模型训练后的关键信息.
# 注意：model调用get_word_vector方法实现
# 例如：获取字符'日'的向量表示(长度: 10维), 查看数字的特征, 例如:
# [ 0.06251448  0.21214555 -0.5582269  -0.2776528  -0.40804806 -0.2712767, -0.03183828 -0.10957453  0.31537786  0.19355372]
# 模型训练后信息:词向量信息,标签信息,词频信息和预测能力,子词信息等
print(model.get_word_vector("日"))

# todo 4 打印模型训练到的 所有类别标签
# 注意：调用labels属性实现
# 例如：# ['__label__stocks', '__label__science'...]
print(model.labels)

# 打印模型训练到的 类别数量长度
print(len(model.labels))

# todo 5 打印模型训练到的 所有单词及对应的词频
print(model.get_words(include_freq=True))

# 用zip()函数, 配对: 字符和频率, 转为列表, 方便查看每个字符对应的出现次数.
# 例如：# [('</s>', 180000), ('0', 60319),  ('：', 28269), ('大', 26024)...]
# 注意： </s>:句子结束标记（sentence end）所以它的词频是样本数
print(list(zip(*model.get_words(include_freq=True))))
print(len(list(zip(*model.get_words(include_freq=True)))))
# 打印华丽的分割线
print('-' * 40)

# todo 6 模型保存
# 注意：调用save_model方法实现
# model_path = config.ft_model_save_path + "/model_char_1_default.bin"
model_path = os.path.join(conf.ft_model_save_path, "model_char_1_default1.bin")
model.save_model(model_path)

print("模型保存成功")

# todo 7 打印模型预测结果
# 注意：调用predict方法实现；预测内容需要是字符级分词的
# 例如: "日 本 地 震 海 啸"
print(model.predict("日 本 地 震 海 啸"))
# todo 8 打印模型词表
# 注意：调用words属性，可截断列表长度打印，方便查看
print(model.words[:10])

# todo 9 打印模型子词
# 注意：调用get_subwords方法实现
# 例如："日本又地震!"
# < 和 > :子词模式中用于包裹词的边界符号
print(model.get_subwords("日本又地震!"))
# todo 10 打印模型输出维度
# 注意：调用get_dimension方法实现
print(model.get_dimension())
# todo 11 打印模型评估
# 结果(10000, 0.8634, 0.8634)
# 说明:(样本数,精确率,召回率)
print(model.test(conf.process_test_datapath_char))

print("结束")
