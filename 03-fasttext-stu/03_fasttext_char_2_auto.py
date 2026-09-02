# 导入工具包
import fasttext
from config import Config
import datetime

"""
fastText 自从 v0.9.0 版本起，支持了内置的 自动调参（Auto Tuning） 功能，
通过设置 autotuneValidationFile 参数，可以让模型在训练时根据你提供的验证集自动调整以下超参数：
学习率（lr）
epoch 数量（epoch）
词向量维度（dim）
字符 n-gram 长度范围（minn, maxn）
单词 n-gram 数量（wordNgrams）
损失函数类型（loss）
注意: 若你没有设置 autotuneValidationFile，fastText 将不会进行自动调参!!!
"""
# todo 1 初始化配置文件
conf = Config()

# 获取时间
# 获取当前时间, 并格式化为: YYYYMMDD的形式, 例如: 20250914
current_time = datetime.datetime.now().date().today().strftime("%Y%m%d")
print(current_time)

# todo 2. 模型训练(自动调参模式): 使用 fasttext训练 字符级 文本分类模型(model)
# 例如：# fasttext.train_supervised(): fasttext的核心训练函数, 用于训练: 有监督学习分类模型
# 注意：重点配置input: 输入的处理后的训练数据字符级文件
#       autotuneValidationFile: 自动调参的验证集
#       autotuneDuration: 自动调参时长
#       thread： 训练线程数, 设置为1(单线程) 避免多线程的随机波动, 确保实验可复现
#       verbose: 可设置0,1,2,3,其中3输出更详细的调试信息，包括自动调参过程、每个 epoch 的 loss等
#       seed: 设置随机数生成器的种子值，确保实验的可重复性
model = fasttext.train_supervised(input=conf.process_train_datapath_char,
                          autotuneValidationFile=conf.process_dev_datapath_char,
                          autotuneDuration=60,
                          thread=1,
                          verbose=3,
                          seed=42)

# todo 其余代码与默认模式类似，不在赘述
# 2、模型保存
# 保存路径
model_path = conf.ft_model_save_path + f"/model_char_2_auto_{current_time}.bin"
model.save_model(model_path)
# 3、模型预测
print(model.predict("日 本 地 震 ： 金 吉 列 关 注 在 日 学 子 系 列 报 道"))

# 4、模型词表的查看
print(model.words[:10])

# 5、模型子词查看
print(model.get_subwords("日"))
# 打印向量维度
print(model.get_dimension())

# 6、模型评估
print(model.test(conf.process_test_datapath_char))
