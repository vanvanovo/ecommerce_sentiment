# 导入工具包
import fasttext
from config import Config

# todo 1 初始化配置文件.
conf = Config()
# todo 2. 模型训练: 使用 fasttext训练 词级 文本分类模型(model)
# 例如：# fasttext.train_supervised(): fasttext的核心训练函数, 用于训练: 有监督学习分类模型
# 注意：重点配置input: 输入的处理后的训练数据 词级文件
#      可配置seed 方便效果复现
model = fasttext.train_supervised(input=conf.process_train_datapath_word, seed=42)
# # todo 3 其余代码与默认模式类似，不在赘述
# # 2、模型保存
model_path = conf.ft_model_save_path + "/model_word_1_default1.bin"
model.save_model(model_path)
#
# # 3、模型预测
print(model.predict("日本 地震 ： 金吉列 关注 在 日 学子 系列报道"))
#
# # 4、模型词表的查看
print(model.words[:10])
#
# # 5、模型子词查看
print(model.get_subwords("日本"))
# # 输出模型维度
print(model.get_dimension())
#
# # 6、模型评估
print(model.test(conf.process_test_datapath_word))
