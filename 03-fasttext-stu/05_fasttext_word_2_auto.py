# 导入工具包
import fasttext
from config import Config
import datetime

# todo 1 初始化配置文件
conf = Config()
# 获取时间
current_time = datetime.datetime.now().date().today().strftime("%Y%m%d")

# 自动调参学习
# todo 2. 模型训练(自动调参模式): 使用 fasttext训练 字符级 文本分类模型(model)
# 例如：# fasttext.train_supervised(): fasttext的核心训练函数, 用于训练: 有监督学习分类模型
# 注意：重点配置input: 输入的处理后的训练数据字符级文件
#       autotuneValidationFile: 自动调参的验证集
#       autotuneDuration: 自动调参时长
#       verbose: 可设置0,1,2,3,其中3输出更详细的调试信息，包括自动调参过程、每个 epoch 的 loss等
#       seed: 设置随机数生成器的种子值，确保实验的可重复性
model = fasttext.train_supervised(input=conf.process_train_datapath_word,
                                  autotuneValidationFile=conf.process_dev_datapath_word,
                                  autotuneDuration=20,
                                  verbose=3,
                                  seed=42)

# # todo 其余代码与默认模式类似，不在赘述
# 2、模型保存
# 保存路径
model_path = conf.ft_model_save_path + f"/model_word_2_auto_{current_time}.bin"
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
# # 打印向量维度
print(model.get_dimension())
#
# # 6、模型评估
print(model.test(conf.process_test_datapath_word))
