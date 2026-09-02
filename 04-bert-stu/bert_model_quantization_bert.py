from a2_bert_classifer_model import BertClassifier # 导入bert模型
from config import Config                          # 导入配置文件
import torch                                       # 深度学习框架，用户模型构建和训练等
from a1_dataloader_utils import build_dataloader   # 数据加载器
from a3_train import model2dev                     # 模型验证

# 拓展: 打印当前支持的量化引擎，以了解环境中可用的量化计算后端
print('engines-->', torch.backends.quantized.supported_engines)  # ['none', 'onednn', 'x86', 'fbgemm']
# 拓展: 设置上述支持量化引擎 ,注意:不是必须的,有默认操作,此处就是显式写出,让大家了解
# PyTorch支持两种：fbgemm和qnnpack
# fbgemm：适合于服务器端
# qnnpack：适合于移动端
# torch.backends.quantized.engine = 'fbgemm'

# todo 1 初始化配置文件.

# 准备数据
train_dataloader, dev_dataloader, test_dataloader = build_dataloader()

# todo 2 初始化bert分类模型(model)
# 例如：model = BertClassifier()

# todo 3加载预训练模型
# 注意：先使用torch.load函数从conf.model_save_path中加载模型，再填充到model.load_state_dict函数中
# 注意：利用map_location参数将模型映射到'cpu'
# 例如：model.load_state_dict(torch.load(conf.model_save_path, map_location='cpu'))

# todo 4 模型评估模式(eval)
# 例如：model.eval()

# 打印量化前的模型
# print("量化前模型-->",model)

# todo 5 打印量化前模型性能
# report, f1score, accuracy, precision, recall = model2dev(model, dev_dataloader, conf.device)
# print('report-->', report)
# print("量化前f1score-->", f1score)
# todo 6 模型动态量化(quantized_model)
# 调用torch.quantization.quantize_dynamic函数实现
# 参数设置：参数1 模型为model; 参数2 qconfig_spec为 指定需要量化的层，字典格式，例如对torch.nn.Linear 线性层进行量化；参数3 指定类型dtype=torch.qint8(默认)
# model：要量化的原始模型
# qconfig_spec：指定模型中哪些部分需要量化
# dtype：量化后的数据类型，默认就是qint8，
# 例如：quantized_model = torch.quantization.quantize_dynamic(model, qconfig_spec={torch.nn.Linear}, dtype=torch.qint8)

 # 打印量化后的模型
# print("量化后模型-->",quantized_model)
# todo 7 打印量化后的模型性能
# report2, f1score2, accuracy2, precision2, recall2 = model2dev(quantized_model, dev_dataloader, conf.device)
# print('report2-->', report2)
# print("量化后f1score-->", f1score2)
# todo 8 量化后的模型保存
# 注意：调用torch.save函数实现
# 参数1 量化后的模型参数 .state_dict()；参数2 量化模型保存路径 conf.bert_model_quantization_model_path
# 例如：torch.save(quantized_model.state_dict(), conf.bert_model_quantization_model_path)

# 打印模型保存路径
# print(f"模型已经保存，保存地址为：{conf.bert_model_quantization_model_path}")
