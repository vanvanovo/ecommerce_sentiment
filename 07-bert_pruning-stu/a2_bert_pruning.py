"""
BERT 全局非结构化剪枝：对所有 encoder 层注意力权重剪枝 30%，L1 范数。
"""
import torch
import torch.nn.utils.prune as prune
from a1_bert_classifer_model import BertClassifier
from a0_dataloader_utils import build_dataloader
from config import Config
from model2dev_utils import model2dev

# todo 1 初始化配置文件，包含模型参数、路径等

# 提前定义权重稀疏度,重复使用
# 计算权重等于0的参数数量，占全部权重的比例
def compute_sparsity(model):
    """
    计算权重稀疏度
    :param model:
    :return: 稀疏度
    """
    total_params = 0
    zero_params = 0
    layer_num = len(model.bert.encoder.layer)
    for i in range(layer_num):
        weight = model.bert.encoder.layer[i].attention.self.query.weight
        zero_params += (weight == 0).sum().item()
        total_params += weight.numel()
    return zero_params / total_params


# 提前定义打印权重,重复使用
def print_weights(weight, name, rows=5, cols=5):
    """
    打印权重前n行，前n列
    :param weight: 权重
    :param name: 名称
    :param rows: 前rows行
    :param cols: 前clos列
    :return:
    """
    print(f"\n{name}（前 {rows}x{cols}）：")
    print('weight-->', weight[:rows, :cols])


if __name__ == '__main__':
    # todo 2 利用数据加载器build_dataloader，获取train_dataloader, dev_dataloader, test_dataloader

    # todo 3 实例化BertClassifier成(model)

    # 加载模型参数
    # state_dict = torch.load(conf.model_save_path)
    # model.load_state_dict(state_dict)
    # model.to(conf.device)
    # 打印模型
    # print("model模型结构-->", model)

    # 打印减枝前模型参数
    # print("剪枝前模型（查看第1层）：")
    # print_weights(model.bert.encoder.layer[0].attention.self.query.weight, "layer[0].attention.self.query.weight 剪枝前")
    # 打印减枝前稀疏度
    # sparsity = compute_sparsity(model)
    # print(f"\n减枝前稀疏度: {sparsity:.4f}")
    # 打印减枝前模型验证准确率和f1值
    # report, f1score, accuracy, precision, recall = model2dev(model, dev_dataloader, conf.device)
    # print(f"\n剪枝前准确率: {accuracy:.4f}, F1: {f1score:.4f}")

    print('========================================剪枝开始=====================================')
    # 全局非结构化剪枝：所有 encoder 层 query 权重 30%
    # 获取Q的权重
    # parameters_to_prune = [(model.bert.encoder.layer[i].attention.self.query, 'weight') for i in range(12)]
    # 获取Q，K，V的权重
    # parameters_to_prune_query = [(model.bert.encoder.layer[i].attention.self.query, 'weight') for i in range(12)]
    # parameters_to_prune_key = [(model.bert.encoder.layer[i].attention.self.key, 'weight') for i in range(12)]
    # parameters_to_prune_value = [(model.bert.encoder.layer[i].attention.self.value, 'weight') for i in range(12)]
    # # 把Q，K，V的权重组合一起，用于剪枝
    # parameters_to_prune_list = parameters_to_prune_query + parameters_to_prune_key + parameters_to_prune_value

    # print("parameters_to_prune-->", parameters_to_prune_list)
    # todo 4 执行全局非结构化剪枝
    #  注意：调用prune.global_unstructured 函数
    # 参数1 需要剪枝的参数列表parameters_to_prune_list；
    # 参数2 剪枝方法 pruning_method=prune.L1Unstructured；
    # 参数3 剪枝比例 amount=0.3
    # prune.global_unstructured(parameters_to_prune_list, # 所有的权重参数
    #                           pruning_method=prune.L1Unstructured,  # type: ignore  # L1Unstructured绝对值方式计算
    #                           amount=0.3, # 比例30%
    # )
    # 移除剪枝的额外结构(固化剪枝)
    # 进行的剪枝操作“固化”到模型中，使其不再是临时的剪枝掩码（mask），而是将剪枝后的稀疏权重永久保存下来
    # for module, param in parameters_to_prune_list:
    #     # 如果不调用 prune.remove()，剪枝只是临时生效，并没有真正去除模型中的参数。
    #     # 调用remove后，模型中只保留剪枝后的权重，不再带有剪枝相关的额外结构
    #     prune.remove(module, param)
    print('========================================剪枝结束=====================================')
    # 打印剪枝后模型参数
    # print("\n剪枝后模型（查看第1层）：")
    # print_weights(model.bert.encoder.layer[0].attention.self.query.weight, "layer[0].attention.self.query.weight 剪枝后")
    # 剪枝后稀疏度
    # sparsity = compute_sparsity(model)
    # print(f"剪枝后稀疏度: {sparsity:.4f}")

    # 剪枝后模型验证准确率和f1值
    # report, f1score, accuracy, precision, recall = model2dev(model, dev_dataloader, conf.device)
    # print(f"\n剪枝后准确率: {accuracy:.4f}, F1: {f1score:.4f}")

    # 保存模型权重
    # torch.save(model.state_dict(), conf.bert_pruning_model_save_path)
    # print("已成功保存模型~")
