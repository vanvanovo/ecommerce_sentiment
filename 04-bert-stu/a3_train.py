# 该.py文件的作用 -> 训练和预测.

import torch                            # 深度学习框架, 提供张量计算, 神经网络构建等...
import torch.nn as nn                   # 神经网络模块, 损失函数, 网络层
from torch.optim import AdamW, Adam           # 优化器, 适用于Transformer类模型的优化器, 缓解梯度消失问题.

# 用于评估模型性能的库
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, classification_report
from tqdm import tqdm # 进度条
from model2dev_utils import model2dev # 导入自定义验证函数(例如: 精确率, 召回率...)
from config import Config  # 配置文件类

# 导入数据处理工具类
from a1_dataloader_utils import build_dataloader  # 获取数据集加载器
# 导入模型
from a2_bert_classifer_model import BertClassifier  # 导入BERT分类模型

# 忽略的警告信息
import warnings
warnings.filterwarnings("ignore")

# todo 1 初始化配置文件，包含模型参数、路径等
conf = Config()

# 定义模型训练函数, 封装完整的训练流程(数据加载, 模型训练, 验证, 保存)
def model2train():
    """
    训练 BERT 分类模型并在验证集上评估，保存最佳模型。
    参数：无显式参数，所有配置通过全局 conf 对象获取。
    返回：无返回值，训练过程中保存最佳模型到指定路径。
    """
    # todo 2准备训练/验证/测试数据, 获取其对应的 数据集加载器(train_dataloader, dev_dataloader, test_dataloader)
    # 注意：测试集后续会单独测试
    train_dataloader, dev_dataloader, test_dataloader = build_dataloader()

    # 准备模型
    # todo 3 初始化bert分类模型(model)
    model = BertClassifier()

    # todo 4 将模型放置到指定的设备，
    # 例如，conf.device
    model = model.to(conf.device)

    # todo 5 定义损失函数(loss_fn)
    #  例如：使用: 交叉熵损失 CrossEntropyLoss()
    loss_fn = nn.CrossEntropyLoss()

    # todo 6 定义优化器(optimizer)
    #  例如：使用: AdamW 优化器
    # 注意：参数配置，model.parameters() 和学习率 conf.learning_rate
    optmizer = AdamW(model.parameters(), lr=conf.learning_rate)

    # 开始训练模型
    # todo 7 初始化最优F1分数(best_f1)
    #  用于筛选性能最好的模型(即: 初始值为0, 后续更新)
    best_f1 = 0.0

    # todo 8 遍历epoch轮次，共conf.num_epochs 轮数
    # 外层循环遍历每个训练轮次,每轮都要遍历所有训练数据.
    #  （每次需要设置训练模式，累计损失，预存训练集测试和真实标签）
    for epoch in range(conf.num_epochs):
        # todo 9 设置模型为训练模式
        model.train()   #
        # todo 19 初始化累计损失total_loss;初始化训练集预测列表train_preds和真实标签列表train_labels
        train_preds = []
        train_labels = []
        total_loss = 0.0
        # todo 20 内层循环遍历训练train_dataloader每个批次batch，并设置批次索引i
        # 注意：可以添加进度条，方便查看训练进度
        for i, batch in enumerate(tqdm(train_dataloader)):
            # todo 21 提取bath数据中的input_ids, attention, labels
            input_ids, attention, labels = batch
            # todo 22 input_ids, attention, labels迁移到conf.device设备上
            input_ids = input_ids.to(conf.device)
            attention = attention.to(conf.device)
            labels = labels.to(conf.device)

            # todo 23 前向传播：模型预测得到预测(logits)
            # 注意：输入参数是input_ids和attention_mask
            logits = model(input_ids, attention)

            # todo 24 调用交叉熵函数loss_fn，计算损失(loss)
            # 注意：输入参数是预测结果和标签
            loss = loss_fn(logits, labels)

            # todo 25 将当前损失值(loss.item())累计到total_loss
            total_loss += loss.item()

            # todo 26 获取预测结果(y_pred_list)
            # 注意：使用argmax函数获取logits的最大值索引
            # 例如：调用torch的argmax(logits, dim=1)
            y_pred_list = torch.argmax(logits, dim=1)

            # todo 27 存储预测y_pred_list和真实标签labels，用于计算训练集指标
            # 例如：将y_pred_list结果extend到train_preds中；将labels结果extend到train_labels中
            # 注意：将y_pred_list迁移到cpu()上并调用tolist()转成列表;labels同理
            train_preds.extend(y_pred_list.cpu().tolist())
            train_labels.extend(labels.cpu().tolist())
            # todo 28 优化器梯度清零
            # 例如：优化器调用zero_grad()实现
            optmizer.zero_grad()
            # todo 29 loss反向传播，计算梯度
            # 例如：loss调用backward()实现
            loss.backward()
            # todo 39 优化器参数更新：根据梯度更新模型参数
            # 例如：优化器调用step()函数，执行一次参数更新
            optmizer.step()
            # 每10个批次或一个轮次结束，计算训练集指标 (运行时取消注释)
            if (i + 1) % 10 == 0 or i == len(train_dataloader) - 1:
                # 计算准确率和f1值
                acc = accuracy_score(train_labels, train_preds)
                f1 = f1_score(train_labels, train_preds, average='macro')
                # 获取batch_count，并计算平均损失
                batch_count = i % 10 + 1
                avg_loss = total_loss / batch_count
                # 打印训练信息
                print(f"\n轮次: {epoch + 1}, 批次: {i + 1}, 损失: {avg_loss:.4f}, acc准确率:{acc:.4f}, f1分数:{f1:.4f}")
                # 清空累计损失和预测和真实标签
                total_loss = 0.0
                train_preds, train_labels = [], []

            # 每100个批次或一个轮次结束，计算验证集指标，打印，保存模型
            if (i + 1) % 100 == 0 or i == len(train_dataloader) - 1:
                # todo 40 每隔 100个批次或一个轮次，验证下模型性能(report, f1score, accuracy, precision, recall)
                # 例如：调用model2dev验证模型在验证集上的性能
                # 注意：参数包括，模型model，验证集加载器，conf.device
                #  计算在测试集的评估报告，准确率，精确率，召回率，f1值
                report, f1score, accuracy, precision, recall = model2dev(model, dev_dataloader, conf.device)

                print("验证集评估报告：\n", report)
                print(f"验证集的f1: {f1score:.4f}, accuracy:{accuracy:.4f}, precision:{precision:.4f}, recall:{recall:.4f}")
                # 将模型再设置为训练模式
                model.train()
                # 如果验证F1分数优于历史最佳，保存模型
                if f1score > best_f1:
                    # 更新历史最佳F1分数
                    best_f1 = f1score
                    # 保存模型
                    torch.save(model.state_dict(), conf.model_save_path)
                    print("保存模型成功, 当前f1分数-->", best_f1)


if __name__ == '__main__':
    model2train()
