# 软标签蒸馏
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from sklearn.metrics import f1_score, accuracy_score
from model2dev_utils import model2dev
from config import Config
from a0_dataloader_utils import build_dataloader
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")  # 忽略警告信息
from a1_bert_classifer_model import BertClassifier
from a2_bilstm_classifer_model import BiLSTMModel

# todo 1 初始化配置文件.
conf = Config()  # 加载配置文件

# 导入教师模型和学生模型，进行训练
def model2train():
    # 准备数据
    train_dataloader, dev_dataloader, test_dataloader = build_dataloader()

    # 准备模型
    # todo 2 定义教师模型Bert
    # 例如：teacher_model = BertClassifier()

    # todo 3 加载模型参数
    # 例如：state_dict = torch.load(conf.model_save_path)

    # todo 4 加载模型参数到模型中
    # 例如：teacher_model.load_state_dict(state_dict)

    # todo 5 将模型移动到指定设备（GPU 或 CPU）
    # 例如：teacher_model.to(conf.device)

    # todo 6 定义学生模型
    # 例如：student_model = BiLSTMModel()

    # todo 7 将模型移动到指定设备（GPU 或 CPU）
    # 例如：student_model.to(conf.device)

    # todo 8 准备损失函数
    # 例如：criterion = nn.CrossEntropyLoss()

    # todo 9 准备优化器(优化的是学生模型)
    # 例如： optimizer = AdamW(student_model.parameters(), lr=conf.lstm_learning_rate)

    # 开始训练模型
    # 最佳F1值
    best_f1 = 0.0
    # todo 10 定义模型蒸馏的参数包括 权重系数alpha，温度参数T
    # 蒸馏参数： 温度参数T用于控制softmax()概率变平滑,这样保证学生模型学到潜在知识
    # 例如：T = 2.0
    # 蒸馏参数：软标签权重用于平衡软硬标签的损失占比,alpha是软标签占比,1-alpha是硬标签占比
    # 例如：alpha = 0.7

    #  遍历每个 epoch
    for epoch in range(conf.num_epochs):
        # todo 11 设置学生模型为训练模式，设置教师模型为评估模式（不更新权重）
        # 例如：teacher_model.eval()
        # 例如：student_model.train()


        # 设置累计损失，初始化预测和真实标签
        total_loss = 0.0
        pred_label_list, true_label_list = [], []

        # 遍历训练数据批次
        for i, batch in enumerate(tqdm(train_dataloader, desc="训练集训练中...")):
            # todo 12 从batch中获取 input_ids, attention_mask, label，并放置到设备.to(conf.device)
            # 例如：
            # input_ids, attention_mask, labels = batch
            # input_ids = input_ids.to(conf.device)
            # attention_mask = attention_mask.to(conf.device)
            # labels = labels.to(conf.device)




            # todo 13 前向传播,获取学生模型的输出
            # 例如：
            # student_logits = student_model(input_ids, attention_mask)
            # student_labels = torch.argmax(student_logits, dim=1)


            # todo 14 获取教师模型的输出 logits软标签与教师模型的硬标签
            # 调用with torch.no_grad():不进行参数更新
            # 例如：
            # with torch.no_grad():
            #     获取教师的logits分数
            #     teacher_logits = teacher_model(input_ids, attention_mask)
            # 获取教师硬标签
            #     teacher_labels = torch.argmax(teacher_logits, dim=1)



            # todo 15 计算软标签损失（KL 散度）
            # 计算教师模型的概率
            # 例如：teacher_log_soft_labels = F.softmax(teacher_logits / T, dim=1)

            # todo 5 计算学生模型的 log-概率
            # 例如：student_log_soft_labels = F.log_softmax(student_logits / T, dim=1)

            # todo 6 计算KL 散度
            # 注意：调用F.kl_div实现
            # 参数1是预测值；参数2是目标值；参数3是聚合方式，使用batchmean
            # 例如： soft_loss = F.kl_div(student_log_soft_labels,teacher_log_soft_labels, reduction='batchmean')

            # todo 7 计算硬标签损失（交叉熵，使用教师模型的标签或者真实的数据标签）
            # 注意：
            # 可以使用教师输出标签，hard_loss1 = criterion(student_logits, teacher_labels)
            # 也可以使用真实标签， hard_loss1 = criterion(student_logits, labels)
            # 这两个是二选一，信任教师模型就选择第一个；信任真实标签就选第二个
            # 例如：hard_loss1 = criterion(student_logits, labels)

            # todo 8 总损失：软标签和硬标签损失的加权和
            # 注意：根据alpha设置权重
            # 例如：loss = (1 - alpha) * hard_loss1 + alpha * soft_loss

            # 打印loss
            # print("loss-->", loss)

            # todo 9 以下代码同bert之前的bert模型
            #  汇集损失和预测标签以及真实标签
            total_loss += loss.item()
            pred_label_list.extend(student_labels.cpu().tolist())
            true_label_list.extend(labels.cpu().tolist())

            # 清空优化器梯度
            optimizer.zero_grad()
            # 反向传播计算梯度
            loss.backward()
            # 更新模型参数
            optimizer.step()

            # 每10个批次或一个轮次结束，计算训练集指标
            if (i + 1) % 10 == 0 or i == len(train_dataloader) - 1:
                # 计算准确率和f1值
                acc = accuracy_score(true_label_list, pred_label_list)
                f1 = f1_score(true_label_list, pred_label_list, average='macro')
                # 获取batch_count，并计算平均损失
                batch_count = i % 10 + 1
                avg_loss = total_loss / batch_count
                # 打印训练信息
                print(f"\nEpoch: {epoch + 1}, Batch: {i + 1}, Loss: {avg_loss:.4f}, acc:{acc:.4f}, f1:{f1:.4f}")
                # 清空累计损失和预测和真实标签
                total_loss = 0.0
                true_label_list, pred_label_list = [], []

            # 每 100 个 batch 验证一次,batch级别验证model2dev
            if (i + 1) % 100 == 0 or i == len(train_dataloader) - 1:
                # 获取验证集的预测结果
                reports, f1, acc, precision, recall = model2dev(student_model, dev_dataloader, conf.device)
                student_model.train()
                print("\n验证集评估报告：\n", reports)
                print(f"验证集f1，accuracy, precision, recall: {f1:.4f}, {acc:.4f}, {precision:.4f}, {recall:.4f}")

                # 验证f1并保存最好模型
                if f1 > best_f1:
                    best_f1 = f1
                    # 保存模型
                    torch.save(student_model.state_dict(), conf.bilstm_save_model_path)


if __name__ == '__main__':
    model2train()
