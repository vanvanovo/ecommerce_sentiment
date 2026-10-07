import torch
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, classification_report
from torch.distributed._tensor.experimental import attention
from tqdm import tqdm
from config import Config

# 导入数据处理工具类
from a1_dataloader_utils import build_dataloader
# 导入bert模型
from a2_bert_classifer_model import BertClassifier

# 忽略的警告信息
import warnings
warnings.filterwarnings("ignore")

# todo 1 初始化配置文件，包含模型参数、路径等
conf = Config()

def model2predict():
    # todo 2 从数据集加载器中获取数据(train_dataloader, dev_dataloader, test_dataloader)
    # 这里使用测试集数据
    train_dataloader, dev_dataloader, test_dataloader = build_dataloader()
    # 准备模型
    # todo 2 初始化bert分类模型(model)
    model = BertClassifier()
    # todo 3加载预训练模型
    # 注意：先使用torch.load函数从conf.model_save_path中加载模型，再填充到model.load_state_dict函数中
    model.load_state_dict(torch.load(conf.model_save_path, map_location=torch.device('cpu')))

    # todo 4 将模型移动到指定的设备
    model.to(conf.device)
    # todo 5 设置模型为评估模式（禁用 dropout,并改变batch_norm行为）
    model.eval()
    # 初始化列表，存储预测结果和真实标签
    all_preds, all_labels = [], []
    # todo 6 torch.no_grad()禁用梯度计算以提高效率并减少内存占用
    with torch.no_grad():
        # todo 7 遍历test_dataloader数据加载器，得到每个批次batch，并设置批次索引i；逐批次进行预测
            for i, batch in enumerate(tqdm(test_dataloader)):
            # todo 8 提取bath数据中的input_ids, attention, labels
                input_ids, attention, labels = batch

            # todo 9 input_ids, attention, labels迁移到conf.device设备上
                input_ids = input_ids.to(conf.device)
                attention = attention.to(conf.device)
                labels = labels.to(conf.device)

            # todo 10 前向传播：模型预测得到预测(outputs)
            # 注意：输入参数是input_ids和attention_mask
                outputs = model(input_ids, attention)

            # 打印输出output和shape
            #     print("output-->", outputs, outputs.shape)

                # todo 11 获取预测结果（最大 logits分数 对应的类别）
                # 注意：使用argmax函数获取logits的最大值索引
                # 例如argmax(logits, dim=1)
                y_pred_list = torch.argmax(outputs, dim=1)

                # 存储预测和真实标签(运行时取消注释)
                all_preds.extend(y_pred_list.cpu().tolist())
                all_labels.extend(labels.cpu().tolist())
                if i > 10:
                    break
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='macro')
    f1score = f1_score(all_labels, all_preds, average='macro')
    recall = recall_score(all_labels, all_preds, average='macro')
    report = classification_report(all_labels, all_preds)

    print("accuracy-->", accuracy)
    print("precision-->", precision)
    print("f1score-->", f1score)
    print("recall-->", recall)
    print("report-->", report)


if __name__ == '__main__':
    model2predict()