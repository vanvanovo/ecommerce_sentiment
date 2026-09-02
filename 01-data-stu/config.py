# 定义配置文件类, 集中管理模型和训练所需的参数.
# 针对 京东商品评论情感分析(ecommerce_sentiment), 二分类.
# 数据源: DAMO_NLP/jd (魔搭), 已清洗为 text\tlabel 格式(jd_data/prepared_clean/*.txt).
import os


class Config:
    # 初始化函数
    def __init__(self):
        # 配置统一的数据根目录
        self.root_path = r'D:\work\ecommerce_sentiment/'

        # 训练集路径
        self.train_path = self.root_path + '01-data-stu/jd_data/prepared_clean/train.txt'
        # 验证集路径
        self.dev_path = self.root_path + '01-data-stu/jd_data/prepared_clean/dev.txt'
        # 测试集路径(用 dev 的一部分作测试, 与 02-rf 对齐)
        self.test_path = self.root_path + '01-data-stu/jd_data/prepared_clean/dev.txt'
        # 类别定义文件(二分类: negative 0 / positive 1)
        self.class_doc_path = self.root_path + '01-data-stu/class.txt'

        # 文件存在性检查
        path_check = {
            "train_path": self.train_path,
            "dev_path": self.dev_path,
            "test_path": self.test_path,
        }
        for data_name, data_path in path_check.items():
            if not os.path.exists(data_path):
                print(f"数据集文件不存在 {data_name}: {data_path}")


if __name__ == '__main__':
    conf = Config()
    print(f"训练集路径---> {conf.train_path}")
    print(f"测试集路径---> {conf.test_path}")
    print(f"验证集路径---> {conf.dev_path}")
    print("结束")