# 配置文件类, 集中管理模型和训练所需的参数.
# 针对 京东商品评论情感分析(ecommerce_sentiment), 二分类(0=差评,1=好评).
# 数据源: DAMO_NLP/jd (魔搭), 已清洗为 text\tlabel 格式(jd_data/prepared_clean/*.txt).
class Config:
    # 初始化函数
    def __init__(self):
        # 配置统一的数据根目录
        self.root_path = r'D:\work\ecommerce_sentiment/'

        # 数据库文件(两列: text<TAB>label, 无表头)
        self.train_datapath = self.root_path + '01-data-stu/jd_data/prepared_clean/train.txt'
        self.test_datapath = self.root_path + '01-data-stu/jd_data/prepared_clean/dev.txt'
        self.dev_datapath = self.root_path + '01-data-stu/jd_data/prepared_clean/dev.txt'
        # 类别文档(二分类: negative 0 / positive 1)
        self.class_doc_path = self.root_path + "01-data-stu/class.txt"
        # 停用词路径
        self.stop_words_path = self.root_path + "01-data-stu/stopwords.txt"

        # 处理后的数据路径(RF 分词后输出到 final_data)
        self.process_train_datapath = self.root_path + "02-rf-stu/final_data/train_process.txt"
        self.process_test_datapath = self.root_path + "02-rf-stu/final_data/test_process.txt"
        self.process_dev_datapath = self.root_path + "02-rf-stu/final_data/dev_process.txt"

        # 保存模型路径
        self.rf_model_save_path = self.root_path + r"02-rf-stu/save_model/rf_model.pkl"
        self.tfidf_model_save_path = self.root_path + r"02-rf-stu/save_model/tfidf_model.pkl"
        # 模型预测结果
        self.model_predict_result = self.root_path + r"02-rf-stu/result/predict_result.txt"


if __name__ == '__main__':
    config = Config()
    print(config.train_datapath)
    print(config.test_datapath)
    print(config.dev_datapath)
    print("结束")