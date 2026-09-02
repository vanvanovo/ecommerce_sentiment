# 配置文件类, 集中管理模型和训练所需的参数.
# 针对 京东商品评论情感分析(ecommerce_sentiment), 二分类(0=差评=negative, 1=好评=positive).
# 数据源: DAMO_NLP/jd (魔搭), 已清洗为 text\tlabel 格式(jd_data/prepared_clean/*.txt).
class Config:
    # 初始化函数
    def __init__(self):
        # 配置统一的数据根目录
        self.root_path = r'D:\work\ecommerce_sentiment/'

        # 原始数据路径(text\tlabel, 无表头)
        self.train_datapath = self.root_path + '01-data-stu/jd_data/prepared_clean/train.txt'
        self.test_datapath = self.root_path + '01-data-stu/jd_data/prepared_clean/dev.txt'
        self.dev_datapath = self.root_path + '01-data-stu/jd_data/prepared_clean/dev.txt'
        # 类别文档(二分类: negative 0 / positive 1)
        self.class_doc_path = self.root_path + "01-data-stu/class.txt"

        # 数据处理保存路径
        # 字符级别 fasttext
        self.process_train_datapath_char = self.root_path + "03-fasttext-stu/final_data/train_process_char.txt"
        self.process_test_datapath_char = self.root_path + "03-fasttext-stu/final_data/test_process_char.txt"
        self.process_dev_datapath_char = self.root_path + "03-fasttext-stu/final_data/dev_process_char.txt"
        # 词级别 fasttext
        self.process_train_datapath_word = self.root_path + "03-fasttext-stu/final_data/train_process_word.txt"
        self.process_test_datapath_word = self.root_path + "03-fasttext-stu/final_data/test_process_word.txt"
        self.process_dev_datapath_word = self.root_path + "03-fasttext-stu/final_data/dev_process_word.txt"

        # 保存处理完的数据(用于训练)
        self.final_data = self.root_path + '03-fasttext-stu/final_data'

        # fasttext 模型路径
        self.ft_model_save_path = self.root_path + '03-fasttext-stu/save_models'

        # 类别字典 {标签索引:标签名称} —— 依赖 class.txt(negative/positive 两行)
        self.id2class_dict = {i: line.strip() for i, line in enumerate(open(self.class_doc_path, encoding='utf-8'))}


if __name__ == '__main__':
    conf = Config()
    print('conf.train_datapath-->', conf.train_datapath)
    print('conf.id2class_dict-->', conf.id2class_dict)
    print("结束")