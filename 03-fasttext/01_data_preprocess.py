"""
概述：
    该py脚本文件用于: 数据的预处理.
"""

# 导包
import jieba            # 中文分词库, 将文本按照 词语 进行分割.
from config import *    # 从config模块中, 导入所有的数据

# todo 1 初始化配置文件.
conf = Config()

# 定义数据处理函数, 将原始文本 -> fasttext模型所需的输入格式.
def process_data(datapath, processed_datapath, is_char=True):
    """
    数据处理函数, 按照fasttext模型所需的格式输入.
    实现逻辑：
    第一步：打开原始数据和数据保存文件
    第二步：循环遍历每个数据行
    第四步：去除首尾字符，跳过空行，拆分文本和标签，标签转int，int获取真实标签名称
    第五步：对文本进行字符和词级别分词
    第六步：将__label__，真实标签，分词 拼接成目标格式数据
    第七步：数据保存到本地
    :param datapath: 原始数据文件的路径(输入)
    :param processed_datapath: 处理后的数据文件路径(输出)
    :param is_char: 布尔值, True表示按照字符级别切割, False表示按词语级别切割(jieba分词)
    :return:
    """
    # 第一步：打开原始数据和数据保存文件
    # todo 2 打开原始数据文件(f)
    with open(datapath, 'r', encoding='utf-8') as f:
        # todo 3 打开数据保存文件(fw)
        with open(processed_datapath, 'w', encoding='utf-8') as fw:
            # 第二步：循环遍历每个数据行
            # todo 4 循环遍历读取f中的每一行(line)
            for line in f.readlines():
                # 第四步：去除首尾字符，跳过空行，拆分文本和标签，标签转int，int获取真实标签名称
                # 实现将line转换为fasttext格式: __label__字符串标签 分词后文本
                # todo 5 去除首尾多余字符
                line = line.strip()
                # todo 6 如果处理后为空, 说明是空行, 就跳过, 继续往后continue
                if not line:
                    continue
                # todo 7 分割文本和标签(text,label)
                # 注意：分割是？
                text, label = line.split('\t')

                # todo 8 把标签转成int类型(label)
                # 例如：'3' -> 3
                label = int(label)

                # todo 9 获取标签字符串(label_str)
                # 通过conf中的id2class_dict, 将标签整数 映射为 对应类别的字符串
                # id2class_dict格式: {0: 'positive', 1: 'negative', ...}
                label_str = conf.id2class_dict[label]
                print(f"label ---> {label_str}")

                # 第五步：对文本进行字符和词级别分词
                # todo 10 根据is_char参数决定文本分词方式
                if is_char:
                    # todo 11 字符级别分割(text_split)， 例如： '天气好' -> '天 气 好'
                    # 例如：text文本使用list拆分成字符列表，在使用空格拼接成字符串（可举例说明）
                    text_split = " ".join(list(text))
                    # todo 12 词 级别分割, 使用jieba分词库进行分词(text_split)，例如: 今天天气很好' -> '今天 天气 很 好'
                    # 例如：text文本使用jieba.lcut拆分成词列表，再使用空格拼接成字符串
                else:
                    text_split = " ".join(jieba.lcut(text))

                # 打印测试结果
                print(text_split)
                print(label_str)

                # 第六步：将__label__，真实标签，分词 拼接成目标格式数据
                # todo 13 构建fasttext要求的数据格式(ft_line)
                # 例如: __label__positive  天 气 好 "\n"
                ft_line = "__label__" + label_str + " " + text_split + "\n"

                # 第七步：数据保存到本地
                # todo 14 将构建好的行 写入到 处理后的数据文件中.
                # 例如：调用fw中的write函数
                fw.write(ft_line)


# 测试
if __name__ == '__main__':
    # 字符级别处理
    # 训练集
    process_data(conf.train_datapath, conf.process_train_datapath_char, is_char=True)
    # 测试集
    process_data(conf.test_datapath, conf.process_test_datapath_char, is_char=True)
    # 验证集
    process_data(conf.dev_datapath, conf.process_dev_datapath_char, is_char=True)

    # 分词级别处理
    # 训练集
    process_data(conf.train_datapath, conf.process_train_datapath_word, is_char=False)
    # 测试集
    process_data(conf.test_datapath, conf.process_test_datapath_word, is_char=False)
    # 验证集
    process_data(conf.dev_datapath, conf.process_dev_datapath_word, is_char=False)

    # 打印结束标志
    print("结束")
