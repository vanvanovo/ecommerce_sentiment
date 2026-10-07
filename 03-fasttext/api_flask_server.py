"""
# 该py文件的主要任务: 通过Flask组件, 构建 路由 + 预测函数的 应用.

"""
# 导包
# 第一段：固定写法，导包并构建类实例
from flask import Flask, request, jsonify # request 处理客户都发送过来的请求，例如报文解析，缓存处理等；jsonify 将给定的参数序列化为JSON
from ft_predict_func import predict_func  # 可调用的模型预测函数

# 忽略警告信息
import warnings
warnings.filterwarnings('ignore')

# todo 1 构建类实例
app = Flask(__name__)

# 第二段：固定写法，装饰器 + 路由地址 + 业务函数
# 设定url: http://127.0.0.1:8008/predict
# 设定POST请求
@app.route('/predict', methods=['POST'])
def predict():
    """
    实现逻辑：
    第一步：获取客户端数据
    第二步：调用预测函数，进行预测，得到预测结果
    第三步：预测结果封装成json，返回客户端
    注意：获取数据和返回数据，涉及计算机网络底层通信原理，项目中理解使用方法即可
    :return: 包含预测结果的数据
    """
    # todo 2 获取post请求携带的用户数据(data)
    # 涉及计算机网络底层原理，理解使用方法即可，例如：get_json()
    data = request.get_json()

    # 打印客户端发送过来的数据
    print('data-->', data)
    # 打印客户端发送过来的数据类型
    print('type(data)-->', type(data))

    # todo 3 调用预测函数，对用户请求数据进行预测，得到预测结果(result)
    result = predict_func(data)

    # 打印预测结果
    print("result-->", result)
    # 打印使用jsonify封装后的结果
    print("jsonify(result)-->", jsonify(result))
    # todo 4 返回jsonify封装后的结果到客户端
    return jsonify(result)


# 启动服务，测试
# 第三段：固定写法，调用run方法，参数可根据实际地址修改
if __name__ == '__main__':
    # 0.0.0.0 整个局域网内都可以访问
    # todo 5 起服务端 + 端口监听 + 接收客户端请求（自动发送返回值到客户端）
    app.run(host='127.0.0.1', port=5000, debug=True)

    print("结束")