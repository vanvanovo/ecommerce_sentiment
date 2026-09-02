from flask import Flask, request, jsonify
import warnings
from a4_bilstm_predict_fun import predict_fun

warnings.filterwarnings('ignore')

# 创建对象
app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict():
    # 获取客户端数据
    data = request.get_json()
    print(type(data))
    print(data)
    # 预测
    result = predict_fun(data)
    # 返回结果给客户端
    return jsonify(result)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8011, debug=True)
