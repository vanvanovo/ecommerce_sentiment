# 测试调用api接口
import time

import requests

url = 'http://127.0.0.1:8011/predict'

start_time = time.time()
try:
    r = requests.post(url, json={'text': "中国人民公安大学2012年硕士研究生目录及书目"})
    print(r.json())

    # 计算耗时
    estimated_time = (time.time() - start_time) * 1000
    print("共花费：%.2f ms" % estimated_time)

except Exception as e:
    print("Error occurred:", e)
