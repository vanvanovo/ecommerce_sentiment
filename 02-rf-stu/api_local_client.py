"""
概述：
# 该py文件的作用是: 创建一个Flask应用, 并且启动应用(可以理解为: 文字版的客户端)
"""

# 导包
import requests

# 定义接口地址，需要与服务端地址一致
# 测试地址，返回值与发送值一致
# url = 'https://httpbin.org/post'
url = 'http://127.0.0.1:8008/predict'

# 异常处理程序，捕获程序异常信息
try:
    # todo 1 使用input获取用户在控制台输入的内容(text)
    #  再将用户输入的内容以字符串形式返回给你的程序
    # 例如："请输入新闻标题：" 获取用户录入的新闻标题

    # 打印用户输入内容和类型
    # print(f"用户输入---> {text}")
    # print(f"用户输入内容的类型---> {type(text)}")

    # todo 2 向服务端发送请求，得到相应(response)
    #  例如：requests发送post请求到url，配置json文本 {'text': text}

    # 打印相应的内容和类型
    #print('response-->', response.json())
    #print(f'type(response)--> {type(response.json())}')
    pass

except Exception as e:
    print("Error occurred:", e)
