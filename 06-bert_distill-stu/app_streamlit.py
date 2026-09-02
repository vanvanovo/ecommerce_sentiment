import streamlit as st
import requests
import time

# streamlit页面
st.title("投满分分类项目")
st.write("这是一个投满分分类项目")

text = st.text_input("请输入文本")
# 点击获取分类
url = 'http://127.0.0.1:8011/predict'
if st.button("基于蒸馏后bert模型获取分类！"):
    start_time = time.time()
    try:
        # 发送post请求
        r = requests.post(url, json={'text': text})
        print(r.json())

        # 计算耗时
        estimated_time = (time.time() - start_time) * 1000
        st.write("耗时：", estimated_time, "ms")
        # 显示预测结果到页面
        st.write("预测结果：", r.json()["pred_class"])
    except Exception as e:
        print("Error occurred:", e)
