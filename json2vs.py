# -*- coding: utf-8 -*-
# @Time : 2023/5/3 23:22
# @Author : fhp
# @File : json2vs.py
# @Software : PyCharm
import json

# 读取json文件
with open('data/V2/MAMS/test_con_new.json', 'r') as f:
    data = json.load(f)

# 将数据转换为json格式字符串，并设置缩进量为4
json_str = json.dumps(data, indent=4)

# 将json字符串写入新文件
with open('data/V2/MAMS/test_con_new.json', 'w') as f:
    f.write(json_str)
