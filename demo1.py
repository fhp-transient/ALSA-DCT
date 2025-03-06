# -*- coding: utf-8 -*-
# @Time : 2023/5/19 20:47
# @Author : fhp
# @File : demo1.py
# @Software : PyCharm
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("pretrained/bert-base-uncased")
text = "we all liked the brunch items we had but when we later asked if the chips were supposed to " \
       "come with the meal, the waitress explained it away by saying sometimes they forget instead " \
       "of apologizing and offering something extra."

tokens = tokenizer.tokenize(text)
print(tokens)
