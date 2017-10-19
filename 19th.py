# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 15:14:01 2017
解读十九大
@author: admin
"""

# jieba分词
# 进行中文分词的利器
# ==============================================
import jieba
from jieba.analyse import extract_tags

with open('19th.txt', 'r', encoding='utf-8', errors='ignore') as f:
    data = f.read()
fenci = list(jieba.cut(data))
text = extract_tags(data, topK=1000, withWeight=False) #找最大的值
print('____________________________')
#print('前1000个最大的：', text)


# Counter计数器进行排序
# ==============================================
from collections import Counter
max_number = 500
cout = Counter(fenci)#text)
cout_n = cout.most_common(max_number) 
#print(test)
#print(cout)
print('____________________________')
print('频率前500的词 : ',cout_n)

# 去除符号和助词、介词等
most_words = [words for words in cout_n if words[0] not in ' ，、。“”（）！；的和是在要为以把了对中到有上不等更二从大\n']

# wordcloud词云
# ===============================================
from wordcloud import WordCloud

#text2 = open('AESOPS FABLES.TXT','r').read()
# from scipy.misc import imread
# bg_pic = imread('party.png')
wordcloud = WordCloud(
            font_path='lixushufa.ttf',
            background_color='white',
            max_words=200,
            mask=None,
            max_font_size=100)
#wc = wordcloud.generate(text)
#wc2 = WordCloud().generate(text2)
wc = wordcloud.generate_from_frequencies(dict(most_words))
wc.to_file('19thnew.png')

import matplotlib.pyplot as plt

plt.figure()
plt.imshow(wc)
#plt.imshow(wordcloud)
plt.axis('off')
plt.show()

# ECharts 给数据美颜 绘制条状图？
# ===============================================
# words_list = []
# count_list =[]
# for word in most_words[:32]:
    # words_list.append(word[0])
    # count_list.append(word[1]) #出现的权重




