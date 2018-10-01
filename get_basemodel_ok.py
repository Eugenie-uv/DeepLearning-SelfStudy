# -*- coding: utf-8 -*-
'''
input：导入get_makebm.py生成的makebm列表，是make × base model，用来构造url
output：抓取每个make下每个base model下的所有车型

对应pachong9的完整版
get_basemodel + get_basemodel_add + 3页pages都提取出来
---------------------------------------
Created on Sep 21, 2018
@author: Eunice_u
'''
import requests
import time
import csv
import re
from urllib.parse import urlencode

# __ file name to save________________________________________________________
filename = 'get_basemodel_ok'
num_pages = 3
# __________________________________________________________________________
headers = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/68.0.3440.106 Safari/537.36',
    'Host': 'www.fueleconomy.gov',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
    'Accept-Encoding': 'gzip, deflate, br',
    'Accept-Language': 'en-US,en;q=0.9,zh-CN;q=0.8,zh;q=0.7'
}

start_time = time.time()

file = open('get_makebm.txt','r') #makebm_test.txt','r')
makebm_list = file.readlines()
file.close()

#import numpy as np
#id_list = np.loadtxt('EVid.txt', dtype='int')
#id_list = np.loadtxt('PHEVid.txt', dtype='int')
num = len(makebm_list)
print('Number of base models of all makes:', num)
'''
print(makebm_list)
print('\n[0]:')
print(makebm_list[0])
print('\n[1]:')
print(makebm_list[1])

print('\nmakebm_list start:')
for makebm_ in makebm_list:
    print(makebm_)
print('makebm_list done!\n')
'''
for i in range(num):
    print('\nNo. make-basemodel = %d/%d:' % (i+1, num))
    make, basemodel = makebm_list[i].split(',', 1)
    #print('Make & Base Model: ', (make, basemodel))
    make = make.strip()
    basemodel = basemodel.strip()
    print('Current lodaing of Make - Base Model: ', (make, basemodel))
    
    paras = {
        'make': make, #'BMW', #brand
        'baseModel': basemodel, #'3 Series', #base model .*?
        'srchtyp': 'ymm'
        }

    # 加入页数循环：
    
    for j in range(num_pages):
        print('Page No.:', j)
        url = 'https://www.fueleconomy.gov/feg/PowerSearch.do?action=noform&path=1&year1=2000&year2=2019&' + urlencode(paras) + '&pageno={}&sortBy=Comb&tabView=0&rowLimit=200'.format(j+1)
        #print(url)

        html = requests.get(url, headers = headers)

# 正则表达式进行解析
#    pattern1 = re.compile('<title>(.*?)</title>', re.S)
        pattern2 = re.compile(r'<a href="Find.do\?action=sbs\&amp;id=(.*?)</a>', re.S)
        pattern3 =  re.compile(r'<a id=\"btnSbsSelect\" href=\"Find\.do\?action=sbsSelect\&amp;id=(\d+)\">Add a Vehicle</a>', re.S)
    
    #titles = re.findall(pattern1, html.text)
    #for title in titles:
    #    print(title)
    
    # 提取每个车型及对应的EPA_id
        items = re.findall(pattern2, html.text)
        items = [item.replace('">', ',') for item in items]
        for each in items:
            print(each)
        print('Loaded how many of pattern2:', len(items))

        items3 = re.findall(pattern3, html.text)
        for each in items3:
            print(each)
        print('Loaded how many of pattern3:', len(items3))
    
        f = open("%s.csv" % filename, 'a') #'a':连续写入(注意要运行时先删除旧文件) ‘w’先清空再写入
        try:
            writer = csv.writer(f)
        #writer.writerow(('baseModel', 'id', 'car model'))
            for each in items:
                id, car = each.split(',', 1) #只分割一次，把id提出来，因为车型号长短不一，不设置分割次数会出错
                writer.writerow((basemodel, id, car))
            for each in items3:
                writer.writerow((basemodel, each))
        finally:
            f.close()
        
'''# txt格式文本
    f = open("%s.txt" % filename, 'w') #'a':连续写入 ‘w’先清空再写入
    #f.write("\n")
    #f.write(str(items))#+"\n")
    for each in items:
        id, car = each.split(',', 1)
        f.write(str(basemodel)+",")
        f.write(str(id)+",")
        f.write(str(car)+",\n")
    f.close()
'''

elapsed_time = time.time() - start_time
print('\nCost time:', elapsed_time)

print('Done')