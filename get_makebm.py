# -*- coding: utf-8 -*-
'''
抓每个brand下的basemodel list
使用的是json网址
导入get_make.py生成的get_make的列表
对应pachong8
'''
import requests
import re
from urllib.parse import urlencode
from urllib.request import Request, urlopen
import json
import time
import csv

# __ file name ___________________________________________
filename = 'get_makebm'

# ________________________________________________________
headers = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/68.0.3440.106 Safari/537.36',
    'Host': 'www.fueleconomy.gov',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
    'Accept-Encoding': 'gzip, deflate, br',
    'Accept-Language': 'en-US,en;q=0.9,zh-CN;q=0.8,zh;q=0.7'
}

make_file = open('get_make.txt','r')
make_list = make_file.readlines()

start_time = time.time()

for make in make_list:
    make = make.strip()
    paras = {
        'make': make#, #'BMW', #brand
        #'baseModel': '3 Series', #base model .*?
       # 'srchtyp': 'ymm'
        }

    print(make)

    url = 'https://www.fueleconomy.gov/feg/Find.do?action=getMenuBaseModelRng&year1=2000&year2=2019&' + urlencode(paras) + '&format=json'

    html = requests.get(url, headers = headers)
    #print(url)
    

    content=requests.get(url, headers=headers).content
    #print(content)
    con = json.loads(content)
    makebmList = con["options"]
    for item in makebmList:
        text = item['text']
        value = item['value']
        #print('%s: %s' % (text, value))
    print('Base Model Loading Completed!\n')
    
    '''
    f = open("%s.csv" % filename, 'w') #'a':连续写入 ‘w’先清空再写入
    try:
        writer = csv.writer(f)
        writer.writerow(('make','base model'))
        for item in commenList:
            writer.writerow((make,item['value']))
            print('%s: %s' % (make, item['value']))
    finally:
        f.close()
    '''
    
    # 保存make brand和base model，一一对应
    f = open("%s.txt" % filename, 'a') #'a':连续写入 ‘w’先清空再写入
    for item in makebmList:
        f.write(str(make)+",")
        f.write(str(item['value'])+"\n")
    #f.write("\n")
    f.close()

elapsed_time = time.time() - start_time
print(elapsed_time)

print('Done')