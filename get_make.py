# -*- coding: utf-8 -*-
'''
抓取网页中的make列表，因为发现名称和database里的不太一样
使用的是json网址
https://www.fueleconomy.gov/feg/Find.do?action=getMenuMakeRng&year1=2000&year2=2019&format=json

'''
import requests
import re
from urllib.parse import urlencode
from urllib.request import Request, urlopen
import json
import csv

# __ file name ___________________________________________
filename = 'get_make'

# ________________________________________________________
headers = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/68.0.3440.106 Safari/537.36',
    'Host': 'www.fueleconomy.gov',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
    'Accept-Encoding': 'gzip, deflate, br',
    'Accept-Language': 'en-US,en;q=0.9,zh-CN;q=0.8,zh;q=0.7'
}

url = 'https://www.fueleconomy.gov/feg/Find.do?action=getMenuMakeRng&year1=2000&year2=2019&format=json'

html = requests.get(url, headers = headers)

content=requests.get(url, headers=headers).content
    #print(content)
con = json.loads(content)
makeList = con["options"]
for item in makeList:
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
for item in makeList:
    f.write(str(item['value'])+"\n")
    #f.write("\n")
f.close()
    
print('Done')
