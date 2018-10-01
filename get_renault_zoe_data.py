# -*- coding: utf-8 -*-
'''
input：https://group.renault.com/RCW_BINARIES/ZE_Simulator/autonomy.php?country=france&locale=en
output：每辆车下的speed,temperature,wheels,ac,heater,eco下的battery，engine，distance_range，unit

---------------------------------------
Guillaume Dionisi帮我写的，每一句都注释了！！！
Created on Sep 27, 2018
@author: GD, GitHub/Eunice_u
'''


import requests # library to load URL's html
from pyquery import PyQuery # library to parse html PyQuery是iQuery的Python实现，用于解析HTML网页
from slimit.parser import Parser # library to parse javascript
from slimit.visitors import nodevisitor # helper to parse tree got from parsing javascript
from slimit import ast # helper to compare slimit classes
import csv

# __ file name to save________________________________________________________
filename = 'get_renault'

# __________________________________________________________________________

url = "https://group.renault.com/RCW_BINARIES/ZE_Simulator/autonomy.php?country=france&locale=en" # url to load
html = requests.get(url).content # get the html code from the URL

pq = PyQuery(html) # load the URL into a Python object that can get specific data
# 根据html标签直接定位：
# 获取<script>......</script>标签内的内容
scripts = pq("script") # get all the <script> tags
javascript = scripts[-1].text # get the content of the last script tag: the last one contains all the json data

#print(javascript)

parser = Parser() # parser object to parse Javascript
#只支持ply3.4？反正先把高版本的卸载了再装上3.4就不出错了：
#pip uninstall ply
#pip install ply==3.4

tree = parser.parse(javascript) # tree containing the Javascript elements as nodes
nodes = nodevisitor.visit(tree) # get the javascript elements

for node in nodes: # look for the jsonData attribute, which is the first array
    if isinstance(node, ast.Array): # first array is the data we want
        jsonData = node # get the data as a slimit array
        break

false = False # booleans in Javascript are written without uppercase
true = True  # booleans in Javascript are written without uppercase

items = [] # array to hold all the data
for node_item in jsonData.items: # for each item in the data array
    item = {} # dictionary to hold a single item
    for prop in node_item.properties: # for each property of this item
        field = eval(prop.left.value) # fields are written as strings, so prop.left.value is a string containing a string, which thus needs to be evaluated
        if field == "ranges": # ranges field is a specific case because its value is a slimit Array
            value = [] # array to hold the ranges
            ranges_array = prop.right # slimit Array holding the ranges data
            for range_item in ranges_array.items: # for each range item
                single_range = {} # JSON dictionary to hold a single range data
                for range_prop in range_item.properties: # for each property of a range
                    range_field = eval(range_prop.left.value) # fields are written as strings, so prop.left.value is a string containing a string, which thus needs to be evaluated
                    #print(range_field)
                    
                    range_value = eval(range_prop.right.value) # fields are written as strings, so prop.right.value is a string containing a string, which thus needs to be evaluated
                    #print(range_value)
                    
                    single_range[range_field] = range_value
                value.append(single_range) # add single range to list of ranges
                #print(value)
                
        else: # if field is not ranges, simply get the value
            value = eval(prop.right.value) # fields are written as strings, so prop.right.value is a string containing a string, which thus needs to be evaluated
        item[field] = value # set the value for the corresponding field in the item dictionary
        #print(item[field])
        
    items.append(item) # add the single item to the list holding all the data
#print(items[0])
#print(type(items)) #items的类型是list

f = open("%s.csv" % filename, 'w') #'a':连续写入(注意要运行时先删除旧文件) ‘w’先清空再写入
'''
try:
    writer = csv.writer(f)
    #writer.writerow(('baseModel', 'id', 'car model'))
    for each in items:
        #print(type(each))#each的type是字典哦！
        #default, speed, temperature, wheels, ac, heater, eco, ranges = each.split(',')#, 1)
        #writer.writerow((default, speed, temp, wheels, ac, heater, eco, ranges))
        #writer.writerows(each.items()) #将每个键/值对写入一个单独的行
        writer.writerow(each.keys())
        writer.writerow(each.values())#一行上的所有键和下一行上的所有值
finally:
    f.close()
'''
try:
    writer = csv.writer(f)
    writer.writerow(('default', 'speed', 'temperature', 'wheels', 'ac', 'heater', 'eco',
    'battery_1', 'engine_1', 'distance_range_1', 'unit_1',
    'battery_2', 'engine_2', 'distance_range_2', 'unit_2'))
    for each in items:
        #writer.writerow(each.keys())
        #print(type(each))#each的type是字典哦！
        #for info in each['ranges']:
            #print(type(each['ranges'])) #是list形式
            #print(type(info)) #info 是字典形式！
        info_1 = each['ranges'][0]
        info_2 = each['ranges'][1]
        writer.writerow((each['default'], each['speed'], each['temperature'], each['wheels'], each['ac'], each['heater'], each['eco'], 
        info_1['battery'], info_1['engine'], info_1['distance_range'], info_1['unit'], 
        info_2['battery'], info_2['engine'], info_2['distance_range'], info_2['unit']))
        '''for i in range(len(each['ranges'])):
            print('The %d one of cars is processing. ' % (i+1))
            info_1 = each['ranges'][0]
            info_2 = each['ranges'][1]
         #   print(info)
          #  print('ok')
            writer.writerow((each['default'], each['speed'], each['temperature'], each['wheels'], each['ac'], each['heater'], each['eco'], 
            info_1['battery'], info_1['engine'], info_1['distance_range'], info_1['unit'], 
            info_2['battery'], info_2['engine'], info_2['distance_range'], info_2['unit']))
            #writer.writerow(each.values())
            #writer.writerow(info.values())
    #print(info[0])'''
finally:
    f.close()
 
'''
title = ['default', 'speed', 'temp', 'wheels', 'ac', 'heater', 'eco', 'ranges']
print(",".join(title))

for each in items:
    for info in each:
        print(",".join([info[field] for field in title]))
'''