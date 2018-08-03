#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__author__ = 'WYou'
'''
Extract EV Charging Types
input: EV Connector Types from 'alt_fuel_stations (Jul 26 2018)'
output:Types of EV Connector
_________________________
Created on July 27, 2018
'''
import numpy
import re

list = []
# path = '.\lt_fuel_stations_uv.xlsx'
with open('EV_ConnectorTypes.txt','r') as f:
    data = f.readlines()
    
    for line in data:
        # item = line.split(', ') #分隔开
        item = re.split(r'[ \n\t\r]',line)
        print(item)
        # number_float = map(float, item) #转为浮点数
        # list.append(item) 
        list.extend(item) 
        # print(item)

print('done lodaing')
# types = [x for x in list if list.count(x)==1]
# print(types)
# types = [x for j in list for x in j]
# print(types)
types = [j for j in list]
t = set(types)
print(t)