#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__author__ = 'WYou'
'''
input: PHEV maker from '08-Apr-2018_PHEV_calib_data_uv'
output:Types of PHEV maker
Created on July 27, 2018
'''
import numpy
import re

list = []
# path = '.\lt_fuel_stations_uv.xlsx'
with open('maker_PHEV.txt','r') as f:
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
types = [j for j in list]
t = set(types)
print(t)