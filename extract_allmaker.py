#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__author__ = 'WYou'
'''
input: All maker from 'all_cars'
output:maker brand
Created on Sep 17th, 2018
'''
import numpy
import re

list = []
# path = '.\lt_fuel_stations_uv.xlsx'
with open('allmake.txt','r') as f:
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