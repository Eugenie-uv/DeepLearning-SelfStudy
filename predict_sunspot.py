# -*- coding: utf-8 -*-
'''
预测太阳黑子
@author:github.com/Euniceu
'''
from reportlab.graphics.shapes import * #Drawing, String
from reportlab.graphics import renderPDF
from reportlab.lib import colors
from reportlab.graphics.charts.lineplots import LinePlot
from reportlab.graphics.charts.textlabels import Label

filename = 'data.txt'
Comment_chars = '#:'

#f = open('data.txt')
#f.read()
data = []
'''    #year month predicted high low
(2017, 6,        14.0,    15.0,    13.0),
(2017, 7,        14.4,    16.4,    12.4),
(2017, 8,        14.6,    17.6,    11.6),
(2017, 9,        14.9,    19.9,     9.9),
(2017, 10,        15.0,    20.0,    10.0),
(2017, 11,        14.9,    20.9,     8.9),
(2017, 12,        15.2,    22.2,     8.2),
(2018, 1,        15.3,    22.3,     8.3),
(2018, 2,        15.1,    23.1,     7.1),
(2018, 3,        14.1,    23.1,     5.1),
(2018, 4,        13.7,    22.7,     4.7),
(2018, 5,        14.1,    24.1,     4.1),
(2018, 6,        13.9,    23.9,     3.9),
(2018, 7,        12.9,    22.9,     2.9),
(2018, 8,        12.2,    22.2,     2.2),
(2018, 9,        11.5,    21.5,     1.5),
(2018, 10,        10.8,    20.8,     0.8)
]
'''
for line in open(filename).readlines():
    if not line.isspace() and not line[0] in Comment_chars:
        data.append([float(n) for n in line.split()])

drawing = Drawing(400, 200)

pred = [row[2] for row in data]
high = [row[3] for row in data]
low = [row[4] for row in data]
times = [((row[0]+row[1]/12.0)) for row in data]

lp = LinePlot()
lp.x = 50
lp.y = 50
lp.height = 125
lp.width = 300
lp.data=[list(zip(times, pred)), list(zip(times, high)), list(zip(times,low))]
lp.lines[0].strokeColor = colors.blue
lp.lines[1].strokeColor = colors.red
lp.lines[2].strokeColor = colors.green

#drawing.add(PolyLine(list(zip(times,pred)), strokeColor=colors.blue))
#drawing.add(PolyLine(list(zip(times,high)), strokeColor=colors.red))
#drawing.add(PolyLine(list(zip(times,low)), strokeColor=colors.green))

drawing.add(lp)
drawing.add(String(250, 150,'Sunspots',fontSize=18, fillColor=colors.red))

renderPDF.drawToFile(drawing, 'Sunspots report.pdf', 'Sunsports')