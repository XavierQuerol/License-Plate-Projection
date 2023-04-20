# -*- coding: utf-8 -*-
"""
Created on Tue May 10 17:34:56 2022

@author: xavid
"""

import cv2
import numpy as np
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import matplotlib.pyplot as plt
from skimage.draw import polygon
from imutils import paths



provinces2 = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂",
             "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "警", "学", "O"]
provinces = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X',
       'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'O', '$', '%','?']
alphabets = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W',
             'X', 'Y', 'Z', 'O']
ads = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X',
       'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'O']


def extract_text(text):
    
    text2 = ""
    lbl = text.split('/')[-1].rsplit('.', 1)[0].split('-')[-3]
    lbl = lbl.split("_")
    for j, element in enumerate(lbl):
        if j == 0:
            element = int(element)
            text2 += ads[int(element)]
        elif j == 1:
            text2 += alphabets[int(element)]
        else:
            text2 += ads[int(element)]
    return text2
        

def find_box(img_name):
    
    iname = img_name.rsplit('/', 1)[-1].rsplit('.', 1)[0].split('-')
    
    
    [rD, lD, lU, rU] = [[int(eel) for eel in el.split('&')] for el in iname[3].split('_')]

    #rU = [rD[0]+(lD[0]-lU[0]), lU[1]-(lD[1]-rD[1])]
    
    return [rD, lD, lU, rU]
    #return [[rD[1],rD[0]], [lD[1],lD[0]], [lU[1],lU[0]], [rU[1],rU[0]]]

img_name = r'C:\Users\xavid\Documents\uni\cursos\tfg\dataset\R'
background_name = r'C:\Users\xavid\Documents\uni\cursos\tfg\dataset\imatges_xavi3'
directori = r'C:\Users\xavid\Documents\uni\cursos\tfg\dataset\imatges_xavi4'
text = r'C:\Users\xavid\Documents\uni\cursos\tfg\dataset\imatges_xavi3\final_imp.txt'


#llegir nom totes les imatges que hi ha a la carpeta background name
#llegir mateix número d'imatges de img name
imgs_2 = [el for el in paths.list_images(img_name)]
imgs_background = [el for el in paths.list_images(background_name)][:150]


f = open(text, 'r')
"""
plt.imshow(background)
plt.show()
plt.imshow(img)
plt.show()
"""
textos = []
z = 0
t = 0
for i, background_name in enumerate(imgs_background):
    
    #img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    background = cv2.imread(background_name)
   
    shape = background.shape
    
    rows,cols,ch = background.shape
    
    line_text = f.readline().split("/")
    
    tex1 = ""
    
    for m, l in enumerate(line_text):
        
        img_name = imgs_2[t]
        img = cv2.imread(img_name)
        img_box = find_box(img_name)
        
        
        l2 = l.split(",")
        background_box2 = []
        for i in range(0,8,2):
            background_box2.append([int(float(l2[i])), int(float(l2[i+1]))])
        background_box = background_box2[2:] + background_box2[:2]
        pts1 = np.float32(img_box) 
        pts2 = np.float32(background_box)
        
        M = cv2.getPerspectiveTransform(pts1,pts2)    
        dst = cv2.warpPerspective(img,M,(cols,rows))
        
        dif = [int(pts2[2][0]-pts1[2][0]),int(pts2[2][1]-pts1[2][1])]
        
        gt = np.zeros([shape[0],shape[1],3], "uint8")
        gt2 = np.ones([shape[0],shape[1],3], "uint8")
        
        r = np.array(pts2[:,1])
        c = np.array(pts2[:,0])
        rr, cc = polygon(r, c)
        gt[rr, cc] = 1
        gt2 -= gt
        
        if m != 0:
            tex1 += "\n"
        
        tex1 += extract_text(img_name)
        
        x= sum([el[0] for el in background_box])/4 / shape[1]
        y= sum([el[1] for el in background_box])/4 / shape[0]
        w=( max([el[0] for el in background_box]) - min([el[0] for el in background_box])) / shape[1]
        h= (max([el[1] for el in background_box]) - min([el[1] for el in background_box])) / shape[0]
        
        tex1 += " " + str(x) + " " + str(y) + " " + str(w) + " " + str(h)
        
        
        
        t += 1
        
        new_image = gt*dst + gt2*background
        background = new_image
    
    im_name = directori + "/image_" + str(z) + ".jpg"
    cv2.imwrite(im_name,new_image)
    
    textos.append(tex1)
    z += 1
    
f.close()   



for i, t in enumerate(textos):
    
    f = open(directori + r"\image_" + str(i) + ".txt", 'w')
    
    f.write(t)
    
    
    
    f.close()

