import os
import numpy as np      
from PIL import Image
#import py_midicsv as pm
from img2midi import *
from midi2img import *
path = 'D:\projects\music generation\data'
os.chdir(path)
midiz = os.listdir()
midis = []
for midi in midiz:
    midis.append(path+'\\'+midi)

new_dir = 'D:\projects\music generation\data'
count = 0
for midi in midis:
    try:
        count+=1
        print(count)
        midi2image(midi)
        basewidth = 106
        img_path = midi.split('\\')[-1].replace(".mid",".png")
        img_path = new_dir+"\\"+img_path
        
        img = Image.open(img_path)
        hsize = 106
        img = img.resize((basewidth,hsize), Image.ANTIALIAS)
        img.save(img_path)
        print(img_path)
    except:
        pass