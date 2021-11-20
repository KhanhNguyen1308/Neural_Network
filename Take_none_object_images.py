import os
import cv2
import pathlib
import numpy as np
from support import show_train_ds, coordinates

xs = []
ys = []
ws = []
hs = []
names = []
count = 0
count_images = 0

folder = "data/"
data_dir = pathlib.Path(folder)
image_count = len(list(data_dir.glob('*.jpg')))
old_percent = 1
for i in os.listdir(folder):
  count_images+=1
  if i != "classes.txt" and i.endswith(".txt"):
    datas = open(folder+i, "r").readlines()
    name = i.replace("txt", "jpg")
    img = cv2.imread(folder+name)
    img_ = img.copy()
    h, w = img.shape[:2]
    x_coor, y_coor, range_x, range_y, web_img = coordinates(img_, w, h, size=416)
    web_img = web_img
    num_ob = 0    
    slide = []
    for data in datas:
          data = data.replace("\n", "")
          data = data.split(" ")
          if data[0] == '0':
            print(data)
            x1 = int(float(data[1])*w-float(data[3])*w/2)
            x2 = int(float(data[1])*w+float(data[3])*w/2)
            y1 = int(float(data[2])*h-float(data[4])*h/2)
            y2 = int(float(data[2])*h+float(data[4])*h/2)
            for start in range(8):
              if (x_coor[start]) <= x1 <= (x_coor[start]+range_x):
                x_col = start
                for end in range(8):
                  if (x_coor[end]) <= x2 <= (x_coor[end]+range_x):
                    num_x = end - start
                    for start_y in range(8):
                      if (y_coor[start_y]) <= y1 <= (y_coor[start_y]+range_y):
                        for end_y in range(8):
                          if (y_coor[end_y]) <= y2 <= (y_coor[end_y]+range_y):
                            num_y = end_y - start_y
                            for y_slide in range(num_y+1):
                              for x_slide in range(num_x+1):
                                coor = (start_y+y_slide)*8+start+x_slide
                                slide.append(coor)

    print(name)
    for i in range(len(web_img)):
          img = web_img[i]
          img = np.array(img)
          if i not in slide:
            cv2.imwrite("/content/drive/MyDrive/Object_Detection/data/nface/none"+str(count)+".jpg", img)
            count+=1    
