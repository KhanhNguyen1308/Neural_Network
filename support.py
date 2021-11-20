import matplotlib.pyplot as plt
import numpy as np


def show_train_ds(train_ds, class_name):
    title = []
    plt.figure(figsize=(15, 15))
    for images, labels in train_ds.take(1):
        for j in labels:
            title.append(np.argmax(j))
        for i in range(20):
            ax = plt.subplot(5, 5, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            
            plt.title(title[i])
            plt.axis("off")


def coordinates(img, w, h, size):
    x = [0]
    y = [0]
    web_img = []
    range_x = int(w/8)
    range_y = int(h/8)
    for i in range(8):
        x_coor = int((w/size)*size/8*(i+1))
        if x_coor > w:
            x_coor = w
        x.append(x_coor)
    for i in range(8):
        y_coor = int((h/size)*size/8*(i+1))
        if y_coor > h:
            y_coor = h
        y.append(y_coor)
    for i in range(8):
        for j in range(8):
            web_img.append(img[y[i]:y[i+1], x[j]:x[j+1]])
    return x, y, range_x, range_y, web_img