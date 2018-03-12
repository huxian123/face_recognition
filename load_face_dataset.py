#encoding=utf-8
__author__ = 'Administrator'

import os
import numpy as np
import sys
import cv2

IMAGE_SIZE = 64
#按照指定图像大小调整尺寸
def resize_image(image, height=IMAGE_SIZE, width=IMAGE_SIZE):
    top, buttom, left, right = (0, 0, 0, 0)
    #获取图像尺寸
    h, w, _ = image.shape

    #对于长宽不相等的图片，找到最长的一边
    longest_edge = max(h, w)

    #计算短边需要增加多上像素宽度使其与长边等长
    if h < longest_edge:
        dh = longest_edge - h
        top = dh // 2
        buttom = dh - top
    elif w < longest_edge:
        dw = longest_edge - w
        left = dw // 2
        right = dw - left
    else:
        pass

    #RGB颜色
    BLACK = [0, 0, 0]

    constant = cv2.copyMakeBorder(image, top, buttom, left, right, cv2.BORDER_CONSTANT, value=BLACK)

    return cv2.resize(constant, (height, width))


#读取训练数据
images = []
labels = []
def read_path(path_name):
    #print(os.listdir(path_name))
    for dir_item in os.listdir(path_name):
        #从初始路径开始叠加，合并成可识别的操作路径
        full_path = os.path.relpath(os.path.join(path_name, dir_item))
        print(full_path)

        if os.path.isdir(full_path):#如果是文件夹，继续递归调用
            read_path(full_path)
        else:
            if dir_item.endswith('.jpg'):

                image = cv2.imread(full_path)
                if image is None:
                    print("照片为空")
                else:
                    image = resize_image(image, IMAGE_SIZE, IMAGE_SIZE)

                #放开这个代码，可以看到resize_image()函数的实际调用效果
                #cv2.imwrite('1.jpg', image)

                images.append(image)
                labels.append(path_name)

    return images, labels

#read_path('./data')

def load_dataset(path_name):
    images, labels = read_path(path_name)

    #将输入的所有图片变成4维数组，尺寸为[图片数量*IMAGE_SIZE*IMAGE）SIZE*3]
    #我和闺女两个人共2000张图片，IMAGE_SIZE为64，故对我来说尺寸为2000 * 64 * 64 * 3
    #图片是64*64像素，一个像素3RGB
    images = np.array(images)
    print(images.shape)

    #标注数据,'me'文件夹下的都是我的图像，全部指定为0，另外一个文件夹全是其他人,全部为1
    labels = np.array([0 if label.endswith('me') else 1 for label in labels ])

    return images, labels

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("usage:%s path_name\r\n" % (sys.argv[0]))
    else:
        images, labels = load_dataset(sys.argv[1])


