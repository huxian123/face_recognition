#encoding=utf-8
__author__ = 'Administrator'

import cv2
import sys
from PIL import  Image

def CatchPICFromVideo(window_name, camera_idx, catch_pic_num, path_name):
    #视频来源，可以自己已保存好的视频，也可以直接来自usb摄像头
    cv2.namedWindow(window_name)

    cap2 = cv2.VideoCapture(camera_idx)

    #告诉OpenCV使用人脸识别分类器
    clasfier = cv2.CascadeClassifier("D:\Program Files\opecv-3.2\opencv\sources\data\haarcascades\haarcascade_frontalface_alt2.xml")

    #识别出人脸后要画出的边框的颜色，RGB格式
    color = (0, 255, 0)
    num = 0
    while cap2.isOpened():
        ok, frame = cap2.read() #读取一帧数据
        if not ok:
            break

        #将当前帧转为灰色图像
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        #人脸检测， 1.2 和2 为图片缩放比例和需要检测的有效点数
        faceRects = clasfier.detectMultiScale(grey, scaleFactor=1.2, minNeighbors=2, minSize=(32,32))
        if len(faceRects) > 0:
            for faceRect in faceRects: #大于0则检测到人脸
                x, y, h, w = faceRect  #单独框处人脸

                #将当前帧保存为图片
                img_name = '%s/%d.jpg' % (path_name, num)
                image = frame[y - 10: y + h + 10, x - 10: x + w + 10]
                cv2.imwrite(img_name, image)

                num += 1
                if num > (catch_pic_num):
                    break

                cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), color, 2)

                #显示当前捕捉到了多少人脸图片了，这样站在那里被拍摄时心里有个数，不用两眼一抹黑傻等着
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(frame,'num:%d' % (num),(x + 30, y + 30), font, 1, (255,0,255),4)

        if num > catch_pic_num:break

        cv2.imshow(window_name, frame)
        c = cv2.waitKey(10)
        if c & 0xFF == ord('q'):
            break

    cap2.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
     if len(sys.argv) != 4:
         print("Usage:%s camera_id face_num_max path_name\r\n" % (sys.argv[0]))
     else:
         CatchPICFromVideo("截取人脸", int(sys.argv[1]), int(sys.argv[2]), sys.argv[3])

