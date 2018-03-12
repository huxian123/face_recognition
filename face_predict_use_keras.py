#encoding=utf-8
__author__ = 'Administrator'

import cv2
import sys
import gc
from face_train_use_keras import Model

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage:%s camera_id\r\n" % sys.argv[0])
        sys.exit(0)


    #加载模型
    model = Model()
    model.load_model(file_path='./model/me.face.model.h5')

    #框住人脸的矩形边框颜色
    color = (0, 255, 0)

    #捕获指定摄像头的实时视频流
    cap = cv2.VideoCapture(int(sys.argv[1]))

     #告诉OpenCV使用人脸识别分类器
    cascade_path = "D:\Program Files\opecv-3.2\opencv\sources\data\haarcascades\haarcascade_frontalface_alt2.xml"

    #循环检测人脸
    while True:
        _, frame = cap.read()

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        cascade = cv2.CascadeClassifier(cascade_path)

        faceRects = cascade.detectMultiScale(frame_gray, scaleFactor=1.2, minNeighbors=3, minSize=(32,32))
        if len(faceRects) > 0:
            for faceRect in faceRects:
                x, y, h, w = faceRect

                image = frame[y - 10:y + h + 10, x - 10:x + w + 10]
                faceID = model.face_predict(image)

                if faceID == 0:
                    cv2.rectangle(frame, (x-10, y-10), (x+w+10, y+h+10), color, thickness=2)

                    #文字提示
                    cv2.putText(frame, "YUJI",
                                (x+30, y+30),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1,
                                (255,0,255),
                                2)
                else:
                    cv2.rectangle(frame, (x-10, y-10), (x+w+5, y+h+5), color, thickness=1)
                    #文字提示
                    cv2.putText(frame, "UNKNOWN",
                                (x+30, y+30),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1,
                                (255,0,255),
                                2)
        cv2.imshow("识别",frame)

        k = cv2.waitKey(10)

        if k & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()