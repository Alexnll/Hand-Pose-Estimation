# 导入训练完成的神经网络模型
# 控制摄像头捕获手势
# 将捕获得到的手势导入神经网络，得到输出
# 绑定相应的操作(初设为绑定四个操作，对应的标签为：a, s, d, w)

import time
import cv2
import numpy as np
import mxnet
from mxnet.gluon import nn
from mxnet import nd


# 映射键盘
def control(result):
    print(result)
    if result == 'a':
        print('a')
    elif result == 's':
        print('s')
    elif result == 'd':
        print('d')
    elif result == 'w':
        print('w')
    elif result == 'o':
        pass


# 开始控制
def start_control(net):
    print("start estimation")
    time.sleep(0.5)
    cap = cv2.VideoCapture(0)

    # 主循环
    while True:
        ret, frame = cap.read()
        cv2.imshow('capture', frame)

        # 退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("exit")
            break
        else:
            # read_pic =  cv2.imread(frame)[120:360, 200:400]
            # 图片处理，将cv2读到的图片处理为mxnet网络的形式
            # treated_pic = read_pic
            continue
            # result = net(treated_pic)
            # control(result)
        
    cap.release()
    cv2.destroyAllWindows()
    print("Estimation End.")


