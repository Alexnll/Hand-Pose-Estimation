# 从摄像头出采集手势照片，用作训练集与测试集
# 基于opencv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import sys, os, re
import cv2

# 清空特定文件夹内的文件
def clear_dir(path='.\dataset\HAND POSE'):
    print("start trying to clean the path")
    time.sleep(1)
    files = os.listdir(path) # 得到文件夹下所有文件的名称
    if len(files) == 0:
        print("dir already cleaned")
        time.sleep(2)
    else:
        for fi in files:
            os.remove(path + '\\' + fi)
        files = os.listdir(path)
        if len(files) == 0:
            print("successfully clean the dir")
            time.sleep(2)        

# 控制摄像头捕获视频
# 图片默认存放路径： .\dataset\HAND POSE
def control_camera(pic_number=10, path='.\dataset\HAND POSE'):
    n = 0    # 记录以拍摄的照片
    n1 = 0
    cap1 = cv2.VideoCapture(0)    # 用于监控摄像头
    # 清空目标目录
    clear_dir(path)
    path = '.\dataset\HAND POSE' + '\\'
    print("start record picture")

    while(True):
        # 当储存文件的数量达到所需的数量时，退出循环
        if n >= pic_number:
            print("reach the picture number")
            print("ending the process")
            time.sleep(1.5)
            break
        
        # 判断循环开始
        if n > n1:
            n1 = n
            print("Start record new picture")

        # capture frame-by-frame
        ret, frame = cap1.read()
        # 绘制输入框
        cv2.rectangle(frame, (200, 120),(440, 360), (255, 0, 0),3)  # 确定左上点， 右下点的宽，高以及线宽
        # Display the resulting frame
        cv2.imshow('capture', frame)
        
        # press q to quit the while loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("exit")
            time.sleep(1)
            break
        # when the number of photo larger than the setted value

        # 键入相应的键盘按键时，从摄像头中保存相应的图片
        if cv2.waitKey(1) & 0xFF == ord('a'):
            n += 1
            cv2.imwrite(path + str(n) + '-a.jpg', frame)
            print('saving image: ' + str(n) + '-a.jpg')
            print("n: ", n)
            print()
            continue
        
        if cv2.waitKey(1) & 0xFF == ord('s'):
            n += 1
            cv2.imwrite(path + str(n) + '-s.jpg', frame)
            print('saving image: ' + str(n) + '-s.jpg')
            print("n: ", n)
            print()
            continue

        if cv2.waitKey(1) & 0xFF == ord('d'):
            n += 1
            cv2.imwrite(path + str(n) + '-d.jpg', frame)
            print('saving image: ' + str(n) + '-d.jpg')
            print("n: ", n)
            print()
            continue

        if cv2.waitKey(1) & 0xFF == ord('w'):
            n += 1
            cv2.imwrite(path + str(n) + '-w.jpg', frame)
            print('saving image: ' + str(n) + '-d.jpg')
            print("n: ", n)
            print()
            continue

        if cv2.waitKey(1) & 0xFF == ord('o'):
            n += 1
            cv2.imwrite(path + str(n) + '-o.jpg', frame)
            print('saving image: ' + str(n) + '-o.jpg')
            print("n: ", n)
            print()
            continue        
    
    cap1.release()
    cv2.destroyAllWindows()
    print("The final picture number: ", n)
    print(" ")
    time.sleep(1)
 
# 图片前处理， 同时读取出每张图片的标签并储存
# 获取图像ROI，尺寸初设为240*240，在进行进一步缩小至120*120
# 对图片重命名，同时将图片编号与label保存并导出到csv文件中
def pic_preprocess(path='.\dataset\HAND POSE'):
    files = os.listdir(path) # 得到文件夹下所有文件的名称
    # 若无照片文件存在
    if len(files) == 0:
        print("no picture exists")
        return 0
    else:
        print("start the picture preprocessing in " + path)
        time.sleep(1)
        label = []
        column_name = ['label_id','pic_label']

        for fi in files:
            if (re.match('.*?\.(\w+)', fi).group(1)) == 'jpg':
                print("treating " + fi)
                # 分割文件文件名，并将其储存到label_id和label_set中
                var = re.match('(\d+)-(a|s|d|w|o).jpg', fi)
                label.append([int(var.group(1)), var.group(2)])

                # 对图片进行尺寸处理
                current_file = path + '\\' + fi
                img = cv2.imread(current_file)
                if img.size >= 100000:   # 确认图片是否在处理前或处理后
                    print("Before treatment: ", img.shape, img.size, img.dtype) # 获取图像属性  
                    img = img[120:360, 200:440]   # 获取图像ROI
                    img = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)   # 图像缩放，采用设置缩放因子的形式
                    print("After treatment: ", img.shape, img.size, img.dtype)
                    cv2.imwrite(current_file, img)

                    # 单张照片处理结束
                    print("finish treating " + fi)
                    print()
                else:
                    # 跳过
                    print("skip " + fi)
                    print()

                # break # 用于调试

        # 存入到labels中并导出为csv文件
        labels = pd.DataFrame(data=label, columns=column_name)
        labels = labels.sort_values(by = 'label_id', ascending=True)
        # print(labels)
        labels.to_csv(path + '\\' + 'labels.csv', encoding='gbk')
        return labels

# 收集得到的图片进行重命名
def try_rename(path='.\dataset\HAND POSE'):
    files = os.listdir(path) # 得到文件夹下所有文件的名称
    # 若无照片文件存在
    if len(files) == 0:
        print("no picture exists")
        return 0
    else:
        print("rename all pic files")
        time.sleep(0.5)
        for fi in files:
            if (re.match('.*?\.(\w+)', fi).group(1)) == 'jpg':
                var = re.match('(\d+)-(a|s|d|w|o).jpg', fi)
                os.rename(path+ '\\' +fi, path+ '\\' +var.group(1) + '.jpg')


# get_pic主程序
def pic_main(number=100):
    control_camera(pic_number=number)
    print("picture capture finish.")
    print()
    time.sleep(0.5)
    pic_preprocess()
    print("picture preprocess finish.")
    print()
    time.sleep(0.5)
    try_rename()
    print("picture rename finish.")
    print()
    time.sleep(0.5)

if __name__ == "__main__":
    n = 50
    # control_camera(pic_number=n)
    # pic_preprocess()
    # try_rename()
    pic_main(number=n)


'''
    # opencv库基本摄像头操作
    cap = cv2.VideoCapture(0)   # 创建摄像头对象
    print(type(cap))
    # 逐帧显示视频播放
    while(True):
        # 利用read()函数读取视频的某帧
        ret, frame = cap.read()
        # 展示
        cv2.imshow('capture', frame)
        # 若检测到键盘键入q，则退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            # 释放摄像头对象和窗口
            cap.release()
            cv2.destroyAllWindows()
            break
'''