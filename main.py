# 主运行程序
import get_pic
import network
import run_network

if __name__ == "__main__":
    # 标签种类数
    label_type = 5
    # 训练所需拍摄的照片数
    n = 100
    # 设定超参数
    batch_size, lr, epoch, ratio = 10, 0.01, 5, 0.5

    get_pic_or_not = input("Do you want to start with getting your hand pose data? Y/N ")
    if get_pic_or_not == 'Y':
        get_pic.pic_main(number=n)
    print()

    train_or_not = input("Do you want to directly start train the classifier? Y/N ")
    if train_or_not == 'Y':
        trained_net = network.network_main(net_used=network.Alex_build(label_type=label_type, ratio=ratio), batch_size=batch_size, lr=lr, epoch=epoch)
        save_or_not = input("Do you want to save the trained parameter of the network? Y/N ")
        if save_or_not == 'Y':
            network.show_network(trained_net)
            network.save_params(trained_net)
    print()
    
    estimation_or_not = input("Start estimation? Y/N ")
    if estimation_or_not == 'Y':
        trained_net = network.read_params()
        run_network.start_control(trained_net)


## 实别时的图片转换
###  测试集正确率不变: 数据集过小
### 训练时的第一层输入出现问题: 无问题，输入时应为四维nd
