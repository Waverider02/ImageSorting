import utils
import logging
from torch import optim
from matplotlib import pyplot as plt
from torch.nn import functional as F
import json
import sys
import os

if __name__ == "__main__":
    # 创建目录
    if not os.path.exists("./figure"):
        os.makedirs("./figure")

    if not os.path.exists("./data"):
        os.makedirs("./data")

    # 配置控制台日志
    logging.basicConfig( 
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # 获取训练集与测试集(第一次需联网下载)
    logging.info("getting train_data, test_data")
    batch_size = 128 # 每一组包含128张图像
    train_data,test_data = utils.get_data() # 图像数据取自MINIST数据集

    # train_data.dataset  训练集包含60000张图像
    # test_data.dataset   测试集包含10000张图像

    """
    b: batch_size
    图像数据 x:torch.Size([b,1,28,28]) 
    图像结果 y:torch.Size([b])
    """

    # 取样, 展示训练集数据
    x,y = next(iter(train_data))
    utils.plot_imag(x,y,'image sample')
    plt.savefig("./figure/figure1.png")

    # 神经网络与优化器实例化
    # 神经网络: [b,1,28,28]->[b,10]
    net = utils.Net() # 默认使用全连接神经网络
    for lp,arg in enumerate(sys.argv):
        if arg == "-m":
            if sys.argv[lp+1] == "Net":
                net = utils.Net()
                logging.info("Net is used")
            elif sys.argv[lp+1] == "CNN":
                net = utils.CNN()
                logging.info("CNN is used")
            else:
                logging.info("Net is used")

    optimizer = optim.SGD(net.parameters(),lr=0.01,momentum=0.9)
    train_loss = []

    # 迭代训练
    logging.info("start training")
    for epoch in range(10): # 训练10轮, 即使用10倍的数据集
        for batch_idx,(x,y) in enumerate(train_data):
            out = net(x) # [b,1,28,28]->[b,10]
            y_onehot = utils.one_hot(y)
            loss = F.mse_loss(out,y_onehot)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
            if batch_idx % 10 == 0:
                msg = utils.to_msg('epoch:',epoch,'  batch_idx:',batch_idx,'  loss:',loss.item())
                logging.info(msg)
    jsondata = dict()
    jsondata.update({"loss":train_loss})
    plt.figure()
    plt.plot(train_loss)
    plt.xlabel("STEP")
    plt.ylabel("MSE LOSS")
    plt.savefig("./figure/figure2.png")

    # 利用测试集对训练后的网络进行评估
    total_correct = 0
    for x,y in test_data:
        out = net(x) # [b,1,28,28]->[b,10]
        pred = out.argmax(dim=1) # 预测值
        correct = pred.eq(y).sum().float().item() # 计数预测正确的数目
        total_correct += correct
    total_num = len(test_data.dataset) # 获取图像总数
    acc = total_correct/total_num # 准确率 = 预测正确图像数目 / 图像总数
    jsondata.update({"acc":acc})
    with open("./data/data.json",'w') as f:
        json.dump(jsondata,f)
    msg = utils.to_msg('test acc:',acc)
    logging.info(msg)

    # 取样, 展示预测结果
    x,y = next(iter(test_data))
    out = net(x)
    pred = out.argmax(dim=1)
    utils.plot_imag(x,pred,'image pred')
    plt.savefig("./figure/figure3.png")