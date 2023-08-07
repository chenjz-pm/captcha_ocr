import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import my_datasets
from model import mymodel

import os

epoch=10000000
batch_size=256
data_file="./dataset/train"

if __name__ == '__main__':
    os.system("start tensorboard --logdir=logs")
    
    train_datas=my_datasets.mydatasets(data_file)
    #test_data=my_datasets.mydatasets("./dataset/test")
    train_dataloader=DataLoader(train_datas,batch_size=batch_size,shuffle=True)
    #test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)
    #writer=SummaryWriter("D:\\logs")
    m=mymodel().cuda()

    loss_fn=nn.MultiLabelSoftMarginLoss().cuda()
    optimizer = torch.optim.Adam(m.parameters(), lr=0.001)
    w=SummaryWriter("logs")
    total_step=0

for j in range(epoch):
    for i,(imgs,targets) in enumerate(train_dataloader):
        imgs=imgs.cuda()
        targets=targets.cuda()
        # print(imgs.shape)
        # print(targets.shape)
        outputs=m(imgs)
        # print(outputs.shape)
        loss = loss_fn(outputs, targets)
        optimizer.zero_grad()

        loss.backward()
        optimizer.step()
        
        total_step+=1
        print("epoch:{} train:{},loss:{}".format(j+1,total_step, loss.item()))
        w.add_scalar("loss",loss,total_step) 
        #writer.add_images("imgs", imgs, j+1)
        w.add_scalar("epoch",j,total_step)
    
    torch.save(m,"model.pth")
    print("save")
#writer.close()
