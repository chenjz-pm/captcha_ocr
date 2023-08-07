from PIL import Image
from torch.utils.data import DataLoader
import one_hot
import model
import torch
import common
import my_datasets
from torchvision import transforms

import os

def pred_pic(pic_path):
    img=Image.open(pic_path)
    tersor_img=transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((40,120)),
        transforms.ToTensor()
    ])
    img=tersor_img(img).cuda()
    img=torch.reshape(img,(-1,1,40,120))
    m = torch.load("model.pth").cuda()
    outputs = m(img)
    outputs=outputs.view(-1,len(common.captcha_array))
    outputs_lable=one_hot.vectotext(outputs)
    return outputs_lable

if __name__ == '__main__':
    for i in os.walk("./dataset/test/"):
        break

    right=0
    for j in range(len(i[2])):
        str=pred_pic("./dataset/test/{}".format(i[2][j]))
        if i[2][j].split("_")[0] == str:
            print("预测={}, 正确结果={}\t{}".format(str,i[2][j].split("_")[0],"正确"))
            right+=1
        else:
            print("预测={}, 正确结果={}\t{}".format(str,i[2][j].split("_")[0],"错误"))
        
    print("正确率{}%".format(right/len(i[2])*100))
