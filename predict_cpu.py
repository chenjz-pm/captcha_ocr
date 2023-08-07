from PIL import Image
from torch.utils.data import DataLoader
import one_hot
import model
import torch
import common
import my_datasets
from torchvision import transforms

def pred_pic(pic_path):
    img=Image.open(pic_path)
    tersor_img=transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((60,160)),
        transforms.ToTensor()
    ])
    img=tersor_img(img)
    print(img.shape)
    img=torch.reshape(img,(-1,1,40,120))
    print(img.shape)
    m = torch.load("model.pth")
    outputs = m(img)
    outputs=outputs.view(-1,len(common.captcha_array))
    outputs_lable=one_hot.vectotext(outputs)
    print(outputs_lable)


if __name__ == '__main__':
    pred_pic("./dataset/test/0xdy_1690855514.png")

