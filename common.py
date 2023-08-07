import os
import random
import time

captcha_array=list("0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
captcha_size=4
from captcha.image import ImageCaptcha
if __name__ == '__main__':
    print(captcha_array)
    image =ImageCaptcha(width=120, height=40)
    for i in range(30000):
        image_val = "".join(random.sample(captcha_array, 4))

        image_name = "./dataset/train/{}_{}.png".format(image_val, int(time.time()))
        image.write(image_val,image_name)
