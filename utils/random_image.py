import os
import random
from PIL import Image

def random_image_from_dataset(dataset_path):
    # 获取所有子文件夹
    subfolders = [f.path for f in os.scandir(dataset_path) if f.is_dir()]
    # 从子文件夹中随机选择一个
    random_subfolder_path = random.choice(subfolders)
    # 获取该子文件夹中的所有图像
    images = os.listdir(random_subfolder_path)
    # 从图像中随机选择一个
    random_image_name = random.choice(images)
    # 获取图像的完整路径
    image_path = os.path.join(random_subfolder_path, random_image_name)
    # 打开并返回图像
    image = Image.open(image_path)
    return image

# 使用方法
if __name__ == "__main__":
    dataset_path = "D:\Datasets\mini_cats_and_dogs"
    image = random_image_from_dataset(dataset_path)
    image.show()