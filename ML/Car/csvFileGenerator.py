import os

# 指定文件夹路径
folder_path = './raw_data/train/left'

# 获取文件夹下的所有文件
file_list = os.listdir(folder_path)

# 过滤出图片文件（假设图片文件的扩展名为 .jpg, .png, .jpeg 等）
image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')
image_files = [file for file in file_list if file.lower().endswith(image_extensions)]

with open("label.csv", "w") as file:
    for image_file in image_files:
        file.write(f"{image_file}, 0\n")