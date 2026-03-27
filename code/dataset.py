import os
import numpy as np
from PIL import Image
from encoding import *
#这几段代码作用是读取所有图片，选择图片最后编码，得到一个4000*36*36的矩阵，保存为npy文件，后续可以直接读取使用
def is_blank_image(img, threshold=10):
    arr = np.array(img)
    return arr.std() < threshold

def read_and_filter_images(img_dir, prefix="Right_3_", count=20):
    images = []
    first_non_blank_found = False
    stop_saving = False
    for i in range(1, count + 1):
        img_path = os.path.join(img_dir, f"{prefix}{i}.png")
        if not os.path.exists(img_path):
            continue
        img = Image.open(img_path).convert('L')
        if not first_non_blank_found:
            if is_blank_image(img):
                continue
            else:
                first_non_blank_found = True
                images.append(np.array(img))
        elif not stop_saving:
            if is_blank_image(img):
                stop_saving = True
                break
            else:
                images.append(np.array(img))
    return images

def select_evenly_spaced(images, num_select=4):
    if len(images) < num_select:
        raise ValueError("非空白图像数量不足")
    center = len(images) // 2
    idxs = []
    if center - 2 >= 0:
        idxs.append(center - 2)
    if center - 1 >= 0:
        idxs.append(center - 1)
    idxs.append(center)
    if center + 1 < len(images):
        idxs.append(center + 1)
    while len(idxs) < num_select:
        if idxs[0] > 0:
            idxs = [idxs[0] - 1] + idxs
        elif idxs[-1] < len(images) - 1:
            idxs.append(idxs[-1] + 1)
        else:
            break
    return [images[i] for i in idxs[:num_select]]

def process_all_data(root_dir, directions=['down', 'up', 'left', 'right'], group_count=1000):#对应1 2 3 4
    all_encoded = []
    for direction in directions:
        dir_path = os.path.join(root_dir, direction)
        for idx in range(1, group_count + 1):
            group_folder = f"{direction}_{idx}"
            group_path = os.path.join(dir_path, group_folder)
            images = read_and_filter_images(group_path, prefix=f"{direction}_{idx}_", count=20)
            if len(images) < 4:
                continue  # 跳过无效数据
            selected_imgs = select_evenly_spaced(images, 4)
            selected_matrix = np.stack(selected_imgs)
            encoded_data = encode_matrix(selected_matrix)
            all_encoded.append(encoded_data)
    return np.stack(all_encoded)

if __name__ == "__main__":
    root_dir = r"D:\作业\毕业设计\data3636"
    data_matrix = process_all_data(root_dir)
    print("最终数据 shape:", data_matrix.shape)  # 应为 (4000, 36, 36)
    # 归一化处理（标准化：减均值除方差）
    mean = data_matrix.mean()
    std = data_matrix.std()
    print(f"归一化前均值: {mean}, 方差: {std}")
    data_matrix = (data_matrix - mean) / std
    print(f"归一化后均值: {data_matrix.mean()}, 方差: {data_matrix.std()}")
    # 保存为npy文件
    np.save("all_encoded_data.npy", data_matrix)
    print("保存完成")
