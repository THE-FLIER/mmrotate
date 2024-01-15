#lambelme  标 转  yolov
import json
import os
import math
import numpy as np
'''
会在同一目录下生成txt训练文件
'''
def dist(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

def sorted(np_points,width, height):
    width, height = width, height
    # left_bottom = [0, 0]
    # left_top = [0, height]
    # right_bottom = [width, 0]
    # right_top = [width, height]
    sorted_points = []
    np_points = np_points.tolist()
    dst = [[0, 0], [width, 0], [width, height],[0, height]]
    for p in dst:
        min_dist = float("inf")
        closest_point = None
        for q in np_points:
            d = dist(p, q)
            if d < min_dist:
                min_dist = d
                closest_point = q
        sorted_points.append(closest_point)


    return np.array(sorted_points, np.float32)

def order_points_with_vitrual_center(pts, width, height):
    pts = np.array(pts, dtype="float32")
    pts_ =pts
    center_x = np.mean(pts[:, 0])
    center_y = np.mean(pts[:, 1])

    # 分为上下两组
    upper = pts[pts[:, 1] < center_y]
    lower = pts[pts[:, 1] >= center_y]

    # 在每组内部按照x值排序以分出左右
    upper_sorted = upper[np.argsort(upper[:, 0]), :]
    lower_sorted = lower[np.argsort(lower[:, 0]), :]

    # 确保上下两组都有两个点
    if upper_sorted.shape[0] != 2 or lower_sorted.shape[0] != 2:
        sorted_pts = sorted(pts_, width, height)
        return sorted_pts
    # 合并左上、右上、右下、左下的点
    sorted_pts = np.array([upper_sorted[0], upper_sorted[1], lower_sorted[1], lower_sorted[0]], np.float32)
    return sorted_pts
def yolo_transform(json_path,dst_dir):
    import glob
    import numpy as np
    json_path = json_path
    json_files = glob.glob(json_path + "/*.json")
    for json_file in json_files:
        # if json_file != r"C:\Users\jianming_ge\Desktop\code\handle_dataset\water_street\223.json":
        #     continue
        print(json_file)
        f = open(json_file)
        json_info = json.load(f)
        # print(json_info.keys())
        #img = cv2.imread(os.path.join(json_path, json_info["imagePath"]))
        height = json_info['imageHeight']
        width = json_info['imageWidth']
        np_w_h = np.array([[width, height]], np.int32)
        txt_file = os.path.basename(json_file)
        txt_file = f'{json_path}/{txt_file[:-5]}.txt'
        f = open(txt_file, "w")
        for point_json in json_info["shapes"]:
            if len(point_json["points"]) ==4:
                txt_content = ""
                np_points = np.array(point_json["points"], np.int32)
                np_points = order_points_with_vitrual_center(np_points, width, height)
                # norm_points = np_points / np_w_h
                norm_points_list = np_points.tolist()
                txt_content += " ".join([" ".join([str(cell[0]), str(cell[1])]) for cell in norm_points_list]) + " book"+" 0"+"\n"
                f.write(txt_content)

def data_spilt(txt_path,dst_dir):
    import os
    import shutil
    import random

    # 源文件夹路径
    src_dir = txt_path

    # 目标文件夹路径
    dst_dir = dst_dir

    # 创建目标文件夹
    os.makedirs(dst_dir, exist_ok=True)

    # 创建train和val文件夹
    train_dir = os.path.join(dst_dir, 'train')
    val_dir = os.path.join(dst_dir, 'val')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    # 在train和val文件夹内创建image和annotation文件夹
    os.makedirs(os.path.join(train_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(train_dir, 'annofiles'), exist_ok=True)
    os.makedirs(os.path.join(val_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(val_dir, 'annofiles'), exist_ok=True)

    # 获取源文件夹内所有的图片文件
    img_files = [f for f in os.listdir(src_dir) if f.endswith('.jpg') or f.endswith('.png')]

    # 随机打乱文件顺序
    random.shuffle(img_files)

    # 按照8:2的比例划分train和val
    num_train = int(len(img_files) * 0.8)
    train_img_files = img_files[:num_train]
    val_img_files = img_files[num_train:]

    # 将图片文件和对应的.txt文件复制到对应的文件夹
    for f in train_img_files:
        shutil.copy(os.path.join(src_dir, f), os.path.join(train_dir, 'images', f))
        shutil.copy(os.path.join(src_dir, f.rsplit('.', 1)[0] + '.txt'),
                    os.path.join(train_dir, 'annofiles', f.rsplit('.', 1)[0] + '.txt'))
    for f in val_img_files:
        shutil.copy(os.path.join(src_dir, f), os.path.join(val_dir, 'images', f))
        shutil.copy(os.path.join(src_dir, f.rsplit('.', 1)[0] + '.txt'),
                    os.path.join(val_dir, 'annofiles', f.rsplit('.', 1)[0] + '.txt'))

if __name__=="__main__":
    txt_path = "./dataset/coco_500_extract"
    dst_dir = './dataset/700_bookspine'
    yolo_transform(txt_path,dst_dir)
    data_spilt(txt_path,dst_dir)
