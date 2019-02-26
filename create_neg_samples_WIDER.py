import argparse
import os
import cv2
import scipy.misc
from tqdm import tqdm
from config import *
from utils import sliding_window, bb_intersection_over_union, is_image_file, list_images

RED = (0, 0, 255)  # 红
GREEN = (0, 255, 0)  # 绿
BLUE = (255, 0, 0)  # 蓝


def stream_train_images(dir_path, true_rectangles_dict, window_size=(128, 128), window_step=32, visualize=False):
    # 函数产生一系列不包含目标的负样本
    winW, winH = window_size  # 定义窗的大小
    for image_path in list_images(dir_path):
        if not is_image_file(image_path):
            continue
        image = scipy.misc.imread(image_path)  # 读图
        parent_dir_path, image_name = os.path.split(image_path)  # 按照路径将文件名和路径分割开
        parent_dir_name = os.path.split(parent_dir_path)[-1]
        image_name_key = os.path.join(parent_dir_name, image_name)
        true_rectangles = true_rectangles_dict[image_name_key]  # 指定目标box的位置
        if visualize:
            # 可视化目标位置
            clone = image.copy()
            for rect in true_rectangles:
                cv2.rectangle(clone, (rect[0], rect[1]), (rect[2], rect[3]), RED, thickness=2)
        for (x, y, window) in sliding_window(image, step_size=window_step, window_size=(winW, winH)):
            if window.shape[0] != winH or window.shape[1] != winW:
                # 如果剪切框不满足大小，就舍弃
                continue
            if visualize:
                # 可视化裁剪过程
                copy = image.copy()
                cv2.rectangle(copy, (x, y), (x + winW, y + winH), BLUE, thickness=2)
                cv2.imshow(image_path, copy)
                cv2.waitKey(1)
            if all(bb_intersection_over_union((x, y, x + winW, y + winH), rect) == 0 for rect in true_rectangles):
                # 规避掉目标所在位置
                if visualize:
                    cv2.rectangle(clone, (x, y), (x + winW, y + winH), GREEN, thickness=2)
                    cv2.imshow(image_path, clone)
                    cv2.waitKey(1)
                yield image_name, window
        if visualize:
            cv2.destroyAllWindows()

if __name__ == '__main__':
    '''
    # 终端输入和处理
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--images', required=True, help='path to images directory')
    ap.add_argument('-lf', '--labels_file', required=True, help='path to labels file')
    ap.add_argument('-np', '--neg_images_path', required=True, help='path to save neg images')
    args = vars(ap.parse_args())
    images_path = args['images']
    labels_file_path = args['labels_file']
    neg_images_path = args['neg_images_path']
    '''
    images_path ="/home/yhq/Create_negSamples/test_cut/pos_samples"
    labels_file_path = "/home/yhq/Create_negSamples/test_cut/pos_samples/object_loc.txt"
    neg_images_path = "/home/yhq/Create_negSamples/test_cut/neg_samples"

    if not os.path.exists(neg_images_path):
        os.mkdir(neg_images_path)
    true_rectangles_dict = dict()  # 以字典形式存储 key = image_path, value = list of bounding boxes
    with open(labels_file_path, 'r') as f:
        # 整理目标所在区域
        while True:
            name = f.readline().strip()
            if not name:
                break
            count = int(f.readline())
            rectangles = []
            for i in range(count):
                # 得到每个类别下一系列box的信息
                ints = list(map(int, f.readline().split()))
                x1, y1, w, h = ints[0], ints[1], ints[2], ints[3]
                rectangles.append([x1, y1, x1 + w, y1 + h])
            true_rectangles_dict[name] = rectangles
    print('\n Creating negative samples...')
    progress_bar = tqdm(total=20)
    i = 0
    # 生成负样本
    for image_name, window in stream_train_images(
            images_path, true_rectangles_dict, window_size=WINDOW_SIZE,
            window_step=WINDOW_STEP_SIZE, visualize=False):
        if i == 20:
            break
        scipy.misc.imsave(os.path.join(neg_images_path, str(i) + '-' + image_name), window)
        i += 1
        progress_bar.update(1)
    print('Finished creating {} negative samples'.format({NEG_SAMPLES}))