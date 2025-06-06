import os
import random

import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm


def get_stat(dt):
    return dt.mean(), dt.min(), dt.max()


def generate_random_negative_crops(image_width, image_height, bboxes, num_crops, crop_size=320):
    negative_crops = []
    attempts = 0
    max_attempts = num_crops * 10

    while len(negative_crops) < num_crops and attempts < max_attempts:
        x = random.randint(0, image_width - crop_size)
        y = random.randint(0, image_height - crop_size)

        overlap = False
        for i, row in bboxes.iterrows():
            if not (x + crop_size < row["xmin"] or x > row["xmax"] or y + crop_size < row["ymin"] or y > row["ymax"]):
                overlap = True
                break

        if not overlap:
            negative_crops.append((x, y, crop_size, crop_size))
        attempts += 1

    return negative_crops


def get_boxes_groups(image_width, image_height, bboxes: pd.DataFrame):
    tmp = np.zeros((image_height, image_width), dtype=np.uint8)
    for i, row in bboxes.iterrows():
        tmp[row["ymin"]:row["ymax"] + 1, row["xmin"]:row["xmax"] + 1] = 1
    
    kernel = np.ones((320, 320), np.uint8)
    mask = cv2.dilate(tmp, kernel, iterations=1)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

    grouped_boxes = []
    for i in range(1, num_labels):
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        width = stats[i, cv2.CC_STAT_WIDTH]
        height = stats[i, cv2.CC_STAT_HEIGHT]

        grouped_boxes.append((x, y, width, height))
    
    return grouped_boxes


def main() -> None:
    df = pd.read_csv("data/_annotations.csv")

    df["bw"] = df["xmax"] - df["xmin"]
    df["bh"] = df["ymax"] - df["ymin"]
    df["area"] = df["bw"] * df["bh"]

    print("Area:", *get_stat(df["area"]))
    print("Height:", *get_stat(df["bh"]))
    print("Width:", *get_stat(df["bw"]))
    
    matrix = np.zeros((3000, 4000), dtype=np.uint8)
    grouped = df.groupby("filename")
    counts = []
    collisions = {}
    for filename, group in grouped:
        counts.append(0)
        tmp = np.zeros((3000, 4000), dtype=np.uint8)
        for i, row in group.iterrows():
            if tmp[row["ymin"]:row["ymax"] + 1, row["xmin"]:row["xmax"] + 1].sum() > 0:
                if filename not in list(collisions.keys()):
                    collisions[filename] = 1
                else:
                    collisions[filename] += 1
            tmp[row["ymin"]:row["ymax"] + 1, row["xmin"]:row["xmax"] + 1] = 1
            counts[-1] += 1
        matrix += tmp
    
    counts = np.array(counts)
    print(collisions)
    
    fig, ax = plt.subplots(nrows=2, ncols=1)

    ax[0].hist(counts, bins=10, edgecolor="black", alpha=0.7)
    ax[1].imshow(matrix)
    
    plt.show()


if __name__ == "__main__":
    # main()
    save_path = "data/crops"
    df = pd.read_csv("data/_annotations.csv")
    annotations = {
        "filename": [],
        "type": [],
        "width": [],
        "height": [],
        "xmin": [],
        "ymin": [],
        "xmax": [],
        "ymax": []
    }
    
    grouped = df.groupby("filename")

    idx = 0
    with tqdm(grouped, desc="Processing") as pbar:
        for filename, group in pbar:
            img = cv2.imread(f"data/train/{filename}")
            
            '''
            for i, row in df.iterrows():
                cv2.rectangle(img, (row["xmin"], row["ymin"]), (row["xmax"], row["ymax"]), (0, 0, 255), 2)
            '''

            boxes = get_boxes_groups(4000, 3000, group)
            for box in boxes:
                crop = img[box[1]:box[1] + box[3], box[0]:box[0] + box[2]]
                save_file = f"crop_{idx:06d}.jpg"
                cv2.imwrite(os.path.join(save_path, save_file), crop)
                # cv2.rectangle(img, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), (0, 255, 0), 2)
                for i, row in group.iterrows():
                    if row["xmin"] >= box[0] and row["xmax"] < (box[0] + box[2]) and row["ymin"] >= box[1] and row["ymax"] < box[1] + box[3]:
                        annotations["filename"].append(save_file)
                        annotations["type"].append(1)
                        annotations["width"].append(box[2])
                        annotations["height"].append(box[3])
                        annotations["xmin"].append(row["xmin"] - box[0])
                        annotations["ymin"].append(row["ymin"] - box[1])
                        annotations["xmax"].append(row["xmax"] - box[0])
                        annotations["ymax"].append(row["ymax"] - box[1])
                idx += 1
            
            neg_boxes = generate_random_negative_crops(4000, 3000, group, num_crops=np.ceil(len(boxes) / 2))
            for box in neg_boxes:
                neg_crop = img[box[1]:box[1] + box[3], box[0]:box[0] + box[2]]
                # cv2.rectangle(img, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), (255, 0, 0), 2)
                save_file = f"crop_{idx:06d}.jpg"
                cv2.imwrite(os.path.join(save_path, save_file), neg_crop)
                annotations["filename"].append(save_file)
                annotations["type"].append(0)
                annotations["width"].append(box[2])
                annotations["height"].append(box[3])
                annotations["xmin"].append(0)
                annotations["ymin"].append(0)
                annotations["xmax"].append(0)
                annotations["ymax"].append(0)
                idx += 1
        
    res_group = pd.DataFrame(annotations)
    res_group.to_csv("data/new_annotations.csv", index=False)
