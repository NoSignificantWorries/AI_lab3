import os
import random

import numpy as np
from ultralytics import YOLO
import cv2
from tqdm import tqdm


def main(patch_size: int = 320, stride: int = 256) -> None:
    model = YOLO("runs/detect/train4/weights/best.pt")
    
    idx = 0
    files = os.listdir("data/train")
    files = random.sample(files, 30)
    
    with tqdm(files, desc="Process") as pbar:
        for file in pbar:
            filepath = os.path.join("data/train", file)

            img = cv2.imread(filepath)
            height, width, _ = img.shape

            nh = int(np.ceil((height - patch_size) / stride + 1))
            new_height = (nh - 1) * stride + patch_size
            pad_h = (new_height - height) // 2

            nw = int(np.ceil((width - patch_size) / stride + 1))
            new_width = (nw - 1) * stride + patch_size
            pad_w = (new_width - width) // 2
            
            new_img = np.zeros((new_height, new_width, 3), dtype=np.uint8)
            new_img[pad_h:pad_h + height, pad_w:pad_w + width, :] = img
            
            batches = []
            patch_batch = []
            batch_coords = []
            for i in range(nh):
                for j in range(nw):
                    img_patch = new_img[i * stride:i * stride + patch_size, j * stride:j * stride + patch_size, :]
                    if len(patch_batch) >= 8:
                        batches.append((patch_batch, batch_coords))
                        patch_batch = []
                        batch_coords = []
                    patch_batch.append(img_patch)
                    batch_coords.append((i * stride, j * stride))
            
            total_boxes = []
            for batch, coords in batches:
                pred = model.predict(batch, verbose=False)
                for patch, coord in zip(pred, coords):
                    bboxes = patch.boxes.xywh
                    conf = patch.boxes.conf
                    if len(bboxes) <= 0:
                        continue
                    bboxes = list(bboxes.to("cpu"))
                    conf = list(conf.to("cpu"))
                    for bbox, cnf in zip(bboxes, conf):
                        if cnf < 0.7:
                            continue
                        bbox = bbox.numpy()
                        
                        '''
                        limg = patch.orig_img
                        x, y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
                        x1 = int(x - w / 2)
                        y1 = int(y - h / 2)
                        x2 = int(x + w / 2)
                        y2 = int(y + h / 2)
                        cv2.rectangle(limg, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.imwrite(f"data/pred/patch_{idx}.jpg", limg)
                        idx += 1
                        '''
                        
                        bbox[0] += coord[1]
                        bbox[1] += coord[0]
                        total_boxes.append(bbox)
            
            for box in total_boxes:
                x, y, w, h = box[0], box[1], box[2], box[3]
                x1 = int(x - w / 2)
                y1 = int(y - h / 2)
                x2 = int(x + w / 2)
                y2 = int(y + h / 2)
                cv2.rectangle(new_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            cv2.imwrite(f"predict/pred_{idx}.jpg", new_img)
            idx += 1

if __name__ == "__main__":
    main()
