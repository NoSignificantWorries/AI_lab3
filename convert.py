import os
import shutil

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def data_split_and_convert(df, base_img_dir: str, save_dir_base: str, images_dir: str, labels_dir: str) -> None:
    grouped = df.groupby("filename")

    with open(save_dir_base, "w") as f:
        for filename, group in grouped:
            label_filename = os.path.splitext(filename)[0] + '.txt'
            label_path = os.path.join(labels_dir, label_filename)
            
            shutil.copy2(os.path.join(base_img_dir, filename), os.path.join(images_dir, filename))
            f.write(os.path.abspath(os.path.join(images_dir, filename)) + "\n")

            with open(label_path, "w") as file:
                for i, row in group.iterrows():
                    file.write(f"0 {row["x"]:.10f} {row["y"]:.10f} {row["b_width"]:.10f} {row["b_height"]:.10f}\n")


def main(root_dir: str, base_csv: str, base_dir: str, save_dir: str) -> None:
    df = pd.read_csv(os.path.join(root_dir, base_csv))
    print(df.shape)

    df["dw"] = 1.0 / df["width"]
    df["dh"] = 1.0 / df["height"]

    df["x"] = ((df["xmax"] + df["xmin"]) / 2) * df["dw"]
    df["y"] = ((df["ymax"] + df["ymin"]) / 2) * df["dh"]
    df["b_width"] = (df["xmax"] - df["xmin"]) * df["dw"]
    df["b_height"] = (df["ymax"] - df["ymin"]) * df["dh"]
    df = df.drop(["width", "height", "xmin", "xmax", "ymin", "ymax", "class"], axis=1)
    
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    df = {"train": train_df, "val": val_df}
    
    for group in df.keys():
        data_split_and_convert(df[group], base_dir, os.path.join(save_dir, f"{group}.txt"), os.path.join(save_dir, "images", group), os.path.join(save_dir, "labels", group))
    

if __name__ == "__main__":
    main("data", "_annotations.csv", "data/train", "data/dataset")
