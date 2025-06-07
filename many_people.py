import os
import shutil

import numpy as np
import pandas as pd


def main() -> None:
    df = pd.read_csv("data/new_annotations.csv")
    
    df = df[df["type"] == 1]

    grouped = df.groupby("filename")

    many = 0
    one = 0
    maximum = 0
    res = None
    for filename, group in grouped:
        objects, _ = group.shape
        if objects > maximum:
            maximum = objects
        if objects >= 5:
            many += 1
            shutil.copy2(os.path.join("data/crops", filename), os.path.join("data/many", filename))
            if res is None:
                res = group
            else:
                res = pd.concat([res, group])
        else:
            one += 1
    print(one, many, maximum)
    
    res.to_csv("data/many.csv", index=False)


if __name__ == "__main__":
    main()
