import os
import shutil
import random

dataset_path = "../dataset"
output_path = "../dataset_split"

split_ratio = (0.7, 0.15, 0.15)

classes = ["real", "fake"]

for cls in classes:
    
    files = os.listdir(os.path.join(dataset_path, cls))
    random.shuffle(files)

    train_split = int(split_ratio[0] * len(files))
    val_split = int(split_ratio[1] * len(files))

    train_files = files[:train_split]
    val_files = files[train_split:train_split+val_split]
    test_files = files[train_split+val_split:]

    for folder, file_list in zip(
        ["train", "validation", "test"],
        [train_files, val_files, test_files]
    ):
        path = os.path.join(output_path, folder, cls)
        os.makedirs(path, exist_ok=True)

        for file in file_list:
            src = os.path.join(dataset_path, cls, file)
            dst = os.path.join(path, file)
            shutil.copy(src, dst)

print("Dataset Split Complete")