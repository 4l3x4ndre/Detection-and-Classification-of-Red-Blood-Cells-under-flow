import numpy as np
from os.path import expanduser as os_path_expanduser, join as os_path_join, isdir as os_path_isdir
from os import listdir, makedirs as os_makedirs, remove as os_remove
from matplotlib import pyplot as plt
from ultralytics import YOLO


def main():
    model = YOLO('yolo11s.pt')

    results = model.train(
        data='conf/config_yolo_training.yaml',
        epochs=100,
        imgsz=(200, 200),
        #device='cpu'
    )

    # Save the results
    with open("copy_paste_results.pkl", "wb") as f:
        pickle.dump(results, f)

    print("Best model saved at:", results.save_dir)  # Path to saved weights

if __name__ == '__main__':
    main()
