import os
import numpy as np
import torch
from pathlib import Path
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from ultralytics import YOLO
from torchvision.ops import box_iou
from tqdm import tqdm
import glob
import cv2
import warnings

from omegaconf import DictConfig, OmegaConf
from hydra import initialize, compose

CONFIG_FILE = '../conf/config_yolo_training.yaml'

with initialize(config_path=".", version_base="1.3"):
    cfg = compose(config_name=CONFIG_FILE)
    print(OmegaConf.to_yaml(cfg))


# Load model
model = YOLO("runs/detect/train_grace_05_10_4000_coslr/weights/best.pt")

# Run evaluation on the test set
# metrics = model.val(data=CONFIG_FILE, split='test')




# Confusion Matrix on copy-paste dataset (test split)


# Run prediction
# results = model.predict(source=os.path.join(cfg.test, "images"), conf=0.25, verbose=False)

print("=== END OF PREDICTION ===")

class ConfusionMatrix:
     def __init__(self, nc, conf=0.25, iou_thres=0.45):
         self.matrix = np.zeros((nc + 1, nc + 1))
         self.nc = nc  # number of classes
         self.conf = conf
         self.iou_thres = iou_thres

     def process_batch(self, detections, labels):
         """
         Return intersection-over-union (Jaccard index) of boxes.
         Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
         Arguments:
             detections (Array[N, 6]), x1, y1, x2, y2, conf, class
             labels (Array[M, 5]), class, x1, y1, x2, y2
         Returns:
             None, updates confusion matrix accordingly
         """
         if detections is None:
             gt_classes = labels.int()
             for gc in gt_classes:
                 self.matrix[self.nc, gc] += 1  # background FN
             return

         detections = detections[detections[:, 4] > self.conf]
         gt_classes = labels[:, 0].int()
         detection_classes = detections[:, 5].int()
         iou = box_iou(labels[:, 1:], detections[:, :4])

         x = torch.where(iou > self.iou_thres)
         if x[0].shape[0]:
             matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()
             if x[0].shape[0] > 1:
                 matches = matches[matches[:, 2].argsort()[::-1]]
                 matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                 matches = matches[matches[:, 2].argsort()[::-1]]
                 matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
         else:
             matches = np.zeros((0, 3))

         n = matches.shape[0] > 0
         m0, m1, _ = matches.transpose().astype(int)
         for i, gc in enumerate(gt_classes):
             j = m0 == i
             if n and sum(j) == 1:
                 self.matrix[detection_classes[m1[j]], gc] += 1  # correct
             else:
                 self.matrix[self.nc, gc] += 1  # true background

         if n:
             for i, dc in enumerate(detection_classes):
                 if not any(m1 == i):
                     self.matrix[dc, self.nc] += 1  # predicted background

     def tp_fp(self):
         tp = self.matrix.diagonal()  # true positives
         fp = self.matrix.sum(1) - tp  # false positives
         # fn = self.matrix.sum(0) - tp  # false negatives (missed detections)
         return tp[:-1], fp[:-1]  # remove background class

     def plot(self, normalize=True, save_dir='', names=()):
         import seaborn as sn

         array = self.matrix / ((self.matrix.sum(0).reshape(1, -1) + 1E-9) if normalize else 1)  # normalize columns
         array[array < 0.005] = np.nan  # don't annotate (would appear as 0.00)

         fig, ax = plt.subplots(1, 1, figsize=(12, 9), tight_layout=True)
         nc, nn = self.nc, len(names)  # number of classes, names
         sn.set(font_scale=1.0 if nc < 50 else 0.8)  # for label size
         labels = (0 < nn < 99) and (nn == nc)  # apply names to ticklabels
         ticklabels = (names + ['background']) if labels else "auto"
         with warnings.catch_warnings():
             warnings.simplefilter('ignore')  # suppress empty matrix RuntimeWarning: All-NaN slice encountered
             sn.heatmap(array,
                        ax=ax,
                        annot=nc < 30,
                        annot_kws={
                            "size": 8},
                        cmap='Blues',
                        fmt='.2f',
                        square=True,
                        vmin=0.0,
                        xticklabels=ticklabels,
                        yticklabels=ticklabels).set_facecolor((1, 1, 1))
         ax.set_ylabel('True')
         ax.set_ylabel('Predicted')
         ax.set_title('Confusion Matrix')
         fig.savefig(Path(save_dir) / 'confusion_matrix.png', dpi=250)
         plt.show()
         plt.close(fig)

     def print(self):
         for i in range(self.nc + 1):
             print(' '.join(map(str, self.matrix[i])))

image_dir = os.path.join(cfg.test, "images")
label_dir = os.path.join(cfg.test, "labels")
img_paths = sorted(glob.glob(f"{image_dir}/*.png"))

nc = 3  # number of classes in the model
class_names = ['croissant', 'slipper', 'other']  # list of class names (used for plotting)
cm = ConfusionMatrix(nc=nc, conf=0.25, iou_thres=0.45)

def load_labels(path, img_shape):
    labels = np.loadtxt(path.split('.png')[0] + '.txt').reshape(-1, 5)  # cls, x, y, w, h (normalized)
    if labels.size == 0:
        return torch.zeros((0, 5))

    h, w = img_shape
    labels[:, 1] *= w  # x_center to absolute
    labels[:, 2] *= h
    labels[:, 3] *= w
    labels[:, 4] *= h
    # Convert xywh → xyxy
    labels_xyxy = labels.copy()
    labels_xyxy[:, 1] = labels[:, 1] - labels[:, 3] / 2
    labels_xyxy[:, 2] = labels[:, 2] - labels[:, 4] / 2
    labels_xyxy[:, 3] = labels[:, 1] + labels[:, 3] / 2
    labels_xyxy[:, 4] = labels[:, 2] + labels[:, 4] / 2
    return torch.tensor(labels_xyxy)

for img_path in tqdm(img_paths):
    img = cv2.imread(img_path)
    h, w = img.shape[:2]

    # Load labels
    label_path = os.path.join(label_dir, os.path.basename(img_path).replace(".png", ".txt"))
    labels = load_labels(label_path, (h, w))

    # Run inference
    results = model(img, verbose=False)[0]
    detections = results.boxes.data.cpu() if hasattr(results, 'boxes') else results.xyxy[0].cpu()  # Nx6

    # Format: detections = [x1, y1, x2, y2, conf, cls]
    if isinstance(detections, np.ndarray):
        detections = torch.tensor(detections)
    if detections.numel() == 0:
        cm.process_batch(None, labels)
    else:
        cm.process_batch(detections, labels)

cm.print()
cm.plot(normalize=True, save_dir='script/', names=class_names)


print("Program done!")
