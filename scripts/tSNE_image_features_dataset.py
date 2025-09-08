from termcolor import colored # Color in terminal
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
from tqdm import tqdm
import timm
from timm.data.transforms_factory import create_transform
from timm.data import resolve_data_config
import torch
from PIL import Image
import cv2
from sklearn.manifold import TSNE
import time 
import pickle
import random


def printr(msg):
    print(colored(msg, 'red'))
def printg(msg):
    print(colored(msg, 'green'))
def printw(msg):
    print(colored(msg, 'yellow'))

def get_image_paths_labels(folders, labels, Ns=None):
    """Process all images in a folder and extract their features."""
    image_paths_label = []

    # Get all image files
    for idx_f, folder in enumerate(folders):
        imgs = os.listdir(folder)
        N = Ns[idx_f]
        if N is not None and N > 0:
            imgs = random.sample(imgs, N)
        for i in imgs:
            image_paths_label.append((os.path.join(folder, i), labels[idx_f]))
    return image_paths_label


def apply_tsne(features, n_components=2, perplexity=30):
    """Apply t-SNE to the features."""
    tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=42)

    # Standardize feature dimensions - first make all features 1D arrays

    standardized_features = []
    for f_l in features:  # batch is a list of tensors
        batch = f_l[0]
        label= f_l[1]
        #flat_feat = feat.flatten()  # convert to 1D NumPy array
        standardized_features.append(batch)
        #break

    # Now handle different lengths by either truncating or padding
    min_length = min(len(f) for f in standardized_features)
    max_length = max(len(f) for f in standardized_features)

    print(f"Feature vectors have varying lengths from {min_length} to {max_length}")

    # Choose approach: we can either truncate all to minimum length or pad to maximum
    # Truncation approach (often works better for t-SNE)
    truncated_features = np.array([f[:min_length] for f in standardized_features])
    print(f"Standardized all feature vectors to length: {min_length}")

    return tsne.fit_transform(truncated_features)

def visualize_tsne(tsne_results, image_paths, labels, output_path='tsne_visualization.png'):
    """Visualize the t-SNE results."""
    plt.figure(figsize=(12, 10))

    df = pd.DataFrame(tsne_results, columns=['x', 'y'])
    df['label'] = labels
    print(df.describe())
    print(df.columns)
    
    # Plot points
    #plt.scatter(tsne_results[:, 0], tsne_results[:, 1], s=10)
    sns.scatterplot(df, x='x', y='y', hue='label')

    # Add image filename labels
    """
    for i, path in enumerate(image_paths):
        filename = os.path.basename(path)
        plt.annotate(filename, (tsne_results[i, 0], tsne_results[i, 1]),
                     fontsize=8, alpha=0.7)
    """

    plt.title('t-SNE Visualization of Image Features')
    plt.xlabel('t-SNE dimension 1')
    plt.ylabel('t-SNE dimension 2')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    #plt.show()

def main():

    
    printw('\nLoading resnet50...')
    timm_model = timm.create_model('resnet50', pretrained=True, features_only=True)
    timm_model.eval()
    timm_model.requires_grad_(False)
    transform = create_transform(**resolve_data_config(timm_model.pretrained_cfg, model=timm_model))
    printg('Resnet50 loaded.')

    N = [2000, 1000, 1000]
    printw(f'\nLoading {N} images...')
    images = get_image_paths_labels(
        ['../data/cut_and_paste/train/images', '../data/cut_and_paste/test/images', '../data/cut_and_paste/val/images'],
        ['train', 'test', 'val'],
        Ns=N)
    printg('\nLoaded images')

    printr('\nCreating features...')
    feats = []
    for img, label in tqdm(images):
        res = timm_model(transform(Image.fromarray(cv2.imread(img))).unsqueeze(0))
        last_feat = res[-1]
        pooled_feat = torch.nn.functional.adaptive_avg_pool2d(
            last_feat, (1, 1)
        ).squeeze().numpy()  # shape: [2048]
        
        feats.append((pooled_feat, label))
    printg(f'\nFeatures created: len(feats)={len(feats)}, len(feats[0][0])={len(feats[0][0])}')


    printr('\n\nApplying t-SNE...')
    tsne_results = apply_tsne(feats, n_components=2, perplexity=30)
    printg('t-SNE applied.')
    
    printr('Visualizing t-SNE...')
    visualize_tsne(
        tsne_results, 
        [img for img, _ in images],
        [label for _, label in images],
        output_path='tsne_visualization.png'
    )
    printg('t-SNE visualization saved to tsne_visualization.png')

    r = {
            'images': [img for img, _ in images],
            'labels': [label for _, label in images],
            'tsne': [tr for tr in tsne_results],
            'feats': [f for f in feats]
    }
    with open('tsne_vis_result.pickle', 'wb') as ofile:
        pickle.dump(r, ofile)

    printr('Features and t-SNE results saved to tsne_vis_result.pickle')

    printr('Program done.')

if __name__ == "__main__":
    main()
