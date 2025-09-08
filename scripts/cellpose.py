from cellpose import models
from cellpose.io import imread
import numpy as np
from typing import List
from os import listdir
from os.path import isfile, join
from cv2 import cvtColor, COLOR_BGR2GRAY, imread as cv2_imread
from tqdm import tqdm

def LoadImages(image_path: str, extension: str = '.png', funcOnFilenames: callable = None, grayscale: bool = False) -> List[np.ndarray]:
    """
    Loads images from a specified path.

    Parameters:
    -----------
    image_path: str
        Path to the directory containing images.
    extension: str, optional
        File extension of the images to load. Default is '.png'.
    funcOnFilenames: callable, optional
        Function to apply on filenames to check if they should be loaded.
        Default is None, which means all files with the specified extension will be loaded.
    grayscale: bool, optional
        If True, images will be loaded in grayscale mode. Default is False.

    Returns:
    --------
    List[np.ndarray]
        List of images loaded as numpy arrays.
    """
    
    images = []
    filenames = []
    for file in sorted(listdir(image_path))[:20]:
        if not isfile(join(image_path, file)) or not file.endswith(extension):
            continue
        if funcOnFilenames is not None and not funcOnFilenames(file):
            continue

        filenames.append(file)
        image = cv2_imread(join(image_path, file))
        if grayscale:
            image = cvtColor(image, COLOR_BGR2GRAY)
        if image is None:
            continue
        images.append(image)
    
    return images, filenames

def SegmentImages(images: List[np.ndarray],model_type: str = 'nuclei', diameter: float = None) -> List[np.ndarray]:
    """
    Segments a list of images using the Cellpose model.

    Parameters:
    -----------
    images: List[np.ndarray]
        List of images to be segmented. Each image should be a numpy array.
    model_type: str
        Type of Cellpose model to use. Default is 'nuclei'. 
    diameter: float, optional
        Diameter of the objects to be segmented. If None, Cellpose will estimate it automatically.

    Returns:
    --------
    List[np.ndarray]
        List of masks corresponding to each input image. Each mask is a numpy array where each pixel
        value corresponds to the label of the segmented object.
    """
    
    # Initialize the Cellpose model
    model = models.Cellpose(gpu=False, model_type=model_type)
    
    # Segment each image and store results
    masks = []
    for image in tqdm(images):
        mask, _, _, _ = model.eval(image, diameter=diameter, channels=[0,0], flow_threshold=0.4, do_3D=False)
        masks.append(mask)
    
    return masks


