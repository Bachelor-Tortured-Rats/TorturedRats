import numpy as np

def threshold_3d(image, threshold):
    """
    Threshold a 3D image.
    
    Args:
        image (np.ndarray): 3D input image.
        threshold (float): threshold value.
        
    Returns:
        np.ndarray: Thresholded image.
    """
    result = np.zeros_like(image)
    result[image >= threshold] = 1
    return result