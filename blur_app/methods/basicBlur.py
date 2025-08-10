import numpy as np
from numba import njit
from numba import cuda

@njit
def blur_image(image, x, y, w, h, blur_radius):
    """
    Blur bbox - jit, cpu

    Parameters:
    image (ndarray): Input image uint8 or float32, shape (H, W, 3)
    x (int): X coordinate - left, upper corner
    y (int): Y coordinate - left, upper corner
    w (int): Width of bbox
    h (int): Height of bbox
    blur_radius (int): blur radius

    Returns:
    ndarray: Image with blurred bbox
    """
    print("selected box blur : jit")
    blurred_image = image.copy()
    for i in range(y, y + h):
        for j in range(x, x + w):
            if i >= 0 and i < image.shape[0] and j >= 0 and j < image.shape[1]:
                sum_r, sum_g, sum_b = 0, 0, 0
                count = 0
                for dy in range(-blur_radius, blur_radius + 1):
                    for dx in range(-blur_radius, blur_radius + 1):
                        ni, nj = i + dy, j + dx
                        if ni >= 0 and ni < image.shape[0] and nj >= 0 and nj < image.shape[1]:
                            sum_r += image[ni, nj, 0]
                            sum_g += image[ni, nj, 1]
                            sum_b += image[ni, nj, 2]
                            count += 1
                blurred_image[i, j, 0] = sum_r / count
                blurred_image[i, j, 1] = sum_g / count
                blurred_image[i, j, 2] = sum_b / count
    return blurred_image