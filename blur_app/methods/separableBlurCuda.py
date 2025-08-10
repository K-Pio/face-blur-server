from numba import cuda
import numpy as np
import math

# ------------------------------------------------------------------------
# 1. Horizontal kernel: blur pixels only horizontally, inside (x,y,w,h)
# ------------------------------------------------------------------------
@cuda.jit
def blur_horiz_kernel(in_img, temp_img, x, y, w, h, radius):
    tx = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    ty = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y

    
    if tx >= w or ty >= h:
        return

    row = y + ty
    col = x + tx
    H, W = in_img.shape[0], in_img.shape[1]

    for c in range(3):
        sum_val = 0.0
        cnt = 0
        
        for dx in range(-radius, radius + 1):
            cc = col + dx
            if 0 <= cc < W:
                sum_val += in_img[row, cc, c]
                cnt += 1
        temp_img[row, col, c] = sum_val / cnt


# ------------------------------------------------------------------------
# 2. Vertical kernel: blur pixels only upright, inside (x,y,w,h)
# ------------------------------------------------------------------------
@cuda.jit
def blur_vert_kernel(temp_img, out_img, x, y, w, h, radius):
    tx = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    ty = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y

    if tx >= w or ty >= h:
        return

    row = y + ty
    col = x + tx
    H, W = temp_img.shape[0], temp_img.shape[1]

    for c in range(3):
        sum_val = 0.0
        cnt = 0
        # zbierz sąsiadów w pionie
        for dy in range(-radius, radius + 1):
            rr = row + dy
            if 0 <= rr < H:
                sum_val += temp_img[rr, col, c]
                cnt += 1
        out_img[row, col, c] = sum_val / cnt


# ------------------------------------------------------------------------
# 3. Host wrapper: copy image, run kernels, get results
# ------------------------------------------------------------------------
def blur_image_separable_cuda(image, x, y, w, h, radius,
                              threadsperblock=(16, 16)):
    """
    Separable Blur - cuda

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
    print("selected separable blur : cuda jit")
    # 1. Conversion and transfer to device (GPU)
    img_f32 = image.astype(np.float32)
    in_dev   = cuda.to_device(img_f32)
    temp_dev = cuda.device_array_like(img_f32)

    # OUT_DEV: full image copy
    out_dev  = cuda.to_device(img_f32)

    # 2. Set grid size
    blocks_x = math.ceil(w / threadsperblock[0])
    blocks_y = math.ceil(h / threadsperblock[1])
    blocks   = (blocks_x, blocks_y)

    # 3. Kernel: horizontal -> vertical
    blur_horiz_kernel[blocks, threadsperblock](
        in_dev, temp_dev, x, y, w, h, radius
    )
    blur_vert_kernel[blocks, threadsperblock](
        temp_dev, out_dev, x, y, w, h, radius
    )

    # 4. Copy output to RAM and convert to uint8
    result = out_dev.copy_to_host()
    return result.astype(np.uint8)