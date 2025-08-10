from numba import cuda
import numpy as np
import math

# -----------------------------------------------------------------------------
# 1. Horizontal kernel with SM: blur pixels only horizontally, inside (x,y,w,h)
# -----------------------------------------------------------------------------
@cuda.jit
def blur_horiz_shared(in_img, temp_img, x, y, w, h, radius):
    bx, by = cuda.blockDim.x, cuda.blockDim.y
    tx, ty = cuda.threadIdx.x, cuda.threadIdx.y
    gid_x = cuda.blockIdx.x * bx + tx
    gid_y = cuda.blockIdx.y * by + ty
    row = y + gid_y
    col = x + gid_x

    tile_w = bx + 2 * radius
    # shared memory: 3 channels × max tile width
    sdata = cuda.shared.array((3, 48), dtype=np.float32)

    # data to shared memory (with halo)
    for offset in range(tx, tile_w, bx):
        c = offset - radius
        global_col = x + cuda.blockIdx.x * bx + c
        for ch in range(3):
            if (0 <= row < in_img.shape[0]) and (0 <= global_col < in_img.shape[1]):
                sdata[ch, offset] = in_img[row, global_col, ch]
            else:
                sdata[ch, offset] = 0.0
    cuda.syncthreads()

    if (gid_x < w) and (gid_y < h):
        for ch in range(3):
            acc = 0.0
            for k in range(2 * radius + 1):
                acc += sdata[ch, tx + k]
            temp_img[row, col, ch] = acc / (2 * radius + 1)


# -----------------------------------------------------------------------------
# 2. Vertical kernel with SM: blur pixels only upright, inside (x,y,w,h)
# -----------------------------------------------------------------------------
@cuda.jit
def blur_vert_shared(temp_img, out_img, x, y, w, h, radius):
    bx, by = cuda.blockDim.x, cuda.blockDim.y
    tx, ty = cuda.threadIdx.x, cuda.threadIdx.y
    gid_x = cuda.blockIdx.x * bx + tx
    gid_y = cuda.blockIdx.y * by + ty
    row = y + gid_y
    col = x + gid_x

    tile_h = by + 2 * radius
    sdata = cuda.shared.array((3, 48), dtype=np.float32)

    for offset in range(ty, tile_h, by):
        r = offset - radius
        global_row = y + cuda.blockIdx.y * by + r
        for ch in range(3):
            if (0 <= global_row < temp_img.shape[0]) and (0 <= col < temp_img.shape[1]):
                sdata[ch, offset] = temp_img[global_row, col, ch]
            else:
                sdata[ch, offset] = 0.0
    cuda.syncthreads()

    if (gid_x < w) and (gid_y < h):
        for ch in range(3):
            acc = 0.0
            for k in range(2 * radius + 1):
                acc += sdata[ch, ty + k]
            out_img[row, col, ch] = acc / (2 * radius + 1)


# -----------------------------------------------------------------------------
# 3. Host wrapper: copy image, run kernels, get results
# -----------------------------------------------------------------------------
def blur_image_separable_cuda_shared(image, x, y, w, h, radius,
                                     threadsperblock=(16, 16)):
    """
    Separable Blur with shared memory - cuda

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
    print("selected separable blur with SM : cuda jit")

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
    blur_horiz_shared[blocks, threadsperblock](
        in_dev, temp_dev, x, y, w, h, radius
    )
    blur_vert_shared[blocks, threadsperblock](
        temp_dev, out_dev, x, y, w, h, radius
    )

    # 4. Copy output to RAM and convert to uint8
    result = out_dev.copy_to_host()
    return result.astype(np.uint8)