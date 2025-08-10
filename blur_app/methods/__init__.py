from .basicBlur import blur_image
from .separableBlurCuda import blur_image_separable_cuda
from .separableBlurCudaWithSM import blur_image_separable_cuda_shared

methods = {
    1 : blur_image,
    2 : blur_image_separable_cuda,
    3 : blur_image_separable_cuda_shared
}

__all__ = ['methods']