import skimage.io
import skimage.transform

from skimage.filters import threshold_local
from skimage.color import rgb2gray


def process_frame_for_context(frame):
    gray_frame = rgb2gray(frame)

    threshold = threshold_local(gray_frame, 21)
    bw_frame = gray_frame > threshold

    return skimage.transform.resize(bw_frame, (30, 48), mode="reflect", order=0).astype("bool").flatten()
