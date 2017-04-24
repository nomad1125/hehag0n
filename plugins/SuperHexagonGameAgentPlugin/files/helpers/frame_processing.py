import skimage.io
import skimage.transform
import skimage.exposure
import skimage.filters
import skimage.measure
import skimage.color
import skimage.segmentation

import numpy as np


def image_data_for_screen_region(frame, screen_region):
    return frame[screen_region[0]:screen_region[2], screen_region[1]:screen_region[3]]


def process_frame_for_context(frame):
    gray_frame = skimage.color.rgb2gray(frame)

    threshold = skimage.filters.threshold_local(gray_frame, 21)
    bw_frame = gray_frame > threshold

    return skimage.transform.resize(bw_frame, (30, 48), mode="reflect", order=0).astype("bool").flatten()


def process_frame_for_game_play(frame):
    gray_frame = np.array(skimage.color.rgb2gray(frame) * 255, dtype="uint8")

    histogram = skimage.exposure.histogram(gray_frame)

    max_bin_indices = np.argpartition(histogram[0], -2)[-2:]
    background_colors = [histogram[1][max_bin_indices[0]], histogram[1][max_bin_indices[1]]]

    gray_frame[gray_frame == background_colors[0]] = 0
    gray_frame[gray_frame == background_colors[1]] = 0

    threshold = skimage.filters.threshold_otsu(gray_frame[40:])
    bw_frame = gray_frame > threshold

    return bw_frame


def get_player_character_bounding_box(frame, screen_region):
    player_area_frame = image_data_for_screen_region(frame, screen_region)
    cleared_player_area_frame = skimage.segmentation.clear_border(player_area_frame)

    label_image = skimage.measure.label(cleared_player_area_frame)

    player_character_bounding_box = None

    for region in skimage.measure.regionprops(label_image):
        if region.area < 200:
            if region.bbox[0] > 50:
                player_character_bounding_box = [c + screen_region[i % 2] for i, c in enumerate(list(region.bbox))]

    return player_character_bounding_box
