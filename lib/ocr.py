import skimage.io
import skimage.transform
import skimage.color

from skimage import exposure

from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.filters import threshold_otsu, threshold_local
from skimage.morphology import closing, square, rectangle

import numpy as np

import editdistance

import uuid
import pickle


WINDOWING_SHAPES = {
    "square": square,
    "rectangle": rectangle
}


def image_contains(image, query, ocr_classifier, fuzziness=0, word_window_shape="rectangle", word_window_size=(1, 5)):
    words = words_in_image(image, ocr_classifier, word_window_shape=word_window_shape, word_window_size=word_window_size)

    if isinstance(query, str):
        query = [query]

    result = dict()

    for word in query:
        result[word] = word.lower() in words

    return result


def image_region_contains(image, image_region, query, ocr_classifier, fuzziness=0, word_window_shape="rectangle", word_window_size=(1, 5)):
    region_image = image[image_region[0]:image_region[2], image_region[1]:image_region[3]]

    return image_contains(
        region_image,
        query,
        ocr_classifier,
        fuzziness=fuzziness,
        word_window_shape=word_window_shape,
        word_window_size=word_window_size
    )


def words_in_image(image, ocr_classifier, word_window_shape="rectangle", word_window_size=(1, 5)):
    character_and_word_data = extract_character_and_word_data(
        image,
        word_window_shape=word_window_shape,
        word_window_size=word_window_size,
        preprocess_mode="PRECISE"
    )

    characters = dict()

    for index, image_data in enumerate(character_and_word_data["character"]["image_data"]):
        character_bounding_box = character_and_word_data["character"]["bounding_boxes"][index]
        character = ocr_classifier.predict([image_data.flatten()])[0]

        characters[character_bounding_box] = character

    return reconstruct_words(character_and_word_data["word"]["bounding_boxes"], characters)


def words_in_image_region(image, image_region, ocr_classifier, word_window_shape="rectangle", word_window_size=(1, 5)):
    region_image = image[image_region[0]:image_region[2], image_region[1]:image_region[3]]

    return words_in_image(
        region_image,
        ocr_classifier,
        word_window_shape=word_window_shape,
        word_window_size=word_window_size
    )


def extract_character_and_word_data(image, word_window_shape="square", word_window_size=3, preprocess_mode="FAST"):
    preprocessed_image = preprocess_image(image, mode=preprocess_mode)

    character_bounding_boxes = detect_image_objects_closing(preprocessed_image, window_size=1, merge=True)
    character_image_data = normalize_objects(preprocessed_image, character_bounding_boxes)

    word_bounding_boxes = detect_image_objects_closing(preprocessed_image, shape=word_window_shape, window_size=word_window_size)
    clean_word_bounding_boxes = remove_nested_bounding_boxes(word_bounding_boxes, remove="outer")

    return {
        "character": {
            "bounding_boxes": character_bounding_boxes,
            "image_data": character_image_data
        },
        "word": {
            "bounding_boxes": clean_word_bounding_boxes
        }
    }


def extract_image_objects(image):
    preprocessed_image = preprocess_image(image)
    objects = detect_image_objects_closing(preprocessed_image, shape="square", window_size=1)
    normalized_objects = normalize_objects(preprocessed_image, objects)

    return normalized_objects


def prepare_dataset_tokens(image, image_uuid, mode="FAST"):
    preprocessed_image = preprocess_image(image, mode=mode)
    objects = detect_image_objects_closing(preprocessed_image, shape="square", window_size=1)
    normalized_objects = normalize_objects(preprocessed_image, objects)

    save_objects("datasets/ocr/characters", objects, normalized_objects, image_uuid)


def preprocess_image(image, mode="FAST"):
    grayscale_image = skimage.color.rgb2gray(image)

    lower, higher = np.percentile(grayscale_image, (5, 95))
    contrast_image = exposure.rescale_intensity(grayscale_image, in_range=(lower, higher))

    try:
        if mode == "PRECISE":
            threshold = threshold_local(contrast_image, 35)
        else:
            threshold = threshold_otsu(contrast_image)

        bw = closing(contrast_image > threshold, square(1))
    except ValueError:
        bw = closing(contrast_image, square(1))

    return bw


def detect_image_objects_closing(image, shape="square", window_size=3, merge=False):
    shape_function = WINDOWING_SHAPES.get(shape)
    size_args = window_size if isinstance(window_size, tuple) else [window_size]

    threshold = threshold_otsu(image)
    bw = closing(image > threshold, shape_function(*size_args))

    cleared = clear_border(bw)

    label_image = label(cleared)

    regions = [region.bbox for region in regionprops(label_image)]

    if merge:
        sorted_regions = sorted(regions, key=lambda r: r[1])

        distance_threshold = 5

        merged_regions = list()
        blacklisted_indices = list()

        for i, region in enumerate(sorted_regions):
            if i in blacklisted_indices:
                continue

            next_region = sorted_regions[i + 1] if i < (len(sorted_regions) - 1) else None

            if next_region is not None:
                if region[1] == next_region[1] and region[3] == next_region[3]:
                    if abs(next_region[0] - region[2]) < distance_threshold:
                        merged_regions.append((region[0], region[1], next_region[2], region[3]))
                        blacklisted_indices.append(i + 1)
                        continue

            merged_regions.append(region)

        return merged_regions
    else:
        return regions


def normalize_objects(image, objects):
    cleared_image = clear_border(image)

    tokens = list()
    final_size = (16, 16)

    for region in objects:
        width, height = [region[3] - region[1], region[2] - region[0]]

        token = cleared_image[region[0]:region[2], region[1]:region[3]]

        if width > height:
            diff = (width - height) - 1
            i = 0

            while i <= diff:
                token = np.r_[token, np.zeros((1, width))]
                i += 1
        elif height > width:
            diff = (height - width) - 1
            i = 0

            while i <= diff:
                token = np.c_[token, np.zeros((height, 1))]
                i += 1

        token = skimage.transform.resize(token, output_shape=final_size, mode="reflect", order=0)
        tokens.append(token)

    return tokens


def remove_nested_bounding_boxes(bounding_boxes, remove="outer"):
    bounding_box_blacklist = list()

    for i, bounding_box in enumerate(bounding_boxes):
        for ii, outer_bounding_box in enumerate(bounding_boxes):
            if i == ii:
                continue

            if i in bounding_box_blacklist or ii in bounding_box_blacklist:
                continue

            is_contained = True

            if bounding_box[0] < outer_bounding_box[0]:
                is_contained = False

            if bounding_box[1] < outer_bounding_box[1]:
                is_contained = False

            if bounding_box[2] > outer_bounding_box[2]:
                is_contained = False

            if bounding_box[3] > outer_bounding_box[3]:
                is_contained = False

            if is_contained:
                bounding_box_blacklist.append(ii) if remove == "outer" else bounding_box_blacklist.append(i)

    return [bounding_box for i, bounding_box in enumerate(bounding_boxes) if i not in bounding_box_blacklist]


def reconstruct_words(word_bounding_boxes, characters):
    words = dict()

    for word_bounding_box in word_bounding_boxes:
        words[word_bounding_box] = list()

    for bounding_box, character in characters.items():
        if character == "":
            continue

        # Grouping
        for word_bounding_box in word_bounding_boxes:
            is_contained = True

            if bounding_box[0] < word_bounding_box[0]:
                is_contained = False

            if bounding_box[1] < word_bounding_box[1]:
                is_contained = False

            if bounding_box[2] > word_bounding_box[2]:
                is_contained = False

            if bounding_box[3] > word_bounding_box[3]:
                is_contained = False

            if is_contained:
                words[word_bounding_box].append([bounding_box, character])

    # Sorting
    for word_bounding_box, character_data in words.items():
        y_sorted = sorted(character_data, key=lambda cd: cd[0][0])
        x_sorted = sorted(y_sorted, key=lambda cd: cd[0][1])

        words[word_bounding_box] = [cd[1] for cd in x_sorted]

    return ["".join(sorted_characters) for word_bounding_box, sorted_characters in words.items() if len(sorted_characters) > 0]


def save_objects(path, objects, normalized_objects, image_uuid):
    for index, normalized_object in enumerate(normalized_objects):
        bounding_box = [str(o) for o in objects[index]]

        file_name = f"char_{str(uuid.uuid4())}_frame_{image_uuid}_{'-'.join(bounding_box)}.png"
        skimage.io.imsave(f"{path}/{file_name}", normalized_object)
