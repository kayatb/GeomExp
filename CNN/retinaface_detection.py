# ==============================================================================
# Detect and extract faces from images using RetinaFace.
# ==============================================================================

import os
import cv2  # https://opencv.org
from batch_face import RetinaFace  # https://github.com/elliottzheng/batch-face

IMAGE_SIZE = 224


def reshape_image(img, size=500):
    """ Calculate new width and/or height for the input image for better
    detections. The detector splits faces into segments if the image
    resolution is too large and the face is very large. """
    if img.shape[0] >= img.shape[1] and img.shape[0] > size:
        height = size
        width = int(img.shape[1] * height / img.shape[0])
    elif img.shape[0] < img.shape[1] and img.shape[1] > size:
        width = size
        height = int(img.shape[0] * width / img.shape[1])
    else:
        (height, width) = img.shape[:2]

    return height, width


def detect_face_retina(path, detector, img_size, reshape_size, conf_thres=0.7):
    """ Detect the faces using RetinaFace. 
    conf_thres is a confidence threshold; higher means a stricter detector with 
    less false positives, but might have more false negatives. """
    roi = []
    image = cv2.imread(path, 1)

    h, w = image.shape[:2]
    h_r, w_r = reshape_image(image, reshape_size)

    img_r = cv2.resize(image, (w_r, h_r), interpolation=cv2.INTER_AREA)
    faces = detector(img_r, cv=False)

    for f_id, box in enumerate(faces):
        box, _, confidence = box

        if confidence > conf_thres:
            startX = int(box[0] * w / w_r)
            startY = int(box[1] * h / h_r)

            endX = int(box[2] * w / w_r)
            endY = int(box[3] * h / h_r)

            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            cur_fr = image[startY: endY, startX: endX]
            roi.append(cv2.resize(cur_fr, img_size, interpolation=cv2.INTER_AREA))

    return roi


if __name__ == '__main__':
    # added 'patch' to torch.serialization.py line in load() line 574:
    # `map_location=torch.device('cpu')`
    # Please note that the PyPI version of batch_face is outdated (as of 24/04/2021),
    # in the newer version on github this bug is fixed.
    detector = RetinaFace()

    path = "PATH/TO/IMAGES"
    files = sorted(os.listdir(path))

    for file in files:
        f = os.path.join(path, file)
        img = detect_face_retina(f, detector, (IMAGE_SIZE, IMAGE_SIZE), 500)

        for face in img:
            cv2.imwrite(f"PATH/TO/SAVE/IMG/{file}", face)
