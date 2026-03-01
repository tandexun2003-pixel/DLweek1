import cv2
import numpy as np


def fire_color_mask(frame_bgr):
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

    # Fire-like colors are usually in warm hue ranges with moderate/high saturation and value.
    lower_1 = np.array([0, 80, 120])
    upper_1 = np.array([35, 255, 255])
    lower_2 = np.array([160, 80, 120])
    upper_2 = np.array([180, 255, 255])

    mask_1 = cv2.inRange(hsv, lower_1, upper_1)
    mask_2 = cv2.inRange(hsv, lower_2, upper_2)
    mask = cv2.bitwise_or(mask_1, mask_2)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    return mask


def glare_mask(frame_bgr):
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

    lower = np.array([0, 0, 220])
    upper = np.array([180, 40, 255])

    mask = cv2.inRange(hsv, lower, upper)
    return mask


def detect_fire_regions(frame_bgr, min_area=300):
    fire = fire_color_mask(frame_bgr)
    glare = glare_mask(frame_bgr)
    refined = cv2.bitwise_and(fire, cv2.bitwise_not(glare))

    contours, _ = cv2.findContours(refined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        boxes.append((x, y, w, h, area))

    fire_ratio = cv2.countNonZero(refined) / refined.size
    return refined, boxes, fire_ratio
