#!/usr/bin/env python

import numpy as np
import cv2
import matplotlib.pyplot as plt



def tm(template, image, threshold=0.999, scale=1):
    """
    Input
        template: A (k, ell, c)-shaped ndarray containing the k x ell template (with c channels).
        image: An (m, n, c)-shaped ndarray containing the m x n image (with c channels).
        threshold: Minimum normalized cross-correlation value to be considered a match.

    Returns
        matches: A list of (top-left y, top-left x, bounding box height, bounding box width) tuples for each match's bounding box.
    """
    ########## Code starts here ##########
    boundingBoxHeight, boundingBoxWidth = template.shape[0], template.shape[1]
    
    outImg = cv2.matchTemplate(image, template, method=cv2.TM_CCORR_NORMED)
    topLeftPoints = np.where(outImg >= threshold)
    
    return [(point[1]//scale, point[0]//scale, boundingBoxHeight//scale, boundingBoxWidth//scale) for point in zip(*topLeftPoints[::-1])]

def template_match(template, image,
                   num_upscales=2, num_downscales=3,
                   detection_threshold=0.93):
    """
    Input
        template: A (k, ell, c)-shaped ndarray containing the k x ell template (with c channels).
        image: An (m, n, c)-shaped ndarray containing the m x n image (with c channels).
        num_upscales: How many times to 2x-upscale image with Gaussian blur before template matching over it.
        num_downscales: How many times to 0.5x-downscale image with Gaussian blur before template matching over it.
        detection_threshold: Minimum normalized cross-correlation value to be considered a match.

    Returns
        matches: A list of (top-left y, top-left x, bounding box height, bounding box width) tuples for each match's bounding box.
    """
    ########## Code starts here ##########
    scale = 2 * num_upscales
    matches = []
    pyramid = np.zeros(num_upscales+num_downscales+1, dtype=object)
    pyramid[num_upscales] = image
    for i in range(num_upscales-1, -1, -1):
        pyramid[i] = cv2.pyrUp(pyramid[i+1])
    
    for i in range(num_upscales, pyramid.size-1):
        pyramid[i+1] = cv2.pyrDown(pyramid[i])
    
    for img in pyramid:
        matchSet = tm(template, img, detection_threshold, scale)
        scale = scale/2
        for point in matchSet:
            matches.append(point)
    
    return matches
    ########## Code ends here ##########


def create_and_save_detection_image(image, matches, filename="image_detections.png"):
    """
    Input
        image: An (m, n, c)-shaped ndarray containing the m x n image (with c channels).
        matches: A list of (top-left y, top-left x, bounding box height, bounding box width) tuples for each match's bounding box.

    Returns
        None, this function will save the detection image in the current folder.
    """
    det_img = image.copy()
    for (y, x, bbox_h, bbox_w) in matches:
        cv2.rectangle(det_img, (x, y), (x + bbox_w, y + bbox_h), (255, 0, 0), 2)

    cv2.imwrite(filename, det_img)


def main():
    template = cv2.imread('messi_face.jpg')
    image = cv2.imread('messipyr.jpg')

    matches = template_match(template, image)
    create_and_save_detection_image(image, matches)

    template = cv2.imread('stop_signs/stop_template.jpg').astype(np.float32)
    for i in range(1, 6):
        image = cv2.imread('stop_signs/stop%d.jpg' % i).astype(np.float32)
        matches = template_match(template, image, detection_threshold=0.87)
        create_and_save_detection_image(image, matches, 'stop_signs/stop%d_detection.png' % i)


if __name__ == '__main__':
    main()
