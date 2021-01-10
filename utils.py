import imutils
import cv2
import numpy as np
from imutils import contours
from os.path import join
from time import time
import json

# seven segment display
DIGITS_LOOKUP = {
    (1, 1, 1, 0, 1, 1, 1): 0,
    (0, 0, 1, 0, 0, 1, 0): 1,
    (1, 0, 1, 1, 1, 0, 1): 2,
    (1, 0, 1, 1, 0, 1, 1): 3,
    (0, 1, 1, 1, 0, 1, 0): 4,
    (1, 1, 0, 1, 0, 1, 1): 5,
    (1, 1, 0, 1, 1, 1, 1): 6,
    (1, 1, 1, 0, 0, 1, 0): 7,  # ???
    (1, 1, 1, 1, 1, 1, 1): 8,
    (1, 1, 1, 1, 0, 1, 1): 9
}

def _show_image(win, img, destroy=True):
    cv2.imshow(win, img)
    cv2.waitKey(0)
    if destroy:
        cv2.destroyWindow(win)


def read_digits(gray, image, skew=7, avg_digit_width=12, debug=False):
    _, gray = cv2.threshold(gray, 100, 255, 0)  # extract white digits
    gray_inv = cv2.bitwise_not(gray)   # turn to black digits

    # to locate digits area
    cnts = cv2.findContours(gray_inv, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    largest_area = sorted(cnts, key=cv2.contourArea)[-1]  # find LCD = the largest white background
    mask = np.zeros(image.shape, np.uint8)  # all black in mask
    cv2.drawContours(mask, [largest_area], 0, (255, 255, 255), -1)  # make roi area white in mask
    output = cv2.bitwise_and(image, mask)  # pick roi from image
    roi_gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)  # white digits
    _, roi_gray = cv2.threshold(roi_gray, 100, 255, 0)  # highlight white digits

    # to find each digit
    thresh = cv2.threshold(roi_gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    (x, y, w, h) = cv2.boundingRect(cnts[0])
    roi_small = thresh[y:y + h, x:x + w]
    warped = roi_small.copy()  # black digits

    # the numbers displayed a little bit leaning to right side, to make them upright
    height, width = warped.shape
    width -= 1
    height -= 1
    rect = np.array([
        [0, 0],
        [width, 0],
        [width, height],
        [0, height]], dtype="float32")
    dst = np.array([
        [-skew, 0],
        [width - skew, 0],
        [width, height],
        [0, height]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(warped, M, (width + 1, height + 1))
    output = cv2.warpPerspective(output, M, (width + 1, height + 1))

    # segment 2 and segment 5 separated so we do vertical dilate and erode to connect them
    vert_dilate3 = cv2.getStructuringElement(cv2.MORPH_RECT, ksize=(1, 3))
    dilation = cv2.dilate(warped, vert_dilate3)
    dilation = cv2.erode(warped, vert_dilate3)  # black digits
    dilation_inv = cv2.bitwise_not(dilation)  # white digits

    # locate each digit
    cnts = cv2.findContours(dilation_inv.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    digitCnts = []
    # loop over the digit area candidates
    for _c, c in enumerate(cnts):
        (x, y, w, h) = cv2.boundingRect(c)
        if debug:
            print("Contour {}: w={}, h={}, x={}, y={}".format(_c, w, h, x, y))

        if 10 <= h <= 35 and w <= 20 and y < 20:
            digitCnts.append(c)
    # sort the contours from left-to-right, then initialize the actual digits themselves
    # avoid error: ValueError: not enough values to unpack (expected 2, got 0)
    try:
        digitCnts = contours.sort_contours(digitCnts, method="left-to-right")[0]
    except ValueError:
        return [-1]

    # todo if len(digitCnts)>4, pick up y != all other y, because y should be same to all digits, x increase by w
    # current solution is y < 20
    # find exact number
    digits = []
    point = None

    # loop over each of the digits
    for _c, c in enumerate(digitCnts):
        (x, y, w, h) = cv2.boundingRect(c)
        if debug:
            print("Selected contour {}: w={}, h={}, x={}, y={}".format(_c, w, h, x, y))
        # manually override the width of number 1
        if w < 9:
            x -= avg_digit_width - w
            w = avg_digit_width
            if debug:
                print("  changed contour : w={}, h={}, x={}, y={}".format(w, h, x, y))
        elif w > avg_digit_width + 1:
            w = avg_digit_width
            point = _c
        roi = dilation_inv[y:y + h, x:x + w]
        # compute the approximate width and height of each segment based on the ROI dimensions.
        (roiH, roiW) = roi.shape
        (dW, dH) = (int(roiW * 0.34), int(roiH * 0.25))
        dHC = int(roiH * 0.1)
        # print("roiH={}, roiW={}, dH={}, dW={}, dHC={}".format(roiH, roiW, dH, dW, dHC))

        # define the coordinates of set of 7 segments
        segments = [
            ((1, 0), (w, dH)),  # top
            ((1, 0), (dW, h // 2)),  # top-left
            ((w - dW, 0), (w - 2, h // 2)),  # top-right
            ((4, (h // 2) - dHC), ((w // 2) + 1, (h // 2) + dHC)),  # center
            ((0, h // 2), (dW, h)),  # bottom-left
            ((w - dW - 2, h // 2), (w, h)),  # bottom-right
            ((0, h - dH), (w - 2, h))  # bottom
        ]
        on = [0] * len(segments)
        # print("segments={}".format(segments))

        if show_image:
            _show_image("ROI {}".format(_c), roi, False)

        # loop over the segments
        for (i, ((xA, yA), (xB, yB))) in enumerate(segments):
            # extract the segment ROI, count the total number of thresholded pixels in the segment, and then compute
            # the area of the segment
            segROI = roi[yA:yB, xA:xB]
            total = cv2.countNonZero(segROI)  # (0, 0, 0)=black && (255, 255, 255)=white
            area = (xB - xA) * (yB - yA)
            # if the total number of non-zero pixels is greater than 50% of the area, mark the segment as "on"
            if debug:
                print(i, total / float(area))
            if total / float(area) > 0.5:
                on[i] = 1

            if show_image:
                _show_image("Segment {} of ROI {}".format(i, _c), segROI)

        # lookup the digit and draw it on the image give -1 for lookup failure
        if tuple(on) in DIGITS_LOOKUP.keys():
            digit = DIGITS_LOOKUP[tuple(on)]
        else:
            digit = -1
            if debug:
                print(on)
                cv2.imshow("ROI", roi)

        digits.append(digit)
        # deal with decimal point
        if point is not None and '.' not in digits:
            digits.append('.')
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 1)  # (0, 255, 0)=green
        cv2.putText(image, str(digit), (x + 1, y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)

    # assert decimal point
    if point is None and len(digits) >= 4:
        digits.insert(-3, '.')

    # display the digits
    _show_image("Image read digits={}".format(digits), image, destroy=False)
    return digits


def stable_marker_detector(gray):
    """
    detect stable marker
    """
    # process gray image and have a copy
    gray_orig = gray.copy()
    gray_inv = cv2.bitwise_not(gray)
    ret, gray = cv2.threshold(gray_inv, 200, 255, 0)  # extract white area

    # exclude digits and outer layers
    cnts = cv2.findContours(gray, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    largest_area = sorted(cnts, key=cv2.contourArea)[-1]
    (x, y, w, h) = cv2.boundingRect(largest_area)
    m = gray_orig.copy()
    mask_gray = np.zeros(gray.shape, np.uint8)  # all black in mask
    cv2.drawContours(mask_gray, [largest_area], 0, (255, 255, 255), -1)
    # make outer of m to white
    m[0:y, :] = 255
    m[y + h:, :] = 255
    m[y:y + h, 0:x] = 255
    m[y:y + h, x + w:] = 255
    m = cv2.bitwise_and(m, mask_gray)  # pick roi from image
    m[0:y, :] = 0
    m[y + h:, :] = 0
    m[y:y + h, 0:x] = 0
    m[y:y + h, x + w:] = 0
    m = cv2.threshold(m, 100, 255, 0)[1]  # extract white area

    cnts = cv2.findContours(m, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    # print('{} contours found'.format(len(cnts)))
    # cv2.imshow('Contours', m)
    # cv2.waitKey(0)

    is_stable = False
    if len(cnts) > 0:
        for _c, c in enumerate(cnts):
            (x, y, w, h) = cv2.boundingRect(c)
            print("marker {}: w={}, h={}, x={}, y={}".format(_c, w, h, x, y))
            # cv2.drawContours(m, cnts, _c, (255, 255, 255), 1)
            # cv2.imshow('Contours', m)
            # cv2.waitKey(0)
            if w < 10 and h < 10 and w * h < 50:
                is_stable = True

    return is_stable


if __name__ == '__main__':
    """
    this setting fits in 20201224_103338.mp4, images test/0.jpg, test/53.jpg
    """
    path = 'test'
    image = join(path, '53.jpg')

    start = time()
    image = cv2.imread(image)

    image = cv2.resize(image, (128, 48), interpolation=cv2.INTER_AREA)
    cv2.imwrite('save/resized.jpg', image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # input should be resized + gray image
    is_stable = stable_marker_detector(gray)
    print('is_stable = {}'.format(is_stable))

    show_image = True
    digits = read_digits(gray, image, skew=7, avg_digit_width=12, debug=show_image)
    # assert number -1
    if -1 in digits:
        print("Error: please put your handset upright and capture the scale image again.")
        results = 0
    else:
        results = float(''.join(map(str, digits)))
        with open('weight.json', 'w') as scale_read:
            json.dump(results, scale_read)
        print('{:.3f}, using {} ms'.format(results, (time()-start)*1000))
