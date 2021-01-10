"""
https://www.pyimagesearch.com/2017/07/03/installing-tesseract-for-ocr/
"""
from imutils.perspective import four_point_transform, order_points
from imutils import contours
import imutils
import cv2
import numpy as np
from os.path import join
from utils import stable_marker_detector

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
show_image = False

def _show_image(win, img, destroy=True):
    cv2.imshow(win, img)
    cv2.waitKey(0)
    if destroy:
        cv2.destroyWindow(win)

# load the example image
path = 'test'
image = join(path, '4.jpg')
image = cv2.imread(image)
image = cv2.resize(image, (128, 48), interpolation=cv2.INTER_AREA)
original = image.copy()
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
"""
detect stable marker
"""
# is_table = stable_marker_detector(gray)

"""
if stable marker is detected
"""
gray_orig = gray.copy()
cv2.imwrite("save/gray_orig.png", gray_orig)
# blurred = cv2.GaussianBlur(gray, (5, 5), 0)
# edged = cv2.Canny(blurred, 50, 200, 255)
# cv2.imwrite("save/edge.png", edged)

_, gray = cv2.threshold(gray, 100, 255, 0)  # extract white area
cv2.imwrite("save/gray.png", gray)
gray_inv = cv2.bitwise_not(gray)
cv2.imwrite("save/gray_inv.png", gray_inv)

# to locate white background & black digits
cnts = cv2.findContours(gray_inv, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
largest_area = sorted(cnts, key=cv2.contourArea)[-1]
mask = np.zeros(image.shape, np.uint8)  # all black in mask
cv2.drawContours(mask, [largest_area], 0, (255, 255, 255), -1)  # make roi area white in mask
dst = cv2.bitwise_and(image, mask)  # pick roi from image
cv2.imwrite("save/dst.png", dst)
output = dst
roi = dst.copy()
roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
cv2.imwrite("save/roi_gray.png", roi_gray)
_, _roi_gray = cv2.threshold(roi_gray, 100, 255, 0)  # white digits
cv2.imwrite("save/_roi_gray.png", _roi_gray)

"""
Step #3: Extract the digit regions
"""
thresh = cv2.threshold(_roi_gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
cv2.imwrite("save/thresh1.png", thresh)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
cv2.imwrite("save/thresh2.png", thresh)

cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
(x, y, w, h) = cv2.boundingRect(cnts[0])
roi_small = thresh[y:y + h, x:x + w]
cv2.imwrite("save/roi.png", roi_small)
warped = roi_small.copy()

# the numbers displayed a little bit leaning to right side we make them upright
height, width = warped.shape
width -= 1
height -= 1

skew = 5

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

# calculate the perspective transform matrix and warp the perspective to grab the screen
M = cv2.getPerspectiveTransform(rect, dst)
warped = cv2.warpPerspective(warped, M, (width + 1, height + 1))
output = cv2.warpPerspective(output, M, (width + 1, height + 1))

# height, width = warped.shape
# width -= 1
# height -= 1
# skew += 2
# warped = warped[skew:height, skew:width - skew]
# output = output[skew:height, skew:width - skew]

# segment 2 and segment 5 separated
# so we do vertical dilate and erode to connect them
# todo, number 0 and 1 need to be tuned
#  change ksize to (1, 3), from (1, 2)
vert_dilate3 = cv2.getStructuringElement(cv2.MORPH_RECT, ksize=(1, 3))
dilation = cv2.dilate(warped, vert_dilate3)
dilation = cv2.erode(warped, vert_dilate3)
cv2.imwrite("save/dilation.png", dilation)
dilation_inv = cv2.bitwise_not(dilation)
cv2.imwrite("save/dilation_inv.png", dilation_inv)

"""
Step #4: Identify the digits
"""

cnts = cv2.findContours(dilation_inv.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
print('{} contours found'.format(len(cnts)))
# todo if len(cnts)>4, pick up y != all other y, because y should be same to all digits, x increase by w
digitCnts = []

# loop over the digit area candidates
for _c, c in enumerate(cnts):
    # compute the bounding box of the contour
    (x, y, w, h) = cv2.boundingRect(c)
    # if the contour is sufficiently high, it must be a digit
    """
    Note: Determining the appropriate width and height constraints requires a few rounds of trial and error. 
    I would suggest looping over each of the contours, drawing them individually, and inspecting their dimensions. 
    Doing this process ensures you can find commonalities across digit contour properties.
    """
    # plot contours
    if show_image:
        temp_img = dilation_inv.copy()
        print("Contour {}: w={}, h={}, x={}, y={}".format(_c, w, h, x, y))
        cv2.drawContours(temp_img, cnts, _c, (255, 255, 255), 2)
        cv2.imshow('Contours', temp_img)
        cv2.waitKey(0)
    if 10 <= h <= 30 and w <= 30:
        if len(digitCnts) == 0:
            digitCnts.append(c)
            # print("Selected contour {}: w={}, h={}, x={}, y={}".format(_c, w, h, x, y))
        else:
            c_dup = 1
            for _digit in digitCnts:
                (_x, _y, _w, _h) = cv2.boundingRect(_digit)
                if (abs(x - _x) <= 2 and abs(y - _y) <= 2):
                    c_dup = 0
            if c_dup > 0:
                digitCnts.append(c)
                # print("Selected contour {}: w={}, h={}, x={}, y={}".format(_c, w, h, x, y))

# sort the contours from left-to-right, then initialize the actual digits themselves
digitCnts = contours.sort_contours(digitCnts, method="left-to-right")[0]
print("Found {} ROIs".format(len(digitCnts)))

digits = []
point = None
# loop over each of the digits
for _c, c in enumerate(digitCnts):
    # extract the digit ROI
    (x, y, w, h) = cv2.boundingRect(c)
    print("Selected contour {}: w={}, h={}, x={}, y={}".format(_c, w, h, x, y))
    # manually override the width of number 1
    if w < 9:  # each digit width = 12
        x -= 12 - w
        w = 12
        print("  changed contour : w={}, h={}, x={}, y={}".format(w, h, x, y))
    elif w > 13:
        w = 12
    #     point = _c
    roi = dilation_inv[y:y + h, x:x + w]
    # compute the approximate width and height of each segment based on the ROI dimensions.
    (roiH, roiW) = roi.shape
    (dW, dH) = (int(roiW * 0.34), int(roiH * 0.25))
    dHC = int(roiH * 0.1)
    print("roiH={}, roiW={}, dH={}, dW={}, dHC={}".format(roiH, roiW, dH, dW, dHC))

    # define the coordinates of set of 7 segments
    segments = [
        ((1, 0), (w, dH)),  # top
        ((1, 0), (dW, h // 2)),  # top-left
        ((w - dW, 0), (w-2, h // 2)),  # top-right
        ((4, (h // 2) - dHC), ((w // 2) + 1, (h // 2) + dHC)),  # center
        ((0, h // 2), (dW, h)),  # bottom-left
        ((w - dW-2, h // 2), (w, h)),  # bottom-right
        ((0, h - dH), (w-2, h))  # bottom
    ]
    on = [0] * len(segments)
    print("segments={}".format(segments))

    if show_image:
        _show_image("ROI {}".format(_c), roi, False)

    # loop over the segments
    for (i, ((xA, yA), (xB, yB))) in enumerate(segments):
        # extract the segment ROI, count the total number of thresholded pixels in the segment, and then compute
        # the area of the segment
        segROI = roi[yA:yB, xA:xB]
        total = cv2.countNonZero(segROI)  # (0, 0, 0)=black && (255, 255, 255)=white
        area = (xB - xA) * (yB - yA)
        # if the total number of non-zero pixels is greater than
        # 35% of the area, mark the segment as "on"
        print(i, total / float(area))
        if total / float(area) >= 0.5:
            on[i] = 1

        if show_image:
            # cv2.rectangle(roi, (xA, yA), (xB, yB), (255, 0, 0), 1)  # (0, 255, 0)=green
            # cv2.imshow("Segment {} of ROI {}".format(i, _c), roi)

            cv2.imshow("Segment {} of ROI {}".format(i, _c), segROI)
            cv2.waitKey(0)
            cv2.destroyWindow("Segment {} of ROI {}".format(i, _c))

    # print(on)
    # lookup the digit and draw it on the image
    # give -1 for lookup failure
    if tuple(on) in DIGITS_LOOKUP.keys():
        digit = DIGITS_LOOKUP[tuple(on)]
        print("@_@ Detected a digit = {}".format(digit))
    else:
        digit = -1
        print(on)
        cv2.imshow("ROI", roi)
    # print(digit)
    digits.append(digit)
    cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 1)  # (0, 255, 0)=green
    cv2.putText(output, str(digit), (x + 1, y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)  # (255, 0, 0)=red


# display the digits
print(digits)
cv2.imshow("Input", image)
cv2.imshow("Output", output)
cv2.waitKey(0)


