import sys
import cv2
import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# import tensorflow as tf
import time
import numpy as np
from imutils import contours
from skimage import measure
import imutils
import math
def skew_bottom(thresh, image):
    coords = np.column_stack(np.where(thresh > 0))
    angle = cv2.minAreaRect(coords)[-1]
    # the `cv2.minAreaRect` function returns values in the
    # range [-90, 0); as the rectangle rotates clockwise the
    # returned angle trends to 0 -- in this special case we
    # need to add 90 degrees to the angle
    if angle < -45:
        angle = -(90 + angle)
    # otherwise, just take the inverse of the angle to make
    # it positive
    else:
        angle = -angle

    # rotate the image to deskew it
    (h, w) = thresh.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(thresh, M, (w, h),
        flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    image = cv2.warpAffine(image, M, (w, h),
                             flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    print(angle)
    return rotated, image


# def load_tf_model(*args):
#     start_time = time.time()
#     # Load saved model and build the detection function
#     detect_fn = tf.saved_model.load(args[0])
#     category_index = ["ROI"]
#     end_time = time.time()
#     elapsed_time = end_time - start_time
#     return [detect_fn, category_index, elapsed_time]

# def order_points(pts):
# 	# initialzie a list of coordinates that will be ordered
# 	# such that the first entry in the list is the top-left,
# 	# the second entry is the top-right, the third is the
# 	# bottom-right, and the fourth is the bottom-left
# 	rect = np.zeros((4, 2), dtype = "float32")
# 	# the top-left point will have the smallest sum, whereas
# 	# the bottom-right point will have the largest sum
# 	s = pts.sum(axis = 1)
# 	rect[0] = pts[np.argmin(s)]
# 	rect[2] = pts[np.argmax(s)]
# 	# now, compute the difference between the points, the
# 	# top-right point will have the smallest difference,
# 	# whereas the bottom-left will have the largest difference
# 	diff = np.diff(pts, axis = 1)
# 	rect[1] = pts[np.argmin(diff)]
# 	rect[3] = pts[np.argmax(diff)]
# 	# return the ordered coordinates
# 	return rect
#
#
# def four_point_transform(image, pts):
# 	# obtain a consistent order of the points and unpack them
# 	# individually
# 	rect = order_points(pts)
# 	(tl, tr, br, bl) = rect
# 	# compute the width of the new image, which will be the
# 	# maximum distance between bottom-right and bottom-left
# 	# x-coordiates or the top-right and top-left x-coordinates
# 	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
# 	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
# 	maxWidth = max(int(widthA), int(widthB))
# 	# compute the height of the new image, which will be the
# 	# maximum distance between the top-right and bottom-right
# 	# y-coordinates or the top-left and bottom-left y-coordinates
# 	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
# 	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
# 	maxHeight = max(int(heightA), int(heightB))
# 	# now that we have the dimensions of the new image, construct
# 	# the set of destination points to obtain a "birds eye view",
# 	# (i.e. top-down view) of the image, again specifying points
# 	# in the top-left, top-right, bottom-right, and bottom-left
# 	# order
# 	dst = np.array([
# 		[0, 0],
# 		[maxWidth - 1, 0],
# 		[maxWidth - 1, maxHeight - 1],
# 		[0, maxHeight - 1]], dtype = "float32")
# 	# compute the perspective transform matrix and then apply it
# 	M = cv2.getPerspectiveTransform(rect, dst)
# 	warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
# 	# return the warped image
# 	return warped
#
#
#
# def extract_y(thresh, kernel):
#     kernel_y = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel,1))
#     thresh_frame_y = cv2.morphologyEx(thresh, cv2.MORPH_DILATE, kernel_y)
#
#     hist = cv2.reduce(thresh_frame_y, 1, cv2.REDUCE_AVG).reshape(-1)
#
#     th = 2
#     H,W = thresh_frame_y.shape[:2]
#     uppers = [y for y in range(H-1) if hist[y]<=th and hist[y+1]>th]
#     lowers = [y for y in range(H-1) if hist[y]>th and hist[y+1]<=th]
#
#     return uppers, lowers
#
#
# def extract_x(thresh, kernel):
#     kernel_x = cv2.getStructuringElement(cv2.MORPH_RECT, (1,kernel))
#     thresh_frame_x = cv2.morphologyEx(thresh, cv2.MORPH_DILATE, kernel_x)
#
#
#
#
#     hist = cv2.reduce(thresh_frame_x, 0, cv2.REDUCE_AVG).reshape(-1)
#
#     th = 2
#
#     H,W = thresh_frame_x.shape[:2]
#     uppers = [x for x in range(W-1) if hist[x] <= th < hist[x + 1]]
#     lowers = [x for x in range(W-1) if hist[x] > th >= hist[x + 1]]
#     return uppers, lowers

def send_odd_ev(COM, patt,high,low):

    ser = serial.Serial(
        port=COM,
        baudrate=115200,
        parity=serial.PARITY_NONE,
        stopbits=serial.STOPBITS_ONE,
        bytesize=serial.EIGHTBITS
    )

    ser.isOpen()
    ser.write(str(f"patt {patt} {high} {low}" + os.linesep).encode())
    ser.close()


def led_off(*args):
    try:
        org = args[0][0].copy()
        result = True
        # args[0][0] = skew_bottom(args[0][0])
        gray = cv2.cvtColor(org, cv2.COLOR_BGR2GRAY)
        thresh_frame = cv2.threshold(gray, int(args[0][1]), 255, cv2.THRESH_BINARY)[1]
        thresh_frame = cv2.erode(thresh_frame, None, iterations=1)

        # thresh_frame, args[0][0] = skew_bottom(thresh_frame, args[0][0])

        # thresh_frame = cv2.dilate(thresh_frame, None, iterations=2)
        # plt.imshow(thresh_frame)
        # plt.show()
        cntrs = cv2.findContours(thresh_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cntrs = cntrs[0] if len(cntrs) == 2 else cntrs[1]
        led_count = len(cntrs)
        #Count leds

        if len(cntrs) > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (int(args[0][3]), int(args[0][3])))
            thresh_frame_roi = cv2.morphologyEx(thresh_frame, cv2.MORPH_DILATE, kernel)
            cntrs = cv2.findContours(thresh_frame_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            cntrs = cntrs[0] if len(cntrs) == 2 else cntrs[1]
            areas = [cv2.contourArea(c) for c in cntrs]

            max_index = np.argmax(areas)
            cnt = cntrs[max_index]
            hull = cv2.convexHull(cnt)

            cv2.drawContours(thresh_frame_roi, [hull], 0, (255, 255, 255), int(args[0][4]))

            cntrs = cv2.findContours(thresh_frame_roi, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cntrs = cntrs[0] if len(cntrs) == 2 else cntrs[1]

            if len(cntrs) > 0:

                areas = [cv2.contourArea(c) for c in cntrs]
                max_index = np.argmax(areas)

                for i, c in enumerate(cntrs):
                    print(i, max_index, cv2.contourArea(c))
                    if i == max_index:
                        x, y, w, h = cv2.boundingRect(c)
                        # cv2.drawContours(args[0][0], [c], -1, (255, 0, 0), 3)

                        cv2.rectangle(org, (x-10, y-10), (x + w+10, y + h+10), (255, 0, 0), 3)

                        cv2.putText(org, f'DETECTED: {led_count}/ { int(args[0][2])} LEDS', (x, y - 100), cv2.FONT_HERSHEY_SIMPLEX, 1.8,
                                    (255, 0, 0), 7)
                        # #testing
                        # M = math.floor(h // 2)
                        # N = math.floor(w // 4)
                        #
                        # for yy in range(y, y+h-M+10, M):
                        #     for xx in range(x, w+x-N+10, N):
                        #         y1 = yy + M
                        #         x1 = xx + N
                        #         cv2.rectangle(org, (xx, yy), (x1, y1), (0, 255, 0), 3)
                        #
                        # # end testing


                    else:

                        cv2.drawContours(org, [c], -1, (0, 0, 255), 3)
                        result = False
        else:
            result = False
            cv2.putText(org, f'DETECTED: {led_count}/{ int(args[0][2])} LEDS', (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.8,
                        (255, 0, 0), 7)

        if result is True:
            cv2.putText(org, "OK", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 255, 0), 2, cv2.LINE_AA)
        else:
            cv2.putText(org, "NG", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 255),
                        2, cv2.LINE_AA)
        return org, result

    except Exception as e:
        print(e)


def curv_led_off(*args):
    try:
        org = args[0][0].copy()
        result = True
        gray = cv2.cvtColor(org, cv2.COLOR_BGR2GRAY)
        thresh_frame = cv2.threshold(gray, int(args[0][1]), 255, cv2.THRESH_BINARY)[1]
        thresh_frame = cv2.erode(thresh_frame, None, iterations=1)

        cntrs = cv2.findContours(thresh_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cntrs = cntrs[0] if len(cntrs) == 2 else cntrs[1]
        led_count = len(cntrs)
        #Count leds

        if len(cntrs) > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (int(args[0][3]), int(args[0][3])))
            thresh_frame_roi = cv2.morphologyEx(thresh_frame, cv2.MORPH_DILATE, kernel)
            cntrs = cv2.findContours(thresh_frame_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            cntrs = cntrs[0] if len(cntrs) == 2 else cntrs[1]
            areas = [cv2.contourArea(c) for c in cntrs]

            max_index = np.argmax(areas)
            cnt = cntrs[max_index]
            hull = cv2.convexHull(cnt)

            cv2.drawContours(thresh_frame_roi, [hull], 0, (255, 255, 255), int(args[0][4]))

            cntrs = cv2.findContours(thresh_frame_roi, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cntrs = cntrs[0] if len(cntrs) == 2 else cntrs[1]

            if len(cntrs) > 0:

                areas = [cv2.contourArea(c) for c in cntrs]
                max_index = np.argmax(areas)


                for i, c in enumerate(cntrs):
                    if i == max_index:
                        x, y, w, h = cv2.boundingRect(c)
                        # cv2.drawContours(args[0][0], [c], -1, (255, 0, 0), 3)
                        cv2.rectangle(org, (x-10, y-10), (x + w+10, y + h+10), (255, 0, 0), 3)

                        cv2.putText(org, f'DETECTED: {led_count}/ { int(args[0][2])} LEDS', (x, y - 100), cv2.FONT_HERSHEY_SIMPLEX, 1.8,
                                    (255, 0, 0), 7)
                        # #testing
                        # M = math.floor(h // 2)
                        # N = math.floor(w // 4)
                        #
                        # for yy in range(y, y+h-M+10, M):
                        #     for xx in range(x, w+x-N+10, N):
                        #         y1 = yy + M
                        #         x1 = xx + N
                        #         cv2.rectangle(org, (xx, yy), (x1, y1), (0, 255, 0), 3)
                        #
                        # # end testing

                    else:
                        # x, y, w, h = cv2.boundingRect(c)
                        # xh, yh, wh, hh = cv2.boundingRect(hull)
                        if led_count < int(args[0][2]):
                            # if w < 0.3*wh and h > 0.3*int(args[0][3]):
                                cv2.drawContours(org, [c], -1, (0, 0, 255), 3)
                                result = False
        else:
            result = False
            cv2.putText(org, f'DETECTED: {led_count}/{ int(args[0][2])} LEDS', (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.8,
                        (255, 0, 0), 7)

        if result is True:
            cv2.putText(org, "OK", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 255, 0), 2, cv2.LINE_AA)
        else:
            cv2.putText(org, "NG", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 255),
                        2, cv2.LINE_AA)
        return org, result

    except Exception as e:
        print(e)


def led_dim(*args):
    try:
        org = args[0][0].copy()
        result = True
        gray = cv2.cvtColor(org, cv2.COLOR_BGR2GRAY)
        thresh_frame = cv2.threshold(gray, int(args[0][1]), 255, cv2.THRESH_BINARY)[1]
        thresh_frame = cv2.erode(thresh_frame, None, iterations=1)
        cntrs = cv2.findContours(thresh_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cntrs = cntrs[0] if len(cntrs) == 2 else cntrs[1]
        led_count = len(cntrs)
        #Count leds

        if len(cntrs) > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (int(args[0][3]), int(args[0][3])))
            thresh_frame_roi = cv2.morphologyEx(thresh_frame, cv2.MORPH_DILATE, kernel)
            cntrs = cv2.findContours(thresh_frame_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cntrs = cntrs[0] if len(cntrs) == 2 else cntrs[1]
            areas = [cv2.contourArea(c) for c in cntrs]

            max_index = np.argmax(areas)
            cnt = cntrs[max_index]
            hull = cv2.convexHull(cnt)

            cv2.drawContours(thresh_frame_roi, [hull], 0, (255, 255, 255), int(args[0][4]))

            cntrs = cv2.findContours(thresh_frame_roi, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cntrs = cntrs[0] if len(cntrs) == 2 else cntrs[1]

            if len(cntrs) > 0:

                areas = [cv2.contourArea(c) for c in cntrs]
                max_index = np.argmax(areas)

                for i, c in enumerate(cntrs):
                    print(i, max_index, cv2.contourArea(c))
                    if i == max_index:
                        x, y, w, h = cv2.boundingRect(c)
                        cv2.rectangle(org, (x-10, y-10), (x + w+10, y + h+10), (255, 0, 0), 3)

                        cv2.putText(org, f'DETECTED: {led_count}/ { int(args[0][2])} LEDS', (x, y - 100), cv2.FONT_HERSHEY_SIMPLEX, 1.8,
                                    (255, 0, 0), 7)
                        # #testing
                        # M = math.floor(h // 2)
                        # N = math.floor(w // 4)
                        #
                        # for yy in range(y, y+h-M+10, M):
                        #     for xx in range(x, w+x-N+10, N):
                        #         y1 = yy + M
                        #         x1 = xx + N
                        #         cv2.rectangle(org, (xx, yy), (x1, y1), (0, 255, 0), 3)
                        #
                        # # end testing


                    else:
                        if led_count <= int(args[0][2]) - int(args[0][5]):
                            cv2.drawContours(org, [c], -1, (0, 0, 255), 3)
                            result = False
        else:
            result = False
            cv2.putText(org, f'DETECTED: {led_count}/{ int(args[0][2])} LEDS', (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.8,
                        (255, 0, 0), 7)

        if result is True:
            cv2.putText(org, "OK", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 255, 0), 2, cv2.LINE_AA)
        else:
            cv2.putText(org, "NG", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 255),
                        2, cv2.LINE_AA)
        return org, result

    except Exception as e:
        print(e)


def led_dim_off(*args):
    try:
        org = args[0][0].copy()
        dim, off = True, True
        off_pos = list()
        result = dict()

        gray = cv2.cvtColor(org, cv2.COLOR_BGR2GRAY)



        #THRESHOLD OFF
        thresh_frame = cv2.threshold(gray, int(args[0][1]), 255, cv2.THRESH_BINARY)[1]

        thresh_frame = cv2.erode(thresh_frame, None, iterations=1)

        cntrs = cv2.findContours(thresh_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cntrs = cntrs[0] if len(cntrs) == 2 else cntrs[1]
        led_count = len(cntrs)
        #Count leds

        if len(cntrs) > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (int(args[0][3]), int(args[0][3])))
            thresh_frame_roi = cv2.morphologyEx(thresh_frame, cv2.MORPH_DILATE, kernel)
            cntrs = cv2.findContours(thresh_frame_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cntrs = cntrs[0] if len(cntrs) == 2 else cntrs[1]

            areas = [cv2.contourArea(c) for c in cntrs]

            area = []
            for c in cntrs:
                area.append(cv2.contourArea(c))

            max_index = np.argmax(areas)
            cnt = cntrs[max_index]
            hull = cv2.convexHull(cnt)

            cv2.drawContours(thresh_frame_roi, [hull], 0, (255, 255, 255), int(args[0][4]))

            cntrs = cv2.findContours(thresh_frame_roi, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cntrs = cntrs[0] if len(cntrs) == 2 else cntrs[1]

            if len(cntrs) > 0:

                areas = [cv2.contourArea(c) for c in cntrs]
                max_index = np.argmax(areas)

                for i, c in enumerate(cntrs):
                    if i == max_index:
                        x, y, w, h = cv2.boundingRect(c)
                        cv2.rectangle(org, (x-10, y-10), (x + w+10, y + h+10), (255, 0, 0), 3)
                        cv2.putText(org, f'DETECTED: {led_count}/ { int(args[0][2])} LEDS', (x, y - 100), cv2.FONT_HERSHEY_SIMPLEX, 1.8,
                                    (255, 0, 0), 7)
                        # #testing
                        #                         # M = math.floor(h // 2)
                        #                         # N = math.floor(w // 4)
                        #                         #
                        #                         # for yy in range(y, y+h-M+10, M):
                        #                         #     for xx in range(x, w+x-N+10, N):
                        #                         #         y1 = yy + M
                        #                         #         x1 = xx + N
                        #                         #         cv2.rectangle(org, (xx, yy), (x1, y1), (0, 255, 0), 3)
                        #                         #
                        #                         # # end testing
                    else:
                        off_pos.append(c)
                        cv2.drawContours(org, [c], -1, (0, 0, 255), 3)
                        off = False

        else:
            off = False
            cv2.putText(org, f'DETECTED: {led_count}/{ int(args[0][2])} LEDS', (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.8,
                        (255, 0, 0), 7)

        # DIMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM
        thresh_frame = cv2.threshold(gray, int(args[0][6]), 255, cv2.THRESH_BINARY)[1]
        thresh_frame = cv2.erode(thresh_frame, None, iterations=1)

        cntrs = cv2.findContours(thresh_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cntrs = cntrs[0] if len(cntrs) == 2 else cntrs[1]
        led_count = len(cntrs)
        print(led_count)
        # Count leds

        if len(cntrs) > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (int(args[0][3]), int(args[0][3])))
            thresh_frame_roi = cv2.morphologyEx(thresh_frame, cv2.MORPH_DILATE, kernel)
            cntrs = cv2.findContours(thresh_frame_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cntrs = cntrs[0] if len(cntrs) == 2 else cntrs[1]
            areas = [cv2.contourArea(c) for c in cntrs]

            max_index = np.argmax(areas)
            cnt = cntrs[max_index]
            hull = cv2.convexHull(cnt)

            cv2.drawContours(thresh_frame_roi, [hull], 0, (255, 255, 255), int(args[0][4]))

            cntrs = cv2.findContours(thresh_frame_roi, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cntrs = cntrs[0] if len(cntrs) == 2 else cntrs[1]

            if len(cntrs) > 0:

                areas = [cv2.contourArea(c) for c in cntrs]
                max_index = np.argmax(areas)

                for i, c in enumerate(cntrs):
                    if i == max_index:
                        x, y, w, h = cv2.boundingRect(c)
                        cv2.rectangle(org, (x - 10, y - 10), (x + w + 10, y + h + 10), (255, 0, 0), 3)
                        cv2.putText(org, f'DETECTED: {led_count}/ {int(args[0][2])} LEDS', (x, y - 100),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.8,
                                    (255, 0, 0), 7)
                    else:
                        x, y, w, h = cv2.boundingRect(c)

                        if len(off_pos) > 0:
                            for o in off_pos:
                                ox, oy, ow, oh = cv2.boundingRect(o)
                                print(x, y, w, h, ox, oy, ow, oh)
                                if abs(ox - x) >= int(args[0][3]) or abs(oy - y) >= int(args[0][3]):
                                    if led_count <= int(args[0][2]) - int(args[0][5]):
                                        cv2.drawContours(org, [c], -1, (255, 0, 255), 3)
                                        dim = False
                        else:
                            if led_count <= int(args[0][2]) - int(args[0][5]):
                                cv2.drawContours(org, [c], -1, (255, 0, 255), 3)
                                dim = False


            else:
                dim = False
                cv2.putText(org, f'DETECTED: {led_count}/{int(args[0][2])} LEDS', (100, 100), cv2.FONT_HERSHEY_SIMPLEX,
                            1.8,
                            (255, 0, 0), 7)

        if dim is True and off is True:
            cv2.putText(org, "OK", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 255, 0), 2, cv2.LINE_AA)

        if dim is False:
            cv2.putText(org, "NG DIM", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 255),
                        2, cv2.LINE_AA)
        if off is False:
            cv2.putText(org, "NG OFF", (500, 150), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 255),
                        2, cv2.LINE_AA)

        result['off'] = off
        result['dim'] = dim
        return org, result

    except Exception as e:
        print(e)


def odd_even(*args):
    try:
        org = args[0][0].copy()
        result = True
        # args[0][0] = skew_bottom(args[0][0])
        gray = cv2.cvtColor(org, cv2.COLOR_BGR2GRAY)
        thresh_frame = cv2.threshold(gray, int(args[0][1]), 255, cv2.THRESH_BINARY)[1]
        thresh_frame_org = thresh_frame
        # thresh_frame = cv2.erode(thresh_frame, None, iterations=1)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (int(args[0][2]), int(args[0][2])))
        thresh_frame = cv2.morphologyEx(thresh_frame, cv2.MORPH_DILATE, kernel)
        # thresh_frame, args[0][0] = skew_bottom(thresh_frame, args[0][0])


        cntrs = cv2.findContours(thresh_frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cntrs = cntrs[0] if len(cntrs) == 2 else cntrs[1]

        if len(cntrs) > 0:
            areas = [cv2.contourArea(c) for c in cntrs]
            max_index = np.argmax(areas)
            cnt = cntrs[max_index]
            hull = cv2.convexHull(cnt)
            cv2.drawContours(thresh_frame, [hull], 0, (255, 255, 255), int(args[0][3]))

            # cv2.imshow(r'lll',thresh_frame)
            # cv2.waitKey()
            cntrs = cv2.findContours(thresh_frame, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            cntrs = cntrs[0] if len(cntrs) == 2 else cntrs[1]

            if len(cntrs) > 0:
                areas = [cv2.contourArea(c) for c in cntrs]
                max_index = np.argmax(areas)
                cntrs = cntrs[:max_index]+cntrs[max_index+1:]
                for c in cntrs:
                    x, y, w, h = cv2.boundingRect(c)
                    # cv2.rectangle(image, (x, y), (x+w, y+h), (255,255,255), 2)
                    crop_img = thresh_frame_org[y:y + h, x:x + w]
                    # cv2.imshow(r'fff', crop_img)
                    # cv2.waitKey()
                    cntrs = cv2.findContours(crop_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                    cntrs = cntrs[0] if len(cntrs) == 2 else cntrs[1]

                    if len(cntrs) > 0:
                        cv2.rectangle(org, (x-10, y-10), (x + w+10, y + h+10), (0, 0, 255), 3)
                        result = False

                    elif w < int(args[0][4]) or h < int(args[0][5]):
                        cv2.rectangle(org, (x-10, y-10), (x + w+10, y + h+10), (0, 0, 255), 3)
                        result = False

        else:
            result = False
            cv2.putText(org, f'DETECTED: 0 MATRIX', (100, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.8,
                        (255, 0, 0), 7)

        if result is True:
            cv2.putText(org, "OK", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 255, 0), 2, cv2.LINE_AA)
        else:
            cv2.putText(org, "NG", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 255),
                        2, cv2.LINE_AA)

        return org, result

    except Exception as e:
        print(e)


def white_spot(*args):
    try:
        org = args[0][0].copy()
        result = True
        gray = cv2.cvtColor(org, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, int(args[0][1]), 255, cv2.THRESH_BINARY)[1]
        # erosions va dilations hinh
        # xoa noise khoi hinh
        thresh = cv2.erode(thresh, None, iterations=1)
        thresh = cv2.dilate(thresh, None, iterations=2)
        labels = measure.label(thresh, neighbors=8, background=0)
        mask = np.zeros(thresh.shape, dtype="uint8")

        for label in np.unique(labels):
            # ignore label cua background
            if label == 0:
                continue
            # tao mask va dem pixel cua mask
            labelMask = np.zeros(thresh.shape, dtype="uint8")
            labelMask[labels == label] = 255
            numPixels = cv2.countNonZero(labelMask)
            # neu du so pixel -> mask
            if numPixels > int(args[0][2]):
                mask = cv2.add(mask, labelMask)
        # kiem contour tu trai qua phai
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)

        cnts = imutils.grab_contours(cnts)
        if 15 > len(cnts) > 0:
            cnts = contours.sort_contours(cnts)[0]
            for (i, c) in enumerate(cnts):
                # ve hinh tron
                (x, y, w, h) = cv2.boundingRect(c)
                ((cX, cY), radius) = cv2.minEnclosingCircle(c)
                cv2.circle(org, (int(cX), int(cY)), int(radius + 10),
                           (0, 0, 255), 3)
                cv2.putText(org, "{}".format(i + 1), (x, y - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)

                cv2.putText(org, "NG - Found {} ".format(len(cnts)), (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 255),
                            2, cv2.LINE_AA)
                result = False
        elif len(cnts) > 15:
            cv2.putText(org, "NG - Found more than 15", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 255),
                        2, cv2.LINE_AA)

            cv2.putText(org, "Kiem Tra Lai Model, SPECs! ", (10, 300), cv2.FONT_HERSHEY_SIMPLEX, 4,
                        (0, 0, 255),
                        2, cv2.LINE_AA)
            result = False
        # else:
        #     cv2.putText(org, "OK - Found 0 ", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 4,
        #                 (0, 255, 0), 2, cv2.LINE_AA)
        #     result = True
        if result is True:
            cv2.putText(org, "OK", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 255, 0), 2, cv2.LINE_AA)
        else:
            cv2.putText(org, "NG", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 255),
                        2, cv2.LINE_AA)

        return org, result
    except Exception as e:
        print(e)


def diff_color(*args):
    try:
        org = args[0][0].copy()
        result = True
        frame_HSV = cv2.cvtColor(org, cv2.COLOR_BGR2HSV)
        frame_threshold = cv2.inRange(frame_HSV, (0, 0, int(args[0][1])), (255, 255, 255))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(args[0][2]), int(args[0][2])))
        closing = cv2.morphologyEx(frame_threshold, cv2.MORPH_CLOSE, kernel)
        kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (int(args[0][3]), int(args[0][3])))

        dil = cv2.dilate(~closing, kernel2, iterations=1)

        dil_inv = ~dil
        cntrs = cv2.findContours(dil_inv, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cntrs = cntrs[0] if len(cntrs) == 2 else cntrs[1]

        if len(cntrs) > 0:
            areas = [cv2.contourArea(c) for c in cntrs]
            max_index = np.argmax(areas)
            cnt = cntrs[max_index]
            hull = cv2.convexHull(cnt)
            cv2.drawContours(dil, [hull], 0, (0, 0, 0), 10)

            for index in range(len(cntrs)):
                if index != max_index and cv2.contourArea(cntrs[index]) > int(args[0][4]):
                    cv2.drawContours(org, cntrs[index], -1, (0, 0, 255), 3)
                    result = False

        if result is True:
            cv2.putText(org, "OK", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 255, 0), 2, cv2.LINE_AA)
        else:
            cv2.putText(org, "NG", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 255),
                        2, cv2.LINE_AA)

        return org, result
    except Exception as e:
        print(e)


def wait(*args):
    try:
        print(args[0][1])
        time.sleep(int(args[0][1]))
        return args[0][0], True
    except Exception as e:
        print(e)





# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# tf.get_logger().setLevel('ERROR')  # Suppress TensorFlow logging (2)
inspection_modes = {"LED_OFF": led_off, "ODD_EVEN": odd_even, "WAIT": wait, "WHITE_SPOTS": white_spot,
                    "DIFFIRENT_COLOR": diff_color,'LED_DIM':led_dim,'LED_DIM_OFF':led_dim_off,"CURV_LED_OFF":curv_led_off}
