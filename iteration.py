import cv2 as cv
import numpy as np

def nothing(x):
    pass


def calcROI(contours):
    left = 10000000
    right = 0
    top = 10000000
    bottom = 0
    for cnt in contours:
        if cnt[cnt[:, 0, 0].argmin(), 0, 0] < left:
            left = cnt[cnt[:, 0, 0].argmin(), 0, 0]
            left_point = cnt[cnt[:, 0, 0].argmin(), 0]
        if cnt[cnt[:, :, 0].argmax()][0][0] > right:
            right = cnt[cnt[:, :, 0].argmax()][0][0]
            right_point = cnt[cnt[:, :, 0].argmax()][0]
        if cnt[cnt[:, :, 1].argmin()][0][1] < top:
            top = cnt[cnt[:, :, 1].argmin()][0][1]
            top_point = cnt[cnt[:, :, 1].argmin()][0]
        if cnt[cnt[:, :, 1].argmax()][0][1] > bottom:
            bottom = cnt[cnt[:, :, 1].argmax()][0][1]
            bottom_point = cnt[cnt[:, :, 1].argmax()][0]
    roi_point = np.array([left_point, right_point, top_point, bottom_point])
    return roi_point


def contour_diff(cont):
    max_x = 0
    min_x = 10000
    max_y = 0
    min_y = 10000
    for p in cont:
        if p[0][0] > max_x:
            max_x = p[0][0]
        if p[0][0] < min_x:
            min_x = p[0][0]
        if p[0][1] > max_y:
            max_y = p[0][1]
        if p[0][1] < min_y:
            min_y = p[0][1]
    return max_x - min_x, max_y - min_y

frame_maze = cv.imread('64 by 64 orthogonal maze.png')
cv.namedWindow("roi_maze")
cv.createTrackbar("dilate","roi_maze",1,100,nothing)
cv.createTrackbar("erode","roi_maze",1,100,nothing)

while True:
    # 图像处理
    frame_maze_clone = frame_maze.copy()
    frame_maze_clone2 = frame_maze.copy()
    value = 65
    t1 = cv.getTickCount()
    frame_gray = cv.cvtColor(frame_maze_clone, cv.COLOR_BGR2GRAY)
    frame_blur = cv.medianBlur(frame_gray, 3)
    ret, frame_edge = cv.threshold(frame_blur, value, 255, cv.THRESH_BINARY_INV)
    contours, hierarchy = cv.findContours(frame_edge, cv.RETR_EXTERNAL,
                                        cv.CHAIN_APPROX_SIMPLE)
    contours_process = []
    cv.drawContours(frame_maze_clone, contours, -1, (0, 0, 255), 1, cv.LINE_AA)
    # 筛选轮廓
    for contour in contours:
        if cv.contourArea(contour) > 100 and contour_diff(
                contour)[0] > 100 and contour_diff(contour)[1] > 100:
            contours_process.append(contour)
    # 提取迷宫区域
    roi_point = calcROI(contours_process)
    x, y, w, h = cv.boundingRect(roi_point)
    cv.rectangle(frame_maze_clone, (x, y), (x + w, y + h), (0, 255, 0), 2,
                cv.LINE_AA)
    draw = np.zeros(frame_maze.shape[:2], dtype=np.float64)
    cv.drawContours(draw, contours_process, 1, (255, 0, 0), 1)
    roi_maze = frame_maze_clone2[y:y + h, x:x + w]
    roi = draw[y:y + h, x:x + w]
    # 求解迷宫路径
    kernelSize = 11
    erode_time = cv.getTrackbarPos("erode","roi_maze")
    dilate_time = cv.getTrackbarPos("dilate","roi_maze")
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (kernelSize, kernelSize))
    dilated = cv.dilate(roi, kernel, iterations=dilate_time)
    eroded = cv.erode(dilated, kernel, iterations=erode_time)
    diff = cv.absdiff(dilated, eroded)
    diff2 = np.uint8(diff)
    contours_result, hierarchy = cv.findContours(diff2, cv.RETR_EXTERNAL,
                                                cv.CHAIN_APPROX_SIMPLE)
    cv.drawContours(roi_maze, contours_result, -1, (255, 150, 255), -1)
    # print((cv.getTickCount()-t1)/cv.getTickFrequency())
    cv.imshow("roi_maze", roi_maze)
    if cv.waitKey(1) == ord('s'):
        cv.imwrite("a.png", roi_maze)





