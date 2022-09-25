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


dir = 16
total_time = 0

waiting_time  = 1
for i in range(1):
    # str = f'{dir}/{dir} by {dir} orthogonal maze ({i}).png'
    str = "16x16_0_w2_T1.png"
    frame_maze = cv.imread(str)
    cv.namedWindow("roi_maze")

    cv.createTrackbar("size", "roi_maze", 3, 50, nothing)

    while True:
    # 图像处理
        frame_maze_clone = frame_maze.copy()
        frame_maze_clone2 = frame_maze.copy()
        value = 100
        t1 = cv.getTickCount()

        # 第一步将图片转换为灰度
        frame_gray = cv.cvtColor(frame_maze_clone, cv.COLOR_BGR2GRAY)
        cv.imshow("1 gray", frame_gray)
        cv.waitKey(waiting_time)

        # 第二步将灰度图片进行滤波处理
        frame_blur = cv.medianBlur(frame_gray, 3)
        cv.imshow("2 frame_blur", frame_blur)
        cv.waitKey(waiting_time)

        # 第三步将滤波处理后的图片转换为二值化图片，此时图片已经颜色翻转了
        ret, frame_edge = cv.threshold(frame_blur, value, 255, cv.THRESH_BINARY_INV)
        cv.imshow("3 frame_edge", frame_edge)
        # cv.imwrite("16X16_0_w8_EZ.png",frame_edge)
        cv.waitKey(waiting_time)

        # 第四步，针对颜色反转后的图片进行轮廓查询
        contours, hierarchy = cv.findContours(frame_edge, cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
        contours_process = []
        cv.drawContours(frame_maze_clone, contours, -1, (0, 0, 0), 2, cv.LINE_AA)
        cv.imshow("4 FZDC", frame_maze_clone)
        # cv.imwrite("20 cells diameter theta maze T2 w4.png",frame_maze_clone)
        cv.waitKey(waiting_time)


        # 第五步，筛选轮廓
        for contour in contours:
            if cv.contourArea(contour) > 50 and contour_diff(contour)[0] > 50 and contour_diff(contour)[1] > 50:
                contours_process.append(contour)

        # 第六步 提取迷宫区域，并在frame_maze_clone上绘制了一个矩形框
        roi_point = calcROI(contours_process)
        x, y, w, h = cv.boundingRect(roi_point)
        cv.rectangle(frame_maze_clone, (x, y), (x + w, y + h), (0, 255, 0), 2,cv.LINE_AA)

        # 第七步 制作一张全黑的与提取魔方图片一样大小的图片，并将轮廓1，以红色及线宽为1的形式，回执绘制到draw上
        draw = np.zeros(frame_maze.shape[:2], dtype=np.float64)
        cv.drawContours(draw, contours_process, 0, (255, 0, 0), 1)
        cv.imshow("7 draw",draw)
        cv.waitKey(waiting_time)


        # 第八步，从迷宫原图上提取出迷宫的有效部分
        roi_maze = frame_maze_clone2[y:y + h, x:x + w]
        cv.imshow("8.1 roi_maze",roi_maze)
        cv.waitKey(waiting_time)
        roi = draw[y:y + h, x:x + w]
        cv.imshow("8.2 roi",roi)
        cv.waitKey(waiting_time)

        # 第九步 求解迷宫路径
        # 9.1 创建卷积核
        # kernelSize = 11
        kernelSize = cv.getTrackbarPos("size", "roi_maze")
        kernel = cv.getStructuringElement(cv.MORPH_RECT, (kernelSize, kernelSize))
        # print(kernel)
        # 9.2 膨胀操作
        dilated = cv.dilate(roi, kernel, iterations=2)
        cv.imshow("9.1 dilated",dilated)
        cv.waitKey(waiting_time)

        #9.3 腐蚀操作
        eroded = cv.erode(dilated, kernel, iterations=1)
        cv.imshow("9.2 eroded", eroded)
        cv.waitKey(waiting_time)

        #9.4 异或操作
        diff = cv.absdiff(dilated, eroded)
        diff2 = np.uint8(diff)
        cv.imshow("9.4 diff2", diff2)
        cv.waitKey(waiting_time)

        # 9.5 提取异或操作所得图像的轮廓，往roi_maze上绘制轮廓
        contours_result, hierarchy = cv.findContours(diff2, cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
        cv.drawContours(roi_maze, contours_result, -1, (255, 150, 255), -1)
        # drawContours的用法，要绘制线宽的图，绘制哪些轮廓，-1出表示轮廓的编号，此处-1为所有轮廓，颜色，-1表示线宽-1表示为全部填充
        cv.imshow("roi_maze", roi_maze)
        # cv.imwrite("result.png",roi_maze)
        cv.waitKey(waiting_time)


        frame_time = (cv.getTickCount() - t1) / cv.getTickFrequency()
        print(frame_time)
        total_time += frame_time

    # cv.imwrite("bbb.png",roi_maze)
print(f"total_time:{total_time/101}")
cv.destroyAllWindows()
print("position 1")