import cv2 as cv
import numpy as np
from queue import PriorityQueue


class Node:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.f = 0
        self.g = 0
        self.h = 0
        self.parent = None

    def init_node(self, parent, end):
        self.parent = parent
        if parent is not None:
            self.g = parent.g + 1
        else:
            self.g = 1
        self.h = abs(self.x - end.x) + abs(self.y - end.y)
        self.f = self.g + self.h

    def __lt__(self, other):  # operator <
        return self.f < other.f


def A_star(org, des):
    open_list = PriorityQueue()
    close_list = []
    open_list.put(org)
    while open_list.qsize() > 0:
        current_node = open_list.get()
        # print(current_node.x, current_node.y)
        close_list.append(current_node)
        # 找到当前节点的所有邻近节点
        for dir in dirs:
            newNode = Node(current_node.x + dir[0], current_node.y + dir[1])
            if (newNode.x == des.x) and (newNode.y == des.y):
                newNode.init_node(current_node, des)
                return newNode
            if Is_valid(newNode.x, newNode.y):
                newNode.init_node(current_node, des)
                mem[(current_node.x + dir[0], current_node.y + dir[1])] = 1
                open_list.put(newNode)
    return None


def Is_valid(x, y):
    if (x < 0 or x >= width) or (y < 0 or y >= height):
        return False

    if frame_maze[y, x] < 65:
        return False

    if (x, y) in mem.keys():
        return False
    return True


def findMazeStartPoint(frame):
    left = 0
    right = 0
    for index in range(frame.shape[1]):
        if frame[0, index] > 150:
            left = index
            break
    for index in range(frame.shape[1] - 1, 0, -1):
        if frame[0, index] > 150:
            right = index
            break
    mid = int((left + right) / 2)
    return Node(mid, 0)


def findMazeEndPoint(frame):
    left = 0
    right = 0
    for index in range(frame.shape[1]):
        if frame[frame.shape[0] - 1, index] > 150:
            left = index
            break
    for index in range(frame.shape[1] - 1, -1, -1):
        if frame[frame.shape[0] - 1, index] > 150:
            right = index
            break
    mid = int((left + right) / 2)
    return Node(mid, frame.shape[0] - 1)


# dir = 8
for dir in [16, 32, 64, 96, 128]:
    total_time = 0
    for i in range(101):
        str = f'{dir}/{dir} by {dir} orthogonal maze ({i}).png'
        frame_maze = cv.imread(str)
        frame_maze_clone = frame_maze.copy()
        frame_maze_clone2 = frame_maze.copy()
        frame_maze = cv.cvtColor(frame_maze, cv.COLOR_BGR2GRAY)
        height, width = frame_maze.shape

        org = findMazeStartPoint(frame_maze)
        des = findMazeEndPoint(frame_maze)

        dirs = [[-1, 0], [0, 1], [0, -1], [1, 0]]
        mem = {(org.x, org.y): 1}
        # 设置起始点和终点
        t1 = cv.getTickCount()
        result_node = A_star(org, des)
        total_time += (cv.getTickCount() - t1) / cv.getTickFrequency()
        print((cv.getTickCount() - t1) / cv.getTickFrequency())
        path = []
        while result_node is not None:
            path.append(result_node)
            result_node = result_node.parent
        path = reversed(path)
        contours = []
        for p in path:
            contours.append([[p.x, p.y]])
        contours = np.array(contours)
        cv.drawContours(frame_maze_clone2, contours, -1, (255, 150, 255), 2)
        # cv.polylines(frame_maze_clone2,[contours],False,(255,30,180),2)
        # cv.imshow("frame", frame_maze_clone2)
        path = f'result/Astar/{dir}/{dir} by {dir} orthogonal maze ({i}).png'
        cv.imwrite(path, frame_maze_clone2)
        cv.waitKey(0)
    print(f"average_time:{total_time / 101}")


def draw_circle(event, x, y, flags, param):
    if event == cv.EVENT_MOUSEMOVE:
        print(x, y)

# cv.namedWindow("mouse")
# cv.setMouseCallback("mouse", draw_circle)
# cv.imshow("mouse", frame_maze)
# cv.waitKey(0)
