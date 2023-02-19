import cv2
import numpy as np


def thresh_img(path):
    img = cv2.imread('path', -1)

    # 将图像转换为 HSV 颜色空间
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 定义蓝色和红色的 HSV 阈值范围
    blue_lower = np.array([100, 50, 50])
    blue_upper = np.array([130, 255, 255])
    red_lower = np.array([0, 50, 50])
    red_upper = np.array([20, 255, 255])

    # 对图像进行颜色分割，获取蓝色和红色区域的轮廓
    blue_mask = cv2.inRange(hsv, blue_lower, blue_upper)
    red_mask = cv2.inRange(hsv, red_lower, red_upper)
    blue_contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    red_contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, red_contours, -1, (255, 0, 0), 3)
    cv2.drawContours(img, blue_contours, -1, (0, 0, 255), 3)
    cv2.imshow("image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def maze_to_array(maze_path):
    # 读取图片
    maze = cv2.imread(maze_path, cv2.IMREAD_GRAYSCALE)
    # 将图像二值化
    _, maze = cv2.threshold(maze, 0, 255, cv2.THRESH_BINARY_INV)
    # 将图像的宽度和高度作为数组的尺寸
    height, width = maze.shape
    # 初始化二维数组
    maze_array = np.zeros((height, width), dtype=int)
    # 将像素值转换为二维数组的值
    for y in range(height):
        for x in range(width):
            if maze[y][x] == 0:
                maze_array[y][x] = 1
    return maze_array, maze



# 定义深度优先搜索函数
def dfs(maze_array, visited, path, x, y, end_x, end_y):
    m, n = len(maze_array), len(maze_array[0])
    if x < 0 or x >= n or y < 0 or y >= m or maze_array[y][x] == 1 or visited[y][x]:
        return False
    visited[y][x] = True
    path.append([x, y])
    if x == end_x and y == end_y:
        return True
    if dfs(maze_array, visited, path, x - 1, y, end_x, end_y) or dfs(maze_array, visited, path, x + 1, y, end_x, end_y) or dfs(maze_array, visited, path, x, y - 1, end_x, end_y) or dfs(maze_array, visited, path, x, y + 1, end_x, end_y):
        return True
    path.pop()
    return False

# def lunkuo(gray):
#     # 进行边缘检测
#     # edges = cv2.Canny(gray, 0, 255)
#     # _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#     # 提取轮廓
#     contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#
#     # 绘制轮廓
#     cv2.drawContours(gray, contours, -1, (0, 255, 0), 100)
#
#     # 将轮廓转化为二维数组
#     h, w= gray.shape
#     maze = np.zeros((h, w), dtype=np.uint8)
#     for cnt in contours:
#         for pt in cnt:
#             x, y = pt[0]
#             maze[y][x] = 1
#     return maze

#该步是多余的，实际处理效果跟maze_to_array差不多
def bina_img(img):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 绘制轮廓
    img_contours = np.zeros(img.shape)
    cv2.drawContours(img_contours, contours, -1, (255, 255, 255), 1)
    return img_contours

#TODO:此时得到的二维数组还不能处理，需要将其继续压缩
if __name__ == '__main__':
    img_array, maze = maze_to_array('../image_source/map1.png')
    # print(img_array)

    n = len(img_array)
    m = len(img_array[0])

    visited = [[False for _ in range(len(img_array[0]))] for _ in range(len(img_array))]    #访问每个点有没有被遍历
    path = []
    start_x, start_y = 0, 0
    end_x, end_y = 4, 4
    dfs(img_array, visited, path, m-1, 0, 0, n-1)
    # print(path)
    # maze2 = lunkuo(maze)
    maze2 = bina_img(maze)
    cv2.imshow("image", maze)
    cv2.imshow('lunkuo', maze2)
    # cv2.imwrite('../image_source/maze.png', maze)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


