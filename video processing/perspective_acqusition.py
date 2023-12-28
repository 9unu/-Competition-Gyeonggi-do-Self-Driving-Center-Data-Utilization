import cv2
import numpy as np


points = np.array([[1570, 473],[242, 438],[1405, 31],[1180, 25]], dtype=np.float32)
mag = [0.5, 4]
img = cv2.imread('sbs.jpg')

h, w = img.shape[:2]

dst = np.float32([[0, 0], [w, 0], [0, h], [w, h]])

# 변환 행렬 생성
M = cv2.getPerspectiveTransform(points, dst)

img_result = cv2.warpPerspective(img, M, (w, h))
img_result = cv2.resize(img_result, None, fx=mag[0], fy=mag[1],
                       interpolation=cv2.INTER_AREA)

cv2.imwrite('result.jpg', img_result)