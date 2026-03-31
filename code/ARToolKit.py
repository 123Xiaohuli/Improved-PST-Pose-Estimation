import cv2
import numpy as np
import imageio
import math
import csv
import os
import time
# --------- 工具函数 ---------
def rotationMatrixToEulerAngles(R):
    sy = math.sqrt(R[0,0]**2 + R[1,0]**2)
    singular = sy < 1e-6
    if not singular:
        roll = math.atan2(R[2,1], R[2,2])
        pitch = math.atan2(-R[2,0], sy)
        yaw = math.atan2(R[1,0], R[0,0])
    else:
        roll = math.atan2(-R[1,2], R[1,1])
        pitch = math.atan2(-R[2,0], sy)
        yaw = 0
    return np.degrees([roll, pitch, yaw])
# ------------------- 相机参数 -------------------
K = np.array([[551.1868593011053, 0.0, 350.1284371709081],
              [0.0, 549.5395232994132, 208.43110383966297],
              [0.0, 0.0, 1.0]])
dist_coeffs = np.array([0.013431040848392244,
                        -0.07529320782091417,
                        -0.007709219525912915,
                        0.011980747551222953, 0.0])
# ------------------- Marker 世界坐标 (单位: m) -------------------
marker_size = 0.077
objp = np.array([
    [-marker_size/2, marker_size/2, 0],
    [ marker_size/2, marker_size/2, 0],
    [ marker_size/2,-marker_size/2, 0],
    [-marker_size/2,-marker_size/2, 0]
], dtype=np.float32)

# ------------------- 读取 ARToolKit 官方 patt.hiro 模板 -------------------
# 下载 patt_hiro.png 或从 artoolkitx/examples 目录
template_marker = imageio.v2.imread("patthiro.png", mode='L')
template_marker = cv2.resize(template_marker, (64,64))  # 固定大小
_, template_marker = cv2.threshold(template_marker, 127, 255, cv2.THRESH_BINARY)

# ------------------- 辅助函数：排序顶点 -------------------
def order_points(pts):
    rect = np.zeros((4,2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect
# --------- CSV 文件写入 ---------
OUTPUT_CSV = "ARToolKit.csv"
os.makedirs(os.path.dirname(OUTPUT_CSV) or ".", exist_ok=True)
csv_file = open(OUTPUT_CSV, mode="w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow([
    "frame_id", "timestamp",
    "roll_ART", "pitch_ART", "yaw_ART"
])
# ------------------- 摄像头 -------------------
cap = cv2.VideoCapture(0)
frame_id = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # 二值化
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # 轮廓检测
    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        # 多边形逼近
        epsilon = 0.02*cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        if len(approx) == 4 and cv2.isContourConvex(approx):
            pts = approx.reshape(4,2)
            rect = order_points(pts)

            # 透视变换到正方形
            dst_size = 64
            dst_pts = np.array([[0,0],[dst_size-1,0],[dst_size-1,dst_size-1],[0,dst_size-1]], dtype=np.float32)
            M = cv2.getPerspectiveTransform(rect, dst_pts)
            warped = cv2.warpPerspective(gray, M, (dst_size,dst_size))

            # 二值化
            _, warped_bin = cv2.threshold(warped, 127, 255, cv2.THRESH_BINARY)

            # 模板匹配 (patt.hiro)
            match_score = np.sum(warped_bin == template_marker)
            if match_score > dst_size*dst_size*0.85:  # 阈值可调
                # 匹配成功，求解位姿
                success, rvec, tvec = cv2.solvePnP(objp, rect, K, dist_coeffs)
                R, _ = cv2.Rodrigues(rvec)
                roll_ART, pitch_ART, yaw_ART = rotationMatrixToEulerAngles(R)
                cv2.drawFrameAxes(frame, K, dist_coeffs, rvec, tvec, 0.03)
                cv2.polylines(frame, [approx], True, (0,255,0), 2)
                cv2.putText(frame, "ID: hiro", tuple(rect[0].astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

                # 写入 CSV
                timestamp = time.time()
                csv_writer.writerow([
                    frame_id, timestamp,
                    roll_ART, pitch_ART, yaw_ART
                ])

    cv2.imshow("ARToolKit Python (patt.hiro)", frame)
    if cv2.waitKey(1) == 27:  # ESC退出
        break
    frame_id += 1

cap.release()
cv2.destroyAllWindows()
