import cv2
import numpy as np
import time
import csv
import os
import math


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

# -----------------------------
# 1. 参数设置
# -----------------------------
squares_x = 5
squares_y = 7
square_length = 0.04
marker_length = 0.02
aruco_dict_type = cv2.aruco.DICT_4X4_50
camera_id = 0
output_csv = "ChArUco.csv"

camera_matrix = np.array([[551.1868593011053, 0.0, 350.1284371709081],
                          [0.0, 549.5395232994132, 208.43110383966297],
                          [0.0, 0.0, 1.0]])
dist_coeffs = np.array([0.013431040848392244,
                        -0.07529320782091417,
                        -0.007709219525912915,
                        0.011980747551222953, 0.0])

# -----------------------------
# 2. 初始化 ArUco & ChArUco
# -----------------------------
aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_type)
charuco_board = cv2.aruco.CharucoBoard(
    (7, 5),           # squaresX, squaresY
    0.017,             # squareLength, 单位: 米
    0.009,             # markerLength, 单位: 米
    aruco_dict
)
parameters = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

charuco_params = cv2.aruco.CharucoParameters()
charuco_detector = cv2.aruco.CharucoDetector(charuco_board, charuco_params)
#print(dir(charuco_detector))
# -----------------------------
# 3. 打开摄像头
# -----------------------------
cap = cv2.VideoCapture(camera_id)
if not cap.isOpened():
    raise IOError("无法打开摄像头")

# -----------------------------
# 4. 准备 CSV 文件
# -----------------------------
os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
csv_file = open(output_csv, mode="w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["frame_id", "timestamp", "roll_charuco", "pitch_charuco", "yaw_charuco"])

# -----------------------------
# 5. 主循环
# -----------------------------
frame_id = 0
rvec = np.zeros((1, 3), dtype=np.float32)
tvec = np.zeros((1, 3), dtype=np.float32)

print("开始 ChArUco 检测，按 q 退出")

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ 摄像头读取失败")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 检测 ArUco 标记
    corners, ids, rejected = detector.detectMarkers(gray)
    if ids is not None and len(ids) > 0:
        cv2.aruco.drawDetectedMarkers(gray, corners, ids)
        corners_list = [c for c in corners]
        # 检测 ChArUco 亚像素角点
        detection_tuple= charuco_detector.detectBoard(gray, markerCorners=corners_list, markerIds=ids)
        detection = detection_tuple[0]
        charuco_corners = detection.corners
        charuco_ids = detection.ids
        if charuco_ids  is not None and len(charuco_corners ) > 3:
            success, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
                charucoCorners=detection.corners,
                charucoIds=detection.ids,
                board=charuco_board,
                cameraMatrix=camera_matrix,
                distCoeffs=dist_coeffs
            )
            print(f"rvec = \n{rvec}")
            if success:
                # 转换旋转向量到旋转矩阵，再得到欧拉角
                R, _ = cv2.Rodrigues(rvec)
                roll_charuco, pitch_charuco, yaw_charuco= rotationMatrixToEulerAngles(R)
                print(f"roll_charuco = \n{roll_charuco}")
                # 绘制坐标轴
                # cv2.aruco.drawAxis(frame, camera_matrix, dist_coeffs, rvec, tvec, 0.05)

                # 保存 CSV
                csv_writer.writerow([frame_id, time.time(), roll_charuco, pitch_charuco, yaw_charuco])

    cv2.imshow("ChArUco Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_id += 1

# -----------------------------
# 6. 清理
# -----------------------------
csv_file.close()
cap.release()
cv2.destroyAllWindows()
print(f"检测结束，结果已保存到 {os.path.abspath(output_csv)}")
