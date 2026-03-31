import cv2
import numpy as np
import time
import os
import math
import csv

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

# --------- 参数 ---------
K = np.array([[551.1868593011053, 0.0, 350.1284371709081],
              [0.0, 549.5395232994132, 208.43110383966297],
              [0.0, 0.0, 1.0]])
dist_coeffs = np.array([0.013431040848392244,
                        -0.07529320782091417,
                        -0.007709219525912915,
                        0.011980747551222953, 0.0])
OUTPUT_CSV = "ArUco.csv"
# ArUco 字典类型
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
parameters = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

# Marker 实际边长（单位：米）
marker_length = 0.095

# --------- CSV 文件写入 ---------
os.makedirs(os.path.dirname(OUTPUT_CSV) or ".", exist_ok=True)
csv_file = open(OUTPUT_CSV, mode="w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow([
    "frame_id", "timestamp", "id",
    "roll_aruco", "pitch_aruco", "yaw_aruco"
])

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ 无法打开摄像头，请检查设备连接。")
    exit()

print("✅ 开始检测 ArUco 标记... 按 ESC 退出")
frame_id = 0
while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ 无法读取帧。")
        break

    start_time = time.time()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 检测 ArUco 标记
    corners, ids, rejected = detector.detectMarkers(gray)

    if ids is not None:
        # 位姿估计
        rvecs, tvecs, _obj = cv2.aruco.estimatePoseSingleMarkers(
            corners, marker_length, K, dist_coeffs)

        for i in range(len(ids)):
            # 绘制检测结果
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            cv2.drawFrameAxes(frame, K, dist_coeffs, rvecs[i], tvecs[i], marker_length * 0.5)

            # 计算旋转矩阵
            R, _ = cv2.Rodrigues(rvecs[i])
            t = tvecs[i].reshape(3, 1)
            # Euler angles
            roll_aruco, pitch_aruco, yaw_aruco = rotationMatrixToEulerAngles(R)
            tx_aruco, ty_aruco, tz_aruco = t.tolist()

            # 写入 CSV
            timestamp = time.time()
            csv_writer.writerow([
                frame_id, timestamp, int(ids),
                roll_aruco, pitch_aruco, yaw_aruco
            ])
            # 打印并保存位姿
            print(f"[ID {ids[i][0]}] t = {t.T}, R = \n{R}")


    # FPS 显示
    fps = 1.0 / (time.time() - start_time)
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("ArUco Pose Estimation", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break
    frame_id += 1

print(f"Saved CSV to {os.path.abspath(OUTPUT_CSV)}")

cap.release()
cv2.destroyAllWindows()


