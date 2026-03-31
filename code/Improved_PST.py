import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
import math
import csv
import time
import os
import apriltag

# --------parameters----------
K = np.array([[781.70203116, 0.0, 325.53882567],
              [0.0, 784.57328886, 250.39436184],
              [0.0, 0.0, 1.0]])
dist_coeffs = np.array([0.08499246 ,
                        0.24047265,
                        0.00134517,
                        0.00697997])
TAG_SIZE = 0.056
CAM_INDEX = 0
OUT_CSV = "test.csv"

# --------- utility functions---------
def build_world_points(tag_size):
    s = tag_size
    half = s/2.0
    return np.array([[-half,  half, 0.0],
                     [ half,  half, 0.0],
                     [ half, -half, 0.0],
                     [-half, -half, 0.0]], dtype=np.float64)

def image_point_pairs_to_arrays(corners):
    return np.array([[float(c[0]), float(c[1])] for c in corners], dtype=np.float64)

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
    return np.degrees([roll, pitch, yaw]) #X-axis,Y-axis,Z-axis

def get_apriltag_pose_from_detection(det):
    try:
        if hasattr(det, 'pose_R') and hasattr(det, 'pose_t'):
            R = np.array(det.pose_R, dtype=np.float64)
            t = np.array(det.pose_t, dtype=np.float64).reshape(3)
            return True, R, t
    except:
        pass
    return False, None, None

def draw_axes_safe(img, K, dist, R, t, size):
    try:
        rvec, _ = cv2.Rodrigues(R)
        tvec = t.reshape(3,1)
        return cv2.drawFrameAxes(img, K, dist, rvec, tvec, size)
    except:
        return img

# --------- PST Depth---------
def estimate_pst_depths(img_pts, world_pairs, focal):
    n = img_pts.shape[0]
    pair_depths = []
    for (i, j, Lij) in world_pairs:
        lij = np.linalg.norm(img_pts[i] - img_pts[j])
        if lij <= 1e-6:
            pair_depths.append((i, j, None))
            continue
        Zij = focal * (Lij / lij)
        pair_depths.append((i, j, Zij))
    per_point = np.zeros(n, dtype=np.float64)
    counts = np.zeros(n, dtype=np.int32)
    for (i, j, Zij) in pair_depths:
        if Zij is None:
            continue
        per_point[i] += Zij
        per_point[j] += Zij
        counts[i] += 1
        counts[j] += 1
    for k in range(n):
        if counts[k] > 0:
            per_point[k] /= counts[k]
        else:
            per_point[k] = np.nan
    valid = ~np.isnan(per_point)
    if valid.sum() == 0:
        global_mean = None
    else:
        global_mean = np.mean(per_point[valid])
    z_values = np.mean([Zij for (_, _, Zij) in pair_depths if Zij is not None])
    return per_point, global_mean, pair_depths, z_values

def compute_pairwise_world_distances(world_pts):
    pairs = []
    n = world_pts.shape[0]
    for i in range(n):
        for j in range(i+1, n):
            Lij = np.linalg.norm(world_pts[i, :2] - world_pts[j, :2])
            pairs.append((i, j, Lij))
    return pairs

# --------- PST 3D-3D Registration Function ---------
def compute_camera_points_from_pst(img_pts, pst_depths, K):
    K_inv = np.linalg.inv(K)
    Pc = []
    for (u,v), Z in zip(img_pts, pst_depths):
        uv1 = np.array([u,v,1.0])
        P = Z * (K_inv @ uv1)
        Pc.append(P)
    return np.array(Pc) #3D in Camera Coordinate System

def umeyama_alignment(src, dst):
    assert src.shape == dst.shape
    N = src.shape[0]
    mu_src = np.mean(src, axis=0)
    mu_dst = np.mean(dst, axis=0)
    Xc = src - mu_src
    Xw = dst - mu_dst
    H = Xc.T @ Xw
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[2,:] *= -1
        R = Vt.T @ U.T
    t = mu_dst - R @ mu_src
    return R, t

# --------- PST_ad : a)Distortion Compensation---------
def undistort_points_custom(img_pts, K, dist_coeffs):

    img_pts = img_pts.reshape(-1, 1, 2)
    undistorted = cv2.undistortPoints(img_pts, K, dist_coeffs, P=K)
    return undistorted.reshape(-1, 2)

# --------- PST_ad : c)MOCCC---------
def multi_view_optical_center_optimization(R_list, t_list):

    R_list = [np.array(Rc, dtype=float) for Rc in R_list]
    t_list = [np.array(t, dtype=float).reshape(3, 1) for t in t_list]

    if len(R_list) < 2:
        return R_list, t_list

    # Calculate all optical centers.
    centers = np.array([-(Rc.T @ t).flatten() for Rc, t in zip(R_list, t_list)])

    # Optical Center Averaging
    mean_center = np.mean(centers, axis=0).reshape(3, 1)


    t_opt_list = []
    for Rc, t in zip(R_list, t_list):
        Oc = -(Rc.T @ t)
        delta = mean_center - Oc
        t_opt = t - Rc @ delta
        t_opt_list.append(t_opt)

    quats = [R.from_matrix(Rc).as_quat() for Rc in R_list]  # xyzw
    quats = np.array(quats)
    mean_quat = np.mean(quats, axis=0)
    mean_quat /= np.linalg.norm(mean_quat)
    R_opt = R.from_quat(mean_quat).as_matrix()

    t_opt = np.mean(np.array(t_opt_list), axis=0)

    return R_opt, t_opt

# --------- MarkerPose---------
def markerpose_optimize(obj_pts, img_pts, rvec_mk, tvec_mk,K, max_iter=15):

    rvec_mk = rvec_mk.reshape(3, 1)
    tvec_mk = tvec_mk.reshape(3, 1)

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    # LM
    for it in range(max_iter):
        R_mk, _ = cv2.Rodrigues(rvec_mk)


        proj = (R_mk @ obj_pts.T) + tvec_mk
        X = proj[0, :]
        Y = proj[1, :]
        Z = proj[2, :]

        u = fx * (X / Z) + cx
        v = fy * (Y / Z) + cy
        proj_pts = np.vstack([u, v]).T


        residual = (img_pts - proj_pts).reshape(-1)

        # Jacobian matrix
        jac = []
        for i in range(len(obj_pts)):
            Xw = obj_pts[i]

            dX = R_mk[0, :]
            dY = R_mk[1, :]
            dZ = R_mk[2, :]

            Xi = X[i]
            Yi = Y[i]
            Zi = Z[i]

            J_u_r = fx * (dX * Zi - Xi * dZ) / (Zi ** 2)
            J_v_r = fy * (dY * Zi - Yi * dZ) / (Zi ** 2)

            J_u_t = np.array([fx / Zi, 0, -fx * Xi / (Zi ** 2)])
            J_v_t = np.array([0, fy / Zi, -fy * Yi / (Zi ** 2)])

            jac.append(np.hstack([J_u_r, J_u_t]))
            jac.append(np.hstack([J_v_r, J_v_t]))

        J = np.vstack(jac)

        # LM
        H = J.T @ J + 1e-6 * np.eye(J.shape[1])
        dp = np.linalg.inv(H) @ (J.T @ residual.reshape(-1, 1))

        rvec_mk += dp[:3]
        tvec_mk += dp[3:]

        if np.linalg.norm(dp) < 1e-6:
            break

    return rvec_mk, tvec_mk
# --------- IPPE ---------
def ippe_pose(object_points, image_points, K):

    assert np.allclose(object_points[:, 2], 0), "Object points must be coplanar (Z=0)"

    res_ippe = cv2.solvePnPGeneric(
        object_points, image_points, K, distCoeffs=None,
        flags=cv2.SOLVEPNP_IPPE_SQUARE
    )

    if isinstance(res_ippe, tuple) and len(res_ippe) >= 3:
        rvecs_list_ippe = res_ippe[1]
        tvecs_list_ippe = res_ippe[2]
    elif isinstance(res_ippe, (list, tuple)) and len(res_ippe) == 2:
        rvecs_list_ippe, tvecs_list_ippe = res_ippe
    else:
        raise RuntimeError(f"cv2.solvePnPGeneric: {type(res_ippe)}, len={len(res_ippe)}")

    poses_ippe = []
    for rvec_ippe, tvec_ippe in zip(rvecs_list_ippe, tvecs_list_ippe):
        R_ippe, _ = cv2.Rodrigues(rvec_ippe)
        poses_ippe.append((R_ippe, tvec_ippe.reshape(3, 1)))
    return poses_ippe

# ---------------- Adaptive-PST----------------
def adaptive_optimize_d1(corners_undist, tag_size, K, dist_coeffs):


    d1 = float(dist_coeffs[0])

    # Tag
    half = tag_size / 2
    objp = np.array([[-half,  half, 0.0],
                     [ half,  half, 0.0],
                     [ half, -half, 0.0],
                     [-half, -half, 0.0]], dtype=np.float64)

    # First Pose Estimation (Without Distortion)
    retval, rvec, tvec = cv2.solvePnP( objp, corners_undist, K, np.zeros(5))

    for _ in range(8):

        dist_tmp = np.array([d1, 0, 0, 0, 0], dtype=np.float64)

        # Obtain the distorted projection using the new d1.
        proj, _ = cv2.projectPoints(objp, rvec, tvec, K, dist_tmp)
        proj = proj.reshape(-1, 2)

        # proj ≈ corners_undist
        error = (corners_undist - proj).reshape(-1)


        retval, rvec, tvec = cv2.solvePnP(objp, corners_undist, K, dist_tmp, rvec, tvec, useExtrinsicGuess=True)


        uv = proj
        xn = (uv[:, 0] - K[0, 2]) / K[0, 0]
        yn = (uv[:, 1] - K[1, 2]) / K[1, 1]
        r2 = xn * xn + yn * yn

        # Jacobian
        J = np.zeros((8, 1))
        for i in range(4):
            J[2*i]   = - xn[i] * (r2[i])
            J[2*i+1] = - yn[i] * (r2[i])

        H = (J.T @ J)[0, 0] + 1e-6
        g = (J.T @ error.reshape(-1,1))[0, 0]
        dk = - g / H


        dk = np.clip(dk, -0.001, 0.001)
        d1 += float(dk)


    dist_final = np.array([d1, 0,0,0,0])
    proj_opt, _ = cv2.projectPoints(objp, rvec, tvec, K, dist_final)
    proj_opt = proj_opt.reshape(-1, 2)
    return d1, proj_opt
# --------- main ---------
def main():
    world_pts3d = build_world_points(TAG_SIZE)
    world_pairs = compute_pairwise_world_distances(world_pts3d)
    cap = cv2.VideoCapture(CAM_INDEX)
    R_all, t_all = [], []
    R_all_ac, t_all_ac = [], []
    R_all_ac_add1, t_all_ac_add1 = [], []
    R_pst_c, t_pst_c = np.eye(3), np.zeros((3,))
    R_pst_ac, t_pst_ac = np.eye(3), np.zeros((3,))
    R_pst_ac_add1, t_pst_ac_add1 = np.eye(3), np.zeros((3,))

    focal_px = (K[0, 0] + K[1, 1]) / 2.0
    FPS_TARGET = 5
    DT_TARGET = 1.0 / FPS_TARGET  # ≈ 0.2187 s

    if not cap.isOpened():
        print("Cannot open camera/video")
        return

    detector = apriltag.Detector()

    header = ['frame_id','ts','tag_id',
              'tx_pnp','ty_pnp','tz_pnp','roll_pnp','pitch_pnp','yaw_pnp',
              'tx_pst','ty_pst','tz_pst','roll_pst','pitch_pst','yaw_pst', 'Dis',
              'tx_pst_a', 'ty_pst_a', 'tz_pst_a', 'roll_pst_a', 'pitch_pst_a', 'yaw_pst_a', 'Dis_a',
              'tx_pst_c', 'ty_pst_c', 'tz_pst_c', 'roll_pst_c', 'pitch_pst_c', 'yaw_pst_c',
              'tx_pst_ac', 'ty_pst_ac', 'tz_pst_ac', 'roll_pst_ac', 'pitch_pst_ac', 'yaw_pst_ac',
              'tx_pst_ac_add1', 'ty_pst_ac_add1', 'tz_pst_ac_add1', 'roll_pst_ac_add1', 'pitch_pst_ac_add1', 'yaw_pst_ac_add1', 'Dis_add1 ',
              'tx_mk', 'ty_mk', 'tz_mk', 'roll_mk', 'pitch_mk', 'yaw_mk',
              'tx_ippe', 'ty_ippe', 'tz_ippe',"roll_ippe", "pitch_ippe", "yaw_ippe"]
    csv_file = open(OUT_CSV,'w',newline='')
    writer = csv.writer(csv_file)
    writer.writerow(header)

    frame_count = 0
    frame_id = 0
    OPT_INTERVAL = 10
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            ts = time.time()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            detections = detector.detect(gray )
            vis = frame.copy()

            for det in detections:
                corners = det.corners if hasattr(det,'corners') else det['corners']
                img_pts = image_point_pairs_to_arrays(corners)
                pts = np.array(det.corners, dtype=np.float32)
                # 亚像素
                pts_refined = cv2.cornerSubPix(
                    gray,
                    pts,
                    winSize=(5, 5),
                    zeroZone=(-1, -1),
                    criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                )

                # ----------------------- 1) Apriltag(PnP) -----------------------
                #has_det_pose, R_det, t_det = get_apriltag_pose_from_detection(det)
                success, rvec, tvec = cv2.solvePnP(world_pts3d, img_pts, K, dist_coeffs)
                if success:
                    R_det,_ = cv2.Rodrigues(rvec)
                    t_det = tvec.reshape(3)
                    has_det_pose = True
                if not has_det_pose:
                    continue

                # ----------------------- 2) Apriltag+PST -----------------------

                pst_per_point, pst_mean, pair_depths, Dis = estimate_pst_depths(img_pts, world_pairs, focal_px)
                cam_pts = compute_camera_points_from_pst(img_pts, pst_per_point, K)
                R_pst, t_pst = umeyama_alignment(cam_pts, world_pts3d)


                if np.linalg.det(R_pst) < 0:
                    R_pst *= -1
                    t_pst *= -1

                R_pst = R_pst.T
                t_pst = -R_pst @ t_pst

                # --- 3) Corner Distortion Compensation---
                undistorted_pts = undistort_points_custom(img_pts, K, dist_coeffs)
                pst_per_point, pst_mean, pair_depths, Dis_a = estimate_pst_depths(undistorted_pts, world_pairs, focal_px)
                cam_pts_a = compute_camera_points_from_pst(undistorted_pts, pst_per_point, K)
                R_pst_a, t_pst_a = umeyama_alignment(cam_pts_a , world_pts3d)


                if np.linalg.det(R_pst_a) < 0:
                    R_pst_a *= -1
                    t_pst_a *= -1
                R_pst_a = R_pst_a.T
                t_pst_a = -R_pst_a @ t_pst_a

                # ------------ 4) MOCCC------------
                R_all.append(R_pst)
                t_all.append(t_pst)

                if len(R_all) >= OPT_INTERVAL:
                   R_pst_c, t_pst_c = multi_view_optical_center_optimization(R_all, t_all)
                   print("Optimization Complete")
                   R_all.clear()
                   t_all.clear()
                else:
                   print("Insufficient frame rate, skipping optical center consistency optimization.")

                # ------------ 5) a+c------------
                R_all_ac.append(R_pst_a)
                t_all_ac.append(t_pst_a)

                if len(R_all_ac) >= OPT_INTERVAL:
                   R_pst_ac, t_pst_ac = multi_view_optical_center_optimization(R_all_ac,  t_all_ac)
                   print("a+c Complete")
                   R_all_ac.clear()
                   t_all_ac.clear()
                else:
                   print("Insufficient frame rate, skipping optical center consistency optimization.")
                # ------------ 6) Adaptive-PST ------------
                d1_opt, opt_corners = adaptive_optimize_d1(img_pts, TAG_SIZE, K, dist_coeffs)

                undistorted_pts_add1 = undistort_points_custom(opt_corners, K, dist_coeffs)
                pst_per_point_add1, pst_mean_add1, pair_depths_add1, Dis_add1 = estimate_pst_depths(undistorted_pts_add1 , world_pairs, focal_px)
                cam_pts_a_add1 = compute_camera_points_from_pst(undistorted_pts_add1, pst_per_point_add1, K)
                R_pst_a_add1, t_pst_a_add1 = umeyama_alignment(cam_pts_a_add1 , world_pts3d)

                if np.linalg.det(R_pst_a_add1) < 0:
                    R_pst_a_add1 *= -1
                    t_pst_a_add1 *= -1

                R_pst_a_add1 = R_pst_a_add1.T
                t_pst_a_add1 = -R_pst_a_add1 @ t_pst_a_add1

                R_all_ac_add1 .append(R_pst_a_add1 )
                t_all_ac_add1 .append(t_pst_a_add1 )

                if len(R_all_ac_add1) >= OPT_INTERVAL:
                   R_pst_ac_add1, t_pst_ac_add1 = multi_view_optical_center_optimization(R_all_ac_add1,  t_all_ac_add1)
                   print("add1+Complete")
                   R_all_ac_add1.clear()
                   t_all_ac_add1.clear()
                else:
                   print("Insufficient frame rate, skipping optical center consistency optimization.")

                # ------------ 7) MarkerPose(2022) ------------
                rvec_mk, tvec_mk = markerpose_optimize(world_pts3d, pts_refined, rvec, tvec, K)
                R_mk, _ = cv2.Rodrigues(rvec_mk)

                # ------------ 8) IPPE ------------
                if img_pts is not None:
                    poses = ippe_pose(world_pts3d, img_pts, K)

                R_ippe, t_ippe = poses[0]

                # ------------9) Euler angles ------------
                #apriltag
                roll_pnp, pitch_pnp, yaw_pnp = rotationMatrixToEulerAngles(R_det)
                tx_pnp, ty_pnp, tz_pnp = t_det.tolist()
                #print(f"pitch_pnp = \n{pitch_pnp}")
                # apriltag+pst
                roll_pst, pitch_pst, yaw_pst = rotationMatrixToEulerAngles(R_pst)
                tx_pst, ty_pst, tz_pst = t_pst.tolist()
                # apriltag+pst+a)
                roll_pst_a, pitch_pst_a, yaw_pst_a = rotationMatrixToEulerAngles(R_pst_a)
                tx_pst_a, ty_pst_a, tz_pst_a = t_pst_a.tolist()
                # apriltag+pst+c)
                if R_pst_c is not None:
                    roll_pst_c, pitch_pst_c, yaw_pst_c = rotationMatrixToEulerAngles(R_pst_c)
                    tx_pst_c, ty_pst_c, tz_pst_c = t_pst_c.tolist()
                else:
                    print("no Eular")
                # apriltag+pst+a)+c)
                if R_pst_ac is not None:
                    roll_pst_ac, pitch_pst_ac, yaw_pst_ac = rotationMatrixToEulerAngles(R_pst_ac)
                    tx_pst_ac, ty_pst_ac, tz_pst_ac = t_pst_ac.tolist()
                else:
                    print("no Eular")
                # apriltag+adaptived1+pst+a)+c)
                roll_pst_ac_add1, pitch_pst_ac_add1, yaw_pst_ac_add1 = rotationMatrixToEulerAngles(R_pst_ac_add1)
                tx_pst_ac_add1, ty_pst_ac_add1, tz_pst_ac_add1 = t_pst_ac_add1.tolist()
                #markerpose
                roll_mk, pitch_mk, yaw_mk = rotationMatrixToEulerAngles(R_mk)
                tx_mk, ty_mk, tz_mk = tvec_mk.tolist()
                #ippe
                roll_ippe, pitch_ippe, yaw_ippe = rotationMatrixToEulerAngles(R_ippe)
                tx_ippe, ty_ippe, tz_ippe = t_ippe.tolist()

                # ------------ 10) CSV ------------
                tag_id = det.tag_id if hasattr(det,'tag_id') else -1
                t_now = time.time()
                writer.writerow([frame_id, ts, tag_id,
                                     tx_pnp, ty_pnp, tz_pnp, roll_pnp, pitch_pnp, yaw_pnp,
                                     tx_pst, ty_pst, tz_pst, roll_pst, pitch_pst, yaw_pst, Dis,
                                     tx_pst_a, ty_pst_a, tz_pst_a, roll_pst_a, pitch_pst_a, yaw_pst_a, Dis_a,
                                     tx_pst_c, ty_pst_c, tz_pst_c, roll_pst_c, pitch_pst_c, yaw_pst_c,
                                     tx_pst_ac, ty_pst_ac, tz_pst_ac, roll_pst_ac, pitch_pst_ac, yaw_pst_ac,
                                     tx_pst_ac_add1, ty_pst_ac_add1, tz_pst_ac_add1, roll_pst_ac_add1, pitch_pst_ac_add1, yaw_pst_ac_add1, Dis_add1,
                                     tx_mk, ty_mk, tz_mk, roll_mk, pitch_mk, yaw_mk,
                                     tx_ippe, ty_ippe, tz_ippe, roll_ippe, pitch_ippe, yaw_ippe])

                # --- 11) Draw ---
                for p in img_pts:
                    cv2.circle(vis, tuple(p.astype(int)), 4, (0,255,0),-1)
                for i in range(4):
                    p1 = tuple(img_pts[i].astype(int))
                    p2 = tuple(img_pts[(i+1)%4].astype(int))
                    cv2.line(vis,p1,p2,(0,255,255),2)

                #vis = draw_axes_safe(vis, K, dist_coeffs, R_det, t_det, size=TAG_SIZE*0.4)
                #cv2.putText(vis,"Apriltag(PnP)",(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,0,0),2)
                #vis = draw_axes_safe(vis, K, dist_coeffs, R_pst, t_pst, size=TAG_SIZE*0.4)
                cv2.putText(vis,str(Dis_add1),(10,60),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)


            cv2.imshow("PST vs PnP", vis)
            key = cv2.waitKey(1) & 0xFF
            if key == 27: break
            frame_id += 1

            t_end = time.time()
            elapsed = t_end - ts
            sleep_time = DT_TARGET - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    finally:
        csv_file.close()
        cap.release()
        cv2.destroyAllWindows()
        print(f"Saved CSV to {os.path.abspath(OUT_CSV)}")

if __name__=="__main__":
    main()
