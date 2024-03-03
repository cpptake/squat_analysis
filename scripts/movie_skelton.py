from ultralytics import YOLO
import cv2
import numpy as np
import pandas as pd
import math

class CFG:
    model_path = 'models/yolov8l-pose.pt'
    mov_name = "フルスクワット"

def calc_degree(vec_a,vec_c):
    """
    2つのベクトルからなす角を求める関数
    https://qiita.com/hacchi_/items/7e6f433d465df9378d7a
    """
    # コサインの計算
    length_vec_a = np.linalg.norm(vec_a)
    length_vec_c = np.linalg.norm(vec_c)
    inner_product = np.inner(vec_a, vec_c)
    cos = inner_product / (length_vec_a * length_vec_c)

    # 角度（ラジアン）の計算
    rad = np.arccos(cos)

    # 弧度法から度数法（rad ➔ 度）への変換
    degree = np.rad2deg(rad)

    return degree
 
def pose_estimation_movie(filename_in, filename_out, model):
    """ 動画ファイルからリアルタイム物体検出する関数 """
 
    # 動画撮影を開始
    cap = cv2.VideoCapture(filename_in)
 
    # 動画ファイル保存用の設定
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    video = cv2.VideoWriter(filename_out, fourcc, fps, (w, h))

    KEYPOINTS_NAMES = [
    "nose",  # 0
    "eye(L)",  # 1
    "eye(R)",  # 2
    "ear(L)",  # 3
    "ear(R)",  # 4
    "shoulder(L)",  # 5
    "shoulder(R)",  # 6
    "elbow(L)",  # 7
    "elbow(R)",  # 8
    "wrist(L)",  # 9
    "wrist(R)",  # 10
    "hip(L)",  # 11
    "hip(R)",  # 12
    "knee(L)",  # 13
    "knee(R)",  # 14
    "ankle(L)",  # 15
    "ankle(R)",  # 16
    ]

    rf_point_R = []
    rf_point_R = []
    horizon_vec = np.zeros(2)
    horizon_vec[0] = 1

    knee_point = np.zeros(2)

    angle_dict = {"大腿角度R": [],
                "大腿角度L": [],
                "膝角度R": [],
                "膝角度L": [],
                "大腿直筋肉長さR":[]}
    
    while cap.isOpened():
        # フレームを抽出する
        success, frame = cap.read()

        try:
            if success:
                # 物体検出
                results = model(frame, max_det=100)
                keypoints = results[0].keypoints
                if keypoints == None:
                    print("keypoint is ",keypoints.conf[0])
                    continue
                else:
                    confs = keypoints.conf[0].tolist()
                    xys = keypoints.xy[0].tolist()

                df_keypoint = pd.DataFrame({
                    "confidence":confs,
                    "points":xys
                },index = KEYPOINTS_NAMES)                
                
                # 上体ベクトル
                body_R_vec  = np.array(xys[6]) - np.array(xys[12])
                body_L_vec  = np.array(xys[5]) - np.array(xys[11])
                # 大腿ベクトル
                thigh_R_vec = np.array(xys[14]) - np.array(xys[12])
                thigh_L_vec = np.array(xys[13]) - np.array(xys[11])
                # 下肢ベクトル
                knee_R_vec = np.array(xys[14]) - np.array(xys[16])
                knee_L_vec = np.array(xys[13]) - np.array(xys[15])

                # 角度計算
                body_deg_R = calc_degree(body_R_vec,thigh_R_vec)
                body_deg_L = calc_degree(body_L_vec,thigh_L_vec)
                leg_deg_R = calc_degree(thigh_R_vec,knee_R_vec)
                leg_deg_L = calc_degree(thigh_L_vec,knee_L_vec)
                shin_deg_R = calc_degree(knee_R_vec,horizon_vec)
                shin_deg_L = calc_degree(knee_L_vec,horizon_vec)

                # 大腿四頭筋長さ推定
                # 起始部
                rf_strt_R = (np.array(xys[6]) - np.array(xys[12]))/10
                rf_strt_L = (np.array(xys[5]) - np.array(xys[11]))/10
                strtR = np.array(xys[12]) + rf_strt_R
                strtL = np.array(xys[11]) + rf_strt_L

                # 膝位置推定
                # 膝の点から大腿角度の法線方向に膝があると仮定する
                knee_rad = math.radians(leg_deg_R/2)
                length = 60#それっぽい値を決め打ち

                knee_point[0] = int(np.array(xys[14])[0] + length * math.sin(knee_rad))
                knee_point[1] = int(np.array(xys[14])[1] - length * math.cos(knee_rad))

                # 停止部
                rf_stop_R = (np.array(xys[14]) - np.array(xys[16]))/10
                rf_stop_L = (np.array(xys[13]) - np.array(xys[15]))/10
                stopR = np.array(xys[14]) 
                stopR[0] = np.array(xys[14])[0] + length * math.sin(math.radians(shin_deg_R))
                stopR[1] = np.array(xys[14])[1] + length * math.sin(math.radians(shin_deg_R))

                # 大腿四頭筋長さ推定
                quad_lenghth = np.linalg.norm(strtR - knee_point) + np.linalg.norm(stopR-knee_point)
                femur_lenghth = np.linalg.norm(np.array(xys[14][0]) - np.array(xys[16][0])) + np.linalg.norm(np.array(xys[14][0]) - np.array(xys[16][0]))
                quad_relate_lenghth = quad_lenghth/femur_lenghth

                # 推定結果をオーバーレイ
                img_annotated = results[0].plot(boxes=False)
                # 姿勢推定の座標取得
                keypoints = results[0].keypoints
                xys = keypoints.xy[0].tolist()
                confs = keypoints.conf[0].tolist()

                angle_dict["大腿角度R"].append(body_deg_R)
                angle_dict["大腿角度L"].append(body_deg_L)
                angle_dict["膝角度R"].append(leg_deg_R)
                angle_dict["膝角度L"].append(leg_deg_L)
                angle_dict["大腿直筋肉長さR"].append(quad_lenghth)

                # 起始点描画
                radius = 8
                color = (0, 0, 255)
                thickness = -1

                # 大腿四頭筋位置の可視化
                cv2.circle(img_annotated, (int(strtR[0]) ,int(strtR[1])), radius, color, thickness)
                cv2.circle(img_annotated, (int(stopR[0]) ,int(stopR[1])), radius, color, thickness)
                cv2.circle(img_annotated, (int(knee_point[0]) ,int(knee_point[1])), radius, color, thickness)
                cv2.line(img_annotated, (int(strtR[0]) ,int(strtR[1])), (int(knee_point[0]),int(knee_point[1])), color, thickness=3)
                cv2.line(img_annotated, (int(stopR[0]) ,int(stopR[1])), (int(knee_point[0]),int(knee_point[1])), color, thickness=3)

                # 物体検出結果画像を表示
                # image_height, image_width, _ = img_annotated.shape
                # text_position = (image_width - text_width - 10, image_height - 10) 
                # cv2.putText(img_annotated, quad_relate_lenghth)
                cv2.imshow("Movie", cv2.resize(img_annotated,(580,980)))
    
                # 保存
                video.write(img_annotated)
    
                # qキーが押されたらウィンドウを閉じる
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                break
        
        except Exception as e:
            print("error ocurred : ",e)
            continue
 
    # リリース
    pd.DataFrame(angle_dict).to_csv(f"output/{CFG.mov_name}.csv",index = True)
    cap.release()
    cv2.destroyAllWindows()
 
    return

if __name__ == '__main__':
    """ Main """
    
    # 動画ファイル
    filename_in = f'input/{CFG.mov_name}.mp4'
    filename_out = f'output/{CFG.mov_name}_skelton_origin.mp4'

    # モデルを設定
    model = YOLO(CFG.model_path)
 
    # 物体検出関数を実行
    pose_estimation_movie(filename_in, filename_out, model)