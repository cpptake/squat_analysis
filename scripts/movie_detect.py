from ultralytics import YOLO
import cv2
 
def object_detection_camera(moviefile, model):
    """ 動画ファイルからリアルタイム物体検出する関数 """
 
    # 動画撮影を開始
    cap = cv2.VideoCapture(moviefile)
 
    while cap.isOpened():
        # フレームを抽出する
        success, frame = cap.read()
 
        if success:
            # 物体検出
            results = model(frame)
 
            # バウンディングボックスをオーバーレイ
            img_annotated = results[0].plot()
 
            # 物体検出結果画像を表示
            cv2.imshow("Movie", img_annotated)
 
            # qキーが押されたらウィンドウを閉じる
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            break
 
    # リリース
    cap.release()
    cv2.destroyAllWindows()
 
    return
 
 
if __name__ == '__main__':
    """ Main """
 
    moviefile = 'input/walk.mp4'
 
    # モデルを設定
    model = YOLO('yolov8n.pt')
 
    # 物体検出関数を実行
    object_detection_camera(moviefile, model)