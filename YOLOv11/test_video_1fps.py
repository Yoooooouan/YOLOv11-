from ultralytics import YOLO
import pandas as pd
import cv2  # 导入 OpenCV
import os

# 加载模型
model = YOLO('/Desktop/YOLO11n成果/target_0.5/weights/best.pt')

# 视频路径
video_path = '/Desktop/yolo_video_test_01_cut_1min.mp4'
output_dir = '/Desktop/detection_results'
os.makedirs(output_dir, exist_ok=True)

# 打开视频文件
cap = cv2.VideoCapture(video_path)

# 获取视频信息（可选，用于创建输出视频或了解进度）
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"视频帧率: {fps}, 总帧数: {total_frames}")

frame_count = 0
all_frames_data = []  # 用于存储所有帧的数据

# 逐帧处理
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # 视频结束

    # 使用YOLO模型检测当前帧
    results = model(frame, conf=0.5, iou=0.45, imgsz=640, verbose=False)  # verbose=False关闭控制台输出

    # 处理检测结果
    for r in results:
        try:
            df = r.to_df()
            # 为当前帧的检测结果添加帧号信息
            df['frame_number'] = frame_count
            # 将当前帧的数据添加到总列表
            all_frames_data.append(df)
            print(f"帧 {frame_count}: 检测到 {len(df)} 个目标")

        except AttributeError as e:
            print(f"帧 {frame_count} 处理出错: {e}")
            # 可在此添加备选处理方案

    frame_count += 1

# 释放视频捕获对象
cap.release()

# 将所有帧的检测结果合并为一个DataFrame并保存
if all_frames_data:
    final_df = pd.concat(all_frames_data, ignore_index=True)
    csv_path = os.path.join(output_dir, 'video_detection_results_per_frame.csv')
    final_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"\n所有检测结果已保存至: {csv_path}")
    print(f"共处理 {frame_count} 帧。")
else:
    print("未处理任何帧或未检测到目标。")