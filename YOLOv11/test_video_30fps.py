from ultralytics import YOLO
import pandas as pd
import os

# 加载模型
model = YOLO('/Desktop/YOLO11n成果/train0.487/weights/best.pt')

# 使用vid_stride参数实现跳帧
results = model.predict(
    source='/Desktop/yolo_video_test_04_cut_2mins.mp4',
    conf=0.5,
    iou=0.45,
    imgsz=640,
    save=True,
    show=False,
    vid_stride=3
)

# 创建输出目录
output_dir = '/Desktop/detection_results'
os.makedirs(output_dir, exist_ok=True)

for i, r in enumerate(results):
    try:
        # 使用 to_df() 方法获取 DataFrame
        df = r.to_df()

        # 定义 CSV 文件路径
        csv_filename = f'detection_results_{i}.csv'
        csv_path = os.path.join(output_dir, csv_filename)

        # 保存为 CSV 文件[8](@ref)
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')  # 使用正确的 to_csv 方法[6,7](@ref)
        print(f"检测结果已保存到: {csv_path}")

        # 打印检测结果摘要
        print(f"图片 {i + 1} 检测到 {len(df)} 个目标：")
        for index, row in df.iterrows():
            print(
                f"检测到: {row['name']}, 置信度: {row['confidence']:.2f}, 位置: [{row['xmin']:.0f}, {row['ymin']:.0f}, {row['xmax']:.0f}, {row['ymax']:.0f}]")

    except AttributeError as e:
        print(f"错误: {e}")
        print("尝试备选方案...")

        # 备选方案：手动创建 DataFrame
        boxes = r.boxes
        if boxes is not None:
            data = []
            for box in boxes:
                class_id = int(box.cls[0].item())
                class_name = model.names[class_id]
                confidence = box.conf[0].item()
                x1, y1, x2, y2 = map(float, box.xyxy[0])

                data.append({
                    'class_id': class_id,
                    'name': class_name,
                    'confidence': confidence,
                    'xmin': x1,
                    'ymin': y1,
                    'xmax': x2,
                    'ymax': y2
                })

            # 手动创建 DataFrame[4](@ref)
            df_manual = pd.DataFrame(data)
            csv_path_manual = os.path.join(output_dir, f'detection_results_manual_{i}.csv')
            df_manual.to_csv(csv_path_manual, index=False, encoding='utf-8-sig')
            print(f"备选方案结果已保存到: {csv_path_manual}")