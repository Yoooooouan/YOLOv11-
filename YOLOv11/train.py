from ultralytics import YOLO

if __name__ == "__main__":
    # 加载预训练模型
    model = YOLO("yolo11n.pt")

    # 开始训练
    model.train(
        data="headrate.yaml",
        epochs=100,
        imgsz=640,
        device="0",
    )
