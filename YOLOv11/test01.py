import torch
from ultralytics import YOLO

print(torch.__version__)  # 应该显示类似 2.x.x+cu121
print(torch.cuda.is_available())  # 应该返回 True


print(f"CUDA Available: {torch.cuda.is_available()}")  # 应该输出 True
print(f"CUDA Device Count: {torch.cuda.device_count()}")
print(f"Current CUDA Device: {torch.cuda.current_device()}")
print(f"Device Name: {torch.cuda.get_device_name()}")

# 尝试加载模型
model = YOLO('yolo11n.pt')
print(f"Model device: {model.device}")