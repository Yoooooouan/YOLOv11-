# YOLOv11-headrate
基于YOLOV系列模型检测抬头率在学生学习方面的应用

参考：https://github.com/oanlit/headrate.git

操作方法：
  打开vs_BuildTools	下载Visual Studio生成工具 2022（10G左右）
  安装Anaconda3-2025.06-0-Windows-x86_64	安装并配置
  在Pycharm中选择解释器为Conda，并新建环境Python3.9名字为YOLOv11

1、创建并激活Anaconda虚拟环境
	打开Anaconda Prompt或终端，执行以下命令来创建一个纯净的环境，以避免包版本冲突
	创建一个名为YOLOv11，Python版本为3.9的虚拟环境
		conda create -n YOLOv11 python=3.9
	激活环境
		conda activate YOLOv11

2、安装依赖包（确保环境所在硬盘有20G以上的空间）
	接下来，在虚拟环境内安装YOLOv11核心库和LabelImg：
	pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu129	“网络要求较高”
	pip install --upgrade ultralytics torch -i https://pypi.tuna.tsinghua.edu.cn/simple
	pip install opencv-python pillow numpy matplotlib scipy
	pip install PyQt5 -i https://pypi.tuna.tsinghua.edu.cn/simple
	pip install labelimg -i https://pypi.tuna.tsinghua.edu.cn/simple

3、使用LabelImg标注人脸数据
	在激活的YOLOv11环境中，输入命令 labelimg`启动标注工具
		1)打开目录：点击 "Open Dir" 选择存放所有人脸图片的文件夹
		2)设置保存目录：点击 "Change Save Dir" 选择标注文件（.txt）的输出文件夹
		3)设置标注格式：在工具栏中，将标注格式切换为YOLO（这至关重要，它会生成YOLO所需的.txt文件，而非XML）
		4)开启自动保存：在 "View" 菜单中勾选 "Auto Save mode"，这样画完一个框后会自动保存，提升效率
		5)开始标注：使用快捷键W激活画框工具，框出人脸，然后输入标签名，例如：Face

4、标注完成后，需要按特定方式组织图片和标签文件
	组织数据集结构：
	    YOLOv11要求数据集按以下目录结构组织：
	    datasets/
	        ├── images/      # 存放所有图片
	        │   ├── train/   # 训练集图片
	        │   └── val/     # 验证集图片
	        └── labels/      # 存放所有标签文件（.txt）
	            ├── train/   # 训练集标签
	            └── val/     # 验证集标签
	    需要手动或将图片和标签文件按照一定比例（如8:2）分别放入`images/train/`, `labels/train/`和`images/val/`, `labels/val/`文件夹中

5、创建数据集配置文件
	    创建一个名为 `data.yaml` 的文件，告诉YOLOv11你的数据集在哪里以及有哪些类别
	    # data.yaml
	    path: /absolute/path/to/datasets/face_data  # 数据集的绝对路径
	    train: images/train  # 训练集路径，相对于path
	    val: images/val      # 验证集路径，相对于path
	    nc: 1                # 类别数量 (number of classes)，例如：face一类
	    names: [face]      # 类别名称列表

6、启动模型训练
	    在Anaconda Prompt中，确保处于YOLOv11环境，然后运行Python代码进行训练。
	    from ultralytics import YOLO
	    # 加载一个预训练模型（如小巧的yolo11n.pt）作为起点，这能加速训练
	    model = YOLO('yolo11n.pt')
	    # 开始训练
	    results = model.train(
	        data='path/to/your/data.yaml',  # 指向你刚创建的data.yaml文件
	        epochs=100,     # 训练轮数，可根据情况调整
	        imgsz=640,      # 输入图像的尺寸
	        batch=16,       # 批大小，根据你的显卡内存调整
	        device="0",       # 使用GPU（device="0"）或CPU（device='cpu'）
	        project='runs', # 训练结果保存的目录
	        name='face_detector'  # 本次训练运行的名称
	    )
	    训练过程会和损失、精度等指标会输出在终端。所有结果，包括训练好的模型（`best.pt`），都会保存在 `runs/detect/face_detector/` 目录下
