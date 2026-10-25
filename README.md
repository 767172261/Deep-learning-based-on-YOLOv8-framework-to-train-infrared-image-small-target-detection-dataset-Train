深度学习基于YOLOv8框架来训练红外图像小目标检测数据集 红外图像小目标检测数据集的训练及应用 推理识别检测飞机 无人机 直升机 鸟类
@[toc]
**数据集描述：**
8302张，yolo/voc标注
4类，标注数量：

image num: 8302

“红外图像下极小目标检测数据集”

| 类别     | 英文名称 | 标注框数量 | 图像数量（含该类） | 总图像数 | 标注格式支持       |
|----------|----------|------------|---------------------|-----------|--------------------|
| 飞机     | Plane    | 2,163      | -                   |    | YOLO 和 VOC (Pascal VOC) |
| 无人机   | Drone    | 3,120      | -                   |     | YOLO 和 VOC (Pascal VOC) |
| 直升机   | Heli     | 2,217      | -                   |     | YOLO 和 VOC (Pascal VOC) |
| 鸟类     | Bird     | 1,958      | -                   | 8,302     | YOLO 和 VOC (Pascal VOC) |
| **总计** | ——       | **9,458**  | ——                  | **8,302** | ——                 |


![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/54eaa25223424ef79bc126293d4ae145.png)
1

1
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/678d48ba04d241ebbf13533a0295337a.png)
1
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/b65cac8a3b6d4c67a1ba6cbcaf699bed.png)
红外图像下极小目标检测数据集进行训练、推理和评估，基于YOLOv8框架来实现。详细的步骤和代码示例，包括如何准备数据、训练模型、进行推理以及评估模型性能。

### 一、环境准备

确保环境中安装了必要的库：

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install ultralytics opencv-python matplotlib tqdm
```

### 二、数据准备

#### 数据集组织

数据集按照以下结构组织：

```
infrared_dataset/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
├── labels/
│   ├── train/
│   ├── val/
│   └── test/
└── data.yaml
```

其中 `data.yaml` 文件内容如下：

```yaml
train: ./infrared_dataset/images/train
val: ./infrared_dataset/images/val
test: ./infrared_dataset/images/test

nc: 4
names: ['Plane', 'Drone', 'Heli', 'Bird']
```

如果你的数据标注为VOC格式，请先转换为YOLO格式（即每个标签文件包含每行一个边界框，格式为：类别ID 中心点x 中心点y 宽 高）。

### 三、训练模型

#### 训练脚本 (`train.py`)

```python
from ultralytics import YOLO
import torch

def train_model():
    # Check GPU availability
    device = 0 if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load pretrained model (YOLOv8s is a good balance between speed and accuracy)
    model = YOLO('yolov8s.pt')

    # Start training
    results = model.train(
        data='infrared_dataset/data.yaml',
        epochs=200,               # Adjust based on your needs
        imgsz=640,                # Image size can be adjusted based on the dataset
        batch=16,                 # Reduce if you encounter out-of-memory issues
        name='infrared_yolov8s',
        project='runs/detect',
        save=True,
        save_period=10,           # Save every 10 epochs
        device=device,
        workers=4,
        patience=50,              # Early stopping after 50 epochs without improvement
        optimizer='AdamW',
        lr0=0.001,                # Initial learning rate
        augment=True              # Enable data augmentation
    )

    print("Training completed!")
    return results

if __name__ == '__main__':
    train_model()
```

### 四、推理与评估

#### 推理脚本 (`detect.py`)

```python
from ultralytics import YOLO
import cv2

def detect_image(image_path, output_path='output_detection.jpg'):
    # Load trained model
    model = YOLO('runs/detect/infrared_yolov8s/weights/best.pt')

    # Perform prediction
    results = model.predict(
        source=image_path,
        conf=0.4,                  # Confidence threshold
        iou=0.5,
        save=True,
        project='runs/detect/predict',
        name='result',
        exist_ok=True
    )

    # Extract and save results (optional)
    for r in results:
        im = r.plot()  # Plot bounding boxes and labels
        cv2.imwrite(output_path, im)
        print(f"Detection completed, saved to {output_path}")

        # Print detected objects
        for box in r.boxes:
            cls_name = model.names[int(box.cls)]
            conf = float(box.conf)
            print(f"Detected: {cls_name} (Confidence: {conf:.3f})")

    return output_path

if __name__ == '__main__':
    detect_image('test_infrared_image.jpg')
```

#### 评估脚本 (`evaluate.py`)

```python
from ultralytics import YOLO

def evaluate_model():
    model = YOLO('runs/detect/infrared_yolov8s/weights/best.pt')

    # Evaluate on validation set
    metrics = model.val(
        data='infrared_dataset/data.yaml',
        split='val',               # Or 'test' if evaluating on the test set
        batch=16,
        imgsz=640,
        device=0 if torch.cuda.is_available() else 'cpu',
        save_json=True,            # Save COCO format results for further analysis
        conf_thres=0.001,
        iou_thres=0.6
    )

    # Output key metrics
    print(f"mAP@0.5: {metrics.box.map50:.4f}")
    print(f"mAP@0.5:0.95: {metrics.box.map:.4f}")
    print(f"Precision: {metrics.box.p:.4f}")
    print(f"Recall: {metrics.box.r:.4f}")

    # Output AP per class
    print("\nAP per class:")
    for i, name in enumerate(metrics.names.values()):
        print(f"  {name}: {metrics.box.ap[i]:.4f}")

if __name__ == '__main__':
    evaluate_model()
```

构建一个可以加载模型、执行推理并对新图像或视频进行实时处理的应用。

### 一、准备

确保已经完成了模型的训练，并且训练好的模型文件（例如 `best.pt`），位于 `runs/detect/infrared_yolov8s/weights/best.pt`。

### 二、模型封装

首先，我们需要封装我们的模型，使其易于使用。这可以通过创建一个类来实现，该类负责加载模型并提供预测方法。

#### `model_utils.py`

```python
from ultralytics import YOLO
import cv2

class InfraredTargetDetector:
    def __init__(self, model_path='runs/detect/infrared_yolov8s/weights/best.pt'):
        self.model = YOLO(model_path)
    
    def predict(self, image_path, conf=0.4):
        """
        对给定路径的图像进行预测。
        
        :param image_path: 输入图像的路径
        :param conf: 置信度阈值
        :return: 检测结果列表，每个结果包含类别名称、置信度分数和边界框坐标
        """
        results = self.model.predict(source=image_path, conf=conf)
        detections = []
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls)
                label = self.model.names[cls_id]
                confidence = float(box.conf)
                bbox = box.xyxy[0].cpu().numpy().tolist()  # [x1, y1, x2, y2]
                detections.append({
                    'class': label,
                    'confidence': confidence,
                    'bbox': bbox
                })
        return detections
    
    def detect_and_draw(self, image_path, output_path='output_detection.jpg', conf=0.4):
        """
        对图像进行检测并在原图上绘制检测结果。
        
        :param image_path: 输入图像的路径
        :param output_path: 输出图像的保存路径
        :param conf: 置信度阈值
        :return: None
        """
        detections = self.predict(image_path, conf)
        img = cv2.imread(image_path)
        for det in detections:
            bbox = det['bbox']
            label = f"{det['class']}: {det['confidence']:.2f}"
            color = (0, 255, 0)  # Green
            cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
            cv2.putText(img, label, (int(bbox[0]), int(bbox[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        cv2.imwrite(output_path, img)
        print(f"Detection completed and saved to {output_path}")

# 示例使用
if __name__ == '__main__':
    detector = InfraredTargetDetector()
    detector.detect_and_draw('test_infrared_image.jpg')
```

### 三、构建命令行工具

接下来，我们创建一个命令行工具，允许用户通过命令行对图像或视频进行检测。

#### `infrared_detector_cli.py`

```python
import argparse
from model_utils import InfraredTargetDetector
import cv2

def main():
    parser = argparse.ArgumentParser(description="Infrared Target Detection System")
    parser.add_argument('--input', type=str, required=True, help='Path to input image or video file.')
    parser.add_argument('--output', type=str, default='output.jpg', help='Path to save the output.')
    parser.add_argument('--conf', type=float, default=0.4, help='Confidence threshold.')
    
    args = parser.parse_args()
    
    detector = InfraredTargetDetector()
    
    if args.input.lower().endswith(('jpg', 'jpeg', 'png')):
        detector.detect_and_draw(args.input, args.output, args.conf)
    elif args.input.lower().endswith(('mp4', 'avi')):
        cap = cv2.VideoCapture(args.input)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*'XVID'), 20.0, (frame_width, frame_height))
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert frame to image path temporarily for prediction
            temp_img_path = 'temp_frame.jpg'
            cv2.imwrite(temp_img_path, frame)
            detections = detector.predict(temp_img_path, args.conf)
            
            for det in detections:
                bbox = det['bbox']
                label = f"{det['class']}: {det['confidence']:.2f}"
                color = (0, 255, 0)  # Green
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
                cv2.putText(frame, label, (int(bbox[0]), int(bbox[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            
            out.write(frame)
        
        cap.release()
        out.release()
        print(f"Video detection completed and saved to {args.output}")
    else:
        print("Unsupported file format.")

if __name__ == "__main__":
    main()
```

### 四、运行你的系统

- **对于单张图片**：
  
  ```bash
  python infrared_detector_cli.py --input path/to/image.jpg --output output.jpg --conf 0.5
  ```

- **对于视频文件**：

  ```bash
  python infrared_detector_cli.py --input path/to/video.mp4 --output output_video.avi --conf 0.5
  ```


