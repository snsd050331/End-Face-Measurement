from PIL import Image, ImageDraw
from pathlib import Path
from ultralytics import YOLO
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 載入預訓練的YOLO模型
model = YOLO("D:/呈仲/ultralytics/weights/bestsquare.pt")

# 設定路徑
folder = Path('D:/ChengChung/End_face_measurement/dataset/new/')
save_path = Path('D:/ChengChung/End_face_measurement/dataset/new/target/')
save_path.mkdir(parents=True, exist_ok=True)  # 確保儲存路徑存在

# 取得所有.bmp檔案
data_list = [p.as_posix() for p in folder.rglob('*.bmp')]

# 設定信心度閾值
confidence_threshold = 0.8

print(f"找到 {len(data_list)} 個檔案")
print(f"信心度閾值: {confidence_threshold}")
print("-" * 50)

for i, img_path in enumerate(data_list):
    name = Path(img_path).stem  # 使用Path來取得檔案名稱
    
    print(f"\n處理圖片 {i+1}/{len(data_list)}: {name}")
    
    # 進行預測
    results = model(img_path, conf=confidence_threshold)  # 直接在模型中設定信心度閾值
    
    # 載入原圖用於繪製邊界框
    image = cv2.imread(img_path)
    if image is None:
        print(f"無法載入圖片: {img_path}")
        continue
    
    found_objects = False
    
    for r in results:
        boxes = r.boxes
        
        if boxes is not None and len(boxes) > 0:
            found_objects = True
            
            # 取得各項資訊
            confidences = boxes.conf.cpu().numpy()
            coordinates = boxes.xyxy.cpu().numpy()
            classes = boxes.cls.cpu().numpy() if boxes.cls is not None else None
            
            print(f"找到 {len(confidences)} 個信心度 > {confidence_threshold} 的目標物:")
            
            for idx, (conf, coord) in enumerate(zip(confidences, coordinates)):
                x1, y1, x2, y2 = coord.astype(int)  # 轉為整數座標
                class_id = int(classes[idx]) if classes is not None else -1
                
                # 取得類別名稱（如果模型有類別名稱的話）
                class_name = model.names[class_id] if hasattr(model, 'names') and class_id != -1 else f"Class_{class_id}"
                # 目標物取出
                target_image = image[y1:y2, x1:x2, :]
                output_path = save_path / f"{name}_obj{idx+1}.jpg"
                cv2.imwrite(str(output_path), target_image)
                # 顯示圖片
                plt.imshow(target_image)
                plt.axis("off")
                plt.show()

        
    # if found_objects:
    #     # 儲存結果圖片
    #     output_path = save_path / f"{name}_detected.jpg"
    #     cv2.imwrite(str(output_path), image)
    #     print(f"結果已儲存至: {output_path}")
    # else:
    #     print(f"沒有找到信心度 > {confidence_threshold} 的目標物")