# 📏端面精準量測 - 檢測鋼胚對角線數據
## 安裝依賴項
首先先安裝Yolo相關套件  
在 [**Python>=3.8**](https://www.python.org/) 環境中安裝 `ultralytics` 包，包括所有[依賴項](https://github.com/ultralytics/ultralytics/blob/main/pyproject.toml)，並確保 [**PyTorch>=1.8**](https://pytorch.org/get-started/locally/)。  
[![PyPI - Version](https://img.shields.io/pypi/v/ultralytics?logo=pypi&logoColor=white)](https://pypi.org/project/ultralytics/) [![Ultralytics Downloads](https://static.pepy.tech/badge/ultralytics)](https://clickpy.clickhouse.com/dashboard/ultralytics) [![PyPI - Python Version](https://img.shields.io/pypi/pyversions/ultralytics?logo=python&logoColor=gold)](https://pypi.org/project/ultralytics/)

```bash
pip install ultralytics
```

在安裝Fuzzy C-Means相關套件  
```bash
pip install opencv-python
pip install scikit-image
pip install matplotlib
pip install scikit-fuzzy
```
## 1-st
可以利用Yolo去訓練自己的目標物，針對目標裁出圖像  
針對輸入影像去出目標物  

```bash
python end_face_measurement.py
```
### 載入預訓練的YOLO模型
model = YOLO("/*Your.pt")
### 設定路徑
folder = Path('/Your image path/')  
save_path = Path('/Save path/')  

## 2-nd

```bash
python fuzzy_c_means_final.py
```
Target會先經過Fuzzy C-Means演算法分類自己設定的分類數  
再利用形態學操作補齊破碎區域  
後續採用findContours的方式描出鋼胚外圍區域

```bash
# 指定圖片資料夾路徑
folder_path = # 修改為您的資料夾路徑
save_path = #修改為您輸出結果的儲存位置

process_folder(folder_path, n_clusters=3, save_results=True, output_folder=save_path)
n_clusters => 輸入預分成幾類
save_results => True/False # 是否儲存結果至資料夾
```
