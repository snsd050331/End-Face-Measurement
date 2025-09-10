import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt
from skimage import io, color
import os
import cv2

def process_image_fcm(image_path, n_clusters=3):
    """
    對單張圖片進行FCM聚類處理，排除黑色背景類別
    """
    class_array = []
    
    # 讀取影像
    image = io.imread(image_path)
    original_image = image.copy()  # 保存原始圖像用於最終顯示
    
    # 如果是彩色圖轉成灰階
    if len(image.shape) == 3:
        # gray = color.rgb2gray(image)
        gray = image[:,:,0]  # 直接取R通道，避免float轉換問題
    else:
        gray = image

    # 攤平成一維向量 (FCM需要一維輸入)
    pixel_values = gray.flatten().astype(np.float64)

    # Fuzzy C-Means
    cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
        data=pixel_values.reshape(1, -1),  # (features, samples)
        c=n_clusters,                      # 聚類數
        m=2,                               # Fuzziness 係數
        error=0.005,                       # 終止條件
        maxiter=1000,                      # 最大迭代次數
        init=None
    )

    # 找出每個像素的 cluster index
    cluster_membership = np.argmax(u, axis=0)
    segmented_image = cluster_membership.reshape(gray.shape)
    
    # 計算每個類別的像素數量
    for i in range(n_clusters):
        pixel_sum = np.sum(segmented_image == i)
        class_array.append(pixel_sum)

    # 找出聚類中心值，用於判斷哪個是背景(最暗的)
    cluster_centers = cntr.flatten()
    
    # 找到最暗的聚類中心(黑色背景)
    background_cluster = np.argmin(cluster_centers)
    
    # 創建結果圖像：背景設為0(黑色)，其他類別合併為1(白色)
    result = np.where(segmented_image == background_cluster, 0, 1)

    # 確保數據類型正確，轉換為 uint8
    result_uint8 = result.astype(np.uint8) * 255  # 乘以255讓白色區域更明顯
    
    # 使用形態學操作前先確認數據類型
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.morphologyEx(result_uint8, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # 直接填充mask中的輪廓
    filled_mask = mask.copy()
    contours_for_fill, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 找輪廓並在原始圖像上繪製
    contours, _ = cv2.findContours(filled_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    result_image = original_image.copy()
    
    for cnt in contours:
        # 多邊形逼近
        epsilon = 0.02 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        if len(approx) == 4:  # 四邊形
            # 計算面積過濾雜訊
            area = cv2.contourArea(approx)
            if area > 500:
                # # 畫出填充的輪廓 (綠色填充)
                # cv2.fillPoly(result_image, [approx], (0, 255, 0, 128))  # 半透明綠色

                # 畫出輪廓邊框 (深綠色)
                cv2.drawContours(result_image, [approx], -1, (0,255,0), 3)

                # 獲取四個角點
                corners = approx.reshape(-1, 2)

                # 畫出角點 (紅色圓點)
                for corner in corners:
                    cv2.circle(result_image, tuple(corner), 8, (255,0,0), -1)
                
                # 計算對角線
                def order_points(pts):
                    rect = np.zeros((4, 2), dtype="float32")

                    # 左上角的點有最小的和，右下角的點有最大的和
                    s = pts.sum(axis=1)
                    rect[0] = pts[np.argmin(s)]  # 左上
                    rect[2] = pts[np.argmax(s)]  # 右下

                    # 右上角的點有最小的差，左下角的點有最大的差
                    diff = np.diff(pts, axis=1)
                    rect[1] = pts[np.argmin(diff)]  # 右上
                    rect[3] = pts[np.argmax(diff)]  # 左下
                    return rect
                ordered_corners = order_points(corners.astype("float32"))
                ordered_corners = ordered_corners.astype(int)

                # 畫對角線 (藍色)
                # 對角線1: 左上到右下
                cv2.line(result_image, tuple(ordered_corners[0]), tuple(ordered_corners[2]), (0,0,255), 2)
                # 對角線2: 右上到左下
                cv2.line(result_image, tuple(ordered_corners[1]), tuple(ordered_corners[3]), (0,0,255), 2)

                # 計算對角線距離
                diagonal1_distance = np.sqrt((ordered_corners[2][0] - ordered_corners[0][0])**2 + 
                                       (ordered_corners[2][1] - ordered_corners[0][1])**2)
                diagonal2_distance = np.sqrt((ordered_corners[3][0] - ordered_corners[1][0])**2 + 
                                       (ordered_corners[3][1] - ordered_corners[1][1])**2)
                
                # 在對角線上分開顯示距離文字
                # 對角線1: 在距離起點1/3處顯示
                ratio1 = 0.3
                pos1 = (ordered_corners[0] + ratio1 * (ordered_corners[2] - ordered_corners[0])).astype(int)
                cv2.putText(result_image, f'D1: {diagonal1_distance:.1f}px', 
                        tuple(pos1), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,0,0), 3)
                
                # 對角線2: 在距離起點2/3處顯示
                ratio2 = 0.7
                pos2 = (ordered_corners[1] + ratio2 * (ordered_corners[3] - ordered_corners[1])).astype(int)
                cv2.putText(result_image, f'D2: {diagonal2_distance:.1f}px', 
                        tuple(pos2), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,0,0), 3)


    # 填充所有外部輪廓
    for cnt in contours_for_fill:
        area = cv2.contourArea(cnt)
        if area > 500:  # 過濾小的雜訊區域
            cv2.fillPoly(filled_mask, [cnt], 255)


    
    # 也創建一個排除背景的分割圖像供可視化
    segmented_no_bg = segmented_image.copy()
    # 將非背景類別重新映射為連續的標籤
    non_bg_clusters = [i for i in range(n_clusters) if i != background_cluster]
    for new_label, old_label in enumerate(non_bg_clusters):
        segmented_no_bg[segmented_image == old_label] = new_label + 1
    segmented_no_bg[segmented_image == background_cluster] = 0

    return gray, segmented_image, segmented_no_bg, result, filled_mask, result_image, image_path, background_cluster, class_array

def process_folder(folder_path, n_clusters=3, save_results=False, output_folder=None):
    """
    處理資料夾中所有圖片，排除黑色背景並合併其他類別
    
    Parameters:
    folder_path: 圖片資料夾路徑
    n_clusters: 聚類數量，預設為3
    save_results: 是否保存結果，預設為False
    output_folder: 結果保存路徑，如果save_results=True且未指定，則在原資料夾創建output子資料夾
    """
    
    # 支援的圖片格式（不區分大小寫）
    image_extensions = ['png', 'jpg', 'jpeg', 'bmp', 'tiff', 'tif']
    
    # 獲取所有圖片檔案
    image_files = []
    for filename in os.listdir(folder_path):
        if os.path.isfile(os.path.join(folder_path, filename)):
            file_ext = filename.lower().split('.')[-1]
            if file_ext in image_extensions:
                image_files.append(os.path.join(folder_path, filename))
    
    # 排序檔案列表，讓處理順序一致
    image_files.sort()
    
    if not image_files:
        print(f"在資料夾 {folder_path} 中未找到圖片檔案")
        return
    
    print(f"找到 {len(image_files)} 張圖片")
    
    # 如果要保存結果，創建輸出資料夾
    if save_results:
        if output_folder is None:
            output_folder = os.path.join(folder_path, 'fcm_results')
        os.makedirs(output_folder, exist_ok=True)
    
    # 處理每張圖片
    for idx, image_path in enumerate(image_files):
        try:
            print(f"處理第 {idx+1}/{len(image_files)} 張圖片: {os.path.basename(image_path)}")
            
            # 進行FCM處理
            gray, segmented_image, segmented_no_bg, result, mask, result_image, _, background_cluster, class_array = process_image_fcm(image_path, n_clusters)
            
            # 顯示詳細信息
            print(f"  背景類別 (最暗): {background_cluster}")
            print(f"  各類別像素數量: {class_array}")
            print(f"  背景像素數: {class_array[background_cluster]}")
            print(f"  前景像素數: {np.sum(result)}")
            
            # 顯示結果 - 3行2列的子圖
            fig, axes = plt.subplots(3, 2, figsize=(12, 15))
            fig.suptitle(f'FCM Results - {os.path.basename(image_path)}', fontsize=14)
            
            axes[0,0].imshow(gray, cmap="gray")
            axes[0,0].set_title("Original (Gray)")
            axes[0,0].axis("off")

            axes[0,1].imshow(segmented_image, cmap="jet")
            axes[0,1].set_title("FCM Segmentation (All Classes)")
            axes[0,1].axis("off")

            axes[1,0].imshow(result, cmap="gray")
            axes[1,0].set_title("Binary Result (Merged Foreground)")
            axes[1,0].axis("off")

            axes[1,1].imshow(mask, cmap="gray")
            axes[1,1].set_title("Mask Result")
            axes[1,1].axis("off")

            # 顯示帶有輪廓的最終結果
            axes[2,0].imshow(result_image)
            axes[2,0].set_title("Final Result with Contours")
            axes[2,0].axis("off")
            
            # 移除多餘的子圖
            axes[2,1].axis("off")
            
            plt.tight_layout()
            
            # 保存結果
            if save_results:
                filename = os.path.splitext(os.path.basename(image_path))[0]
                
                # 保存可視化結果
                viz_save_path = os.path.join(output_folder, f'{filename}_fcm_visualization.png')
                plt.savefig(viz_save_path, dpi=150, bbox_inches='tight')
                
                # 保存最終結果圖像
                final_save_path = os.path.join(output_folder, f'{filename}_final_result.png')
                cv2.imwrite(final_save_path, cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR))
                
                print(f"  可視化結果已保存至: {viz_save_path}")
                print(f"  最終結果已保存至: {final_save_path}")
            
            plt.show()
            
        except Exception as e:
            print(f"處理圖片 {image_path} 時發生錯誤: {str(e)}")
            continue

def process_single_image_demo(image_path, n_clusters=3):
    """
    處理單張圖片的示範函數，返回詳細信息
    """
    gray, segmented_image, segmented_no_bg, result, mask, result_image, _, background_cluster, class_array = process_image_fcm(image_path, n_clusters)
    
    print(f"圖片: {os.path.basename(image_path)}")
    print(f"圖片尺寸: {gray.shape}")
    print(f"總像素數: {gray.size}")
    print(f"聚類數量: {n_clusters}")
    print(f"背景類別 (最暗): {background_cluster}")
    print(f"各類別像素數量: {class_array}")
    print(f"背景像素數: {class_array[background_cluster]}")
    print(f"前景像素數: {np.sum(result)}")
    print(f"前景比例: {np.sum(result)/gray.size*100:.2f}%")
    
    return gray, segmented_image, segmented_no_bg, result, mask, result_image

# 使用範例
if __name__ == "__main__":
    # 指定圖片資料夾路徑
    folder_path = "D:/ChengChung/End_face_measurement/dataset/test/target_data/"
    # folder_path = "D:/ChengChung/End_face_measurement/target_data/"  # 修改為您的資料夾路徑
    save_path = "D:/ChengChung/End_face_measurement/dataset/test/output_results/"
    
    # 方法1: 只顯示結果，不保存
    # process_folder(folder_path, n_clusters=3, save_results=False)
    
    # 方法2: 顯示結果並保存到指定資料夾
    process_folder(folder_path, n_clusters=3, save_results=True, output_folder=save_path)
    
    # 方法3: 處理單張圖片進行測試
    # single_image_path = "your_test_image.jpg"  # 替換為測試圖片路徑
    # gray, segmented, segmented_no_bg, binary_result, mask, result_image = process_single_image_demo(single_image_path, n_clusters=3)