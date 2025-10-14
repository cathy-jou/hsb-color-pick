import streamlit as st
from PIL import Image
import numpy as np
import colorsys
import pandas as pd
from collections import Counter

# --- 核心分析函數 (保持不變) ---
def analyze_color(img: Image.Image, block_size: int = 16, top_colors: int = 5):
    """
    將圖片馬賽克化並分析其主要 HSB 顏色。

    Args:
        img (Image.Image): PIL Image 物件。
        block_size (int): 用於馬賽克化處理的單元格大小。
        top_colors (int): 欲回傳的主要顏色數量。

    Returns:
        tuple: (pandas.DataFrame, Image.Image) 
               包含顏色資訊的 DataFrame 和馬賽克化後的 PIL Image。
    """
    img = img.convert("RGB")
    width, height = img.size
    
    # 2. 降低解析度/馬賽克化 (區塊平均法)
    img_np = np.array(img)
    small_width = width // block_size
    small_height = height // block_size
    
    downsampled_np = np.zeros((small_height, small_width, 3), dtype=np.uint8)
    
    # 計算每個區塊的平均顏色
    for y in range(small_height):
        for x in range(small_width):
            y_start, y_end = y * block_size, (y + 1) * block_size
            x_start, x_end = x * block_size, (x + 1) * block_size
            
            block = img_np[y_start:y_end, x_start:x_end]
            if block.size > 0:
                # 計算平均並四捨五入到最接近的整數
                avg_color = block.mean(axis=(0, 1)).astype(np.uint8)
                downsampled_np[y, x] = avg_color
            
    # 馬賽克後的圖片 (用於顯示)
    pixelated_img = Image.fromarray(downsampled_np, 'RGB')
    
    # 3. 轉換為 HSB (HSV) 並統計
    rgb_pixels = downsampled_np.reshape(-1, 3) / 255.0
    hsv_pixels_int = [] # 儲存 HSB 0-255 範圍
    
    for r, g, b in rgb_pixels:
        h, s, v = colorsys.rgb_to_hsv(r, g, b)
        # 將 H, S, V 轉換為 0-255 範圍以便於統計 (減少顏色種類)
        h_int = int(h * 255) 
        s_int = int(s * 255) 
        v_int = int(v * 255) 
        hsv_pixels_int.append((h_int, s_int, v_int))

    # 4. 統計最常出現的顏色
    color_counts = Counter(hsv_pixels_int)
    total_pixels = len(hsv_pixels_int)
    
    # 5. 整理結果到 DataFrame
    results = []
    for (h_int, s_int, b_int), count in color_counts.most_common(top_colors):
        # 轉換為標準 HSB 範圍 (H:0-360, S:0-100%, B:0-100%)
        h_degree = round(h_int / 255.0 * 360)
        s_percent = round(s_int / 255.0 * 100)
        b_percent = round(b_int / 255.0 * 100)
        
        # 轉換為 HTML/CSS 顏色代碼 (用於視覺化)
        r, g, b = colorsys.hsv_to_rgb(h_int/255.0, s_int/255.0, b_int/255.0)
        hex_color = f'#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}'
        
        results.append({
            "排名": len(results) + 1,
            "色相 (H)": h_degree,
            "飽和度 (S)": s_percent,
            "亮度 (B)": b_percent,
            "像素數量": count,
            "比例 (%)": round(count / total_pixels * 100, 2),
            "顏色代碼 (RGB)": hex_color # 額外資訊：方便視覺化
        })
        
    df = pd.DataFrame(results)
    return df, pixelated_img

# --- Streamlit 界面設計 ---
st.set_page_config(
    page_title="圖片顏色分析工具 (HSB)",
    layout="wide"
)

st.title("🎨 圖片主要顏色 (HSB) 分析工具")
st.markdown("上傳圖片後，程式會先進行馬賽克化（降低解析度）處理，然後統計最主要的 HSB 顏色資訊。")

# 側邊欄控制項
st.sidebar.header("設定參數")

# 1. ***修改點：允許上傳多個檔案***
uploaded_files = st.sidebar.file_uploader(
    "上傳圖片 (.jpg, .png) - 可選取多張", 
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True  # 關鍵修改
)

# 2. 核心參數
block_size = st.sidebar.slider(
    "馬賽克區塊大小 (Block Size)",
    min_value=4,
    max_value=64,
    value=16,
    step=4,
    help="值越大，降低解析度越劇烈，顏色統計越少樣化。"
)

top_colors = st.sidebar.slider(
    "輸出主要顏色數量",
    min_value=1,
    max_value=20,
    value=5,
    step=1
)

# 3. ***修改點：迴圈處理所有上傳的檔案***
if uploaded_files: # 檢查列表是否為空
    
    st.subheader("分析結果")
    
    for file_index, uploaded_file in enumerate(uploaded_files):
        
        st.markdown(f"## 📁 檔案 #{file_index + 1}: {uploaded_file.name}")
        
        try:
            # 讀取圖片
            image = Image.open(uploaded_file)
            
            # 左右分欄顯示
            col1, col2 = st.columns(2)
            
            with col1:
                st.image(image, caption="原始圖片", use_column_width=True)
                st.write(f"原始解析度: {image.size[0]}x{image.size[1]}")

            # 執行分析
            with st.spinner(f"正在分析 {uploaded_file.name}..."):
                color_df, pixelated_image = analyze_color(image, block_size, top_colors)

            with col2:
                st.image(pixelated_image, caption=f"馬賽克化結果 (Block Size={block_size})", use_column_width=True)
                st.write(f"分析解析度: {pixelated_image.size[0]}x{pixelated_image.size[1]}")


            # 顯示顏色結果
            st.markdown("---")
            st.subheader(f"📊 前 {top_colors} 主要 HSB 顏色資訊")
            
            # 視覺化輸出（使用 st.markdown 和 CSS）
            st.markdown("#### 顏色視覺化")
            color_html = ""
            for index, row in color_df.iterrows():
                hex_color = row['顏色代碼 (RGB)']
                h, s, b = row['色相 (H)'], row['飽和度 (S)'], row['亮度 (B)']
                
                # 創建 HTML 元素
                color_html += f"""
                <div style="display: inline-block; margin: 10px; text-align: center; border: 1px solid #ccc; padding: 5px; min-width: 120px;">
                    <div style="width: 100px; height: 100px; background-color: {hex_color}; margin: auto; border-radius: 5px;"></div>
                    <p style="margin-top: 5px; font-size: 14px;">No. {row['排名']}</p>
                    <p style="font-size: 12px; margin: 0;">HSB: ({h}°, {s}%, {b}%)</p>
                    <p style="font-size: 12px; margin: 0;">比例: {row['比例 (%)']}%</p>
                </div>
                """

            # 渲染 HTML
            st.markdown(color_html, unsafe_allow_html=True)


            # 顯示詳細數據表格 
            st.markdown("---")
            st.markdown("#### 詳細數據表格")

            # 選擇並重命名欄位以更好地在界面顯示
            display_cols = ['排名', '色相 (H)', '飽和度 (S)', '亮度 (B)', '像素數量', '比例 (%)', '顏色代碼 (RGB)']

            st.dataframe(
                color_df[display_cols],
                hide_index=True,
                use_container_width=True
            )
            
            # 在每張圖片的結果間增加間隔
            st.markdown("<br><br>", unsafe_allow_html=True) 

        except Exception as e:
            st.error(f"分析檔案 {uploaded_file.name} 時發生錯誤: {e}")
            st.exception(e)

    # 顯示總結資訊 (只需要顯示一次)
    st.markdown("---")
    st.info("分析完成。備註：H (色相) 範圍 0-360；S (飽和度) 和 B (亮度) 範圍 0-100%。")

else:
    st.info("請在左側側邊欄上傳圖片以開始分析。")