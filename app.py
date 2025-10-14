import streamlit as st
from PIL import Image
import numpy as np
import colorsys
import pandas as pd
from collections import Counter

# --- 核心分析函數：現在只進行馬賽克化和像素提取 ---
def pixelate_and_extract_hsv(img: Image.Image, block_size: int = 16):
    """
    將單張圖片馬賽克化並提取其所有像素的 HSB (0-255) 列表。

    Args:
        img (Image.Image): PIL Image 物件。
        block_size (int): 用於馬賽克化處理的單元格大小。

    Returns:
        tuple: (list, Image.Image) 
               包含 HSB 像素列表和馬賽克化後的 PIL Image。
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
                avg_color = block.mean(axis=(0, 1)).astype(np.uint8)
                downsampled_np[y, x] = avg_color
            
    # 馬賽克後的圖片 (用於顯示)
    pixelated_img = Image.fromarray(downsampled_np, 'RGB')
    
    # 3. 轉換為 HSB (HSV) 並提取像素列表
    rgb_pixels = downsampled_np.reshape(-1, 3) / 255.0
    hsv_pixels_int = [] # 儲存 HSB 0-255 範圍
    
    for r, g, b in rgb_pixels:
        h, s, v = colorsys.rgb_to_hsv(r, g, b)
        # 將 H, S, V 轉換為 0-255 範圍以便於統計
        h_int = int(h * 255) 
        s_int = int(s * 255) 
        v_int = int(v * 255) 
        hsv_pixels_int.append((h_int, s_int, v_int))
        
    return hsv_pixels_int, pixelated_img

# --- Streamlit 界面設計 ---
st.set_page_config(
    page_title="圖片顏色分析平台",
    layout="wide"
)

st.title("圖片顏色(HSB)分析")
st.markdown("上傳多張圖片，程式會先進行馬賽克化，然後**統一統計所有圖片像素**的最主要 HSB 顏色資訊。")

# 側邊欄控制項
st.sidebar.header("設定參數")

# 1. 允許上傳多個檔案
uploaded_files = st.sidebar.file_uploader(
    "上傳圖片 (.jpg, .png) - 可選取多張", 
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True  # 關鍵設定
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

# --- 統一分析邏輯 ---
if uploaded_files: 
    
    # 準備合併所有圖片的像素數據
    all_hsv_pixels = []
    pixelated_images = []
    
    st.subheader("圖片處理與視覺化")
    st.markdown(f"**將對 {len(uploaded_files)} 張圖片進行統一分析 (Block Size={block_size})。**")
    
    # 1. 處理每張圖片並收集像素
    with st.spinner(f"正在處理 {len(uploaded_files)} 張圖片並收集像素..."):
        
        # 顯示所有圖片的馬賽克化結果
        cols = st.columns(len(uploaded_files) if len(uploaded_files) <= 4 else 4)

        for file_index, uploaded_file in enumerate(uploaded_files):
            try:
                image = Image.open(uploaded_file)
                
                # 執行馬賽克化並提取像素
                hsv_pixels, pixelated_img = pixelate_and_extract_hsv(image, block_size)
                
                all_hsv_pixels.extend(hsv_pixels)
                pixelated_images.append(pixelated_img)

                # 顯示單張圖片的馬賽克結果
                col_index = file_index % len(cols)
                with cols[col_index]:
                    st.image(pixelated_img, caption=f"檔案 #{file_index + 1}", use_container_width=True)

            except Exception as e:
                st.warning(f"跳過檔案 {uploaded_file.name} (錯誤: {e})")

    if not all_hsv_pixels:
        st.error("所有圖片處理失敗或未成功提取任何像素數據。")
    else:
        # 2. 統一統計所有收集到的像素
        st.markdown("---")
        st.subheader(f"總體顏色分析結果 ({len(uploaded_files)} 張圖片統一統計)")
        
        color_counts = Counter(all_hsv_pixels)
        total_pixels = len(all_hsv_pixels)

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
                "顏色代碼 (RGB)": hex_color
            })
            
        color_df = pd.DataFrame(results)

        # 3. 視覺化輸出
        st.markdown("#### 顏色視覺化")
        color_html = ""
        for index, row in color_df.iterrows():
            hex_color = row['顏色代碼 (RGB)']
            h, s, b = row['色相 (H)'], row['飽和度 (S)'], row['亮度 (B)']
            
            color_html += f"""
            <div style="display: inline-block; margin: 10px; text-align: center; border: 1px solid #ccc; padding: 5px; min-width: 120px;">
                <div style="width: 100px; height: 100px; background-color: {hex_color}; margin: auto; border-radius: 5px;"></div>
                <p style="margin-top: 5px; font-size: 14px;">No. {row['排名']}</p>
                <p style="font-size: 12px; margin: 0;">HSB: ({h}°, {s}%, {b}%)</p>
                <p style="font-size: 12px; margin: 0;">比例: {row['比例 (%)']}%</p>
            </div>
            """

        st.markdown(color_html, unsafe_allow_html=True)

        # 4. 顯示詳細數據表格
        st.markdown("---")
        st.markdown("#### 詳細數據表格")

        display_cols = ['排名', '色相 (H)', '飽和度 (S)', '亮度 (B)', '像素數量', '比例 (%)', '顏色代碼 (RGB)']

        st.dataframe(
            color_df[display_cols],
            hide_index=True,
            use_container_width=True
        )

        st.markdown("---")
        st.info("分析完成。備註：H (色相) 範圍 0-360；S (飽和度) 和 B (亮度) 範圍 0-100%。")

else:
    st.info("請在左側側邊欄上傳圖片以開始分析。")