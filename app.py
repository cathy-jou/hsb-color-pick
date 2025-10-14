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
    page_title="圖片顏色統一分析工具 (HSB)",
    layout="wide"
)

st.title("🎨 多張圖片主要顏色 (HSB) 統一分析工具")
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
                image = Image.open(uploaded_file