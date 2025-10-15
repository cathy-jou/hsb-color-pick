import streamlit as st
from PIL import Image
import numpy as np
import colorsys
import pandas as pd
from collections import Counter

# --- 核心分析函數：馬賽克化和像素提取 ---
def pixelate_and_extract_hsv(img: Image.Image, block_size: int = 16):
    """
    將單張圖片馬賽克化並提取其所有像素的 HSB 資訊。
    
    為了簡化主色分析，此函數會將圖片降採樣（馬賽克化），
    並以 H:0-360, S:0-100, B:0-100 的尺度提取每個像素的 HSB 數據，
    以及原始 RGB 數據。

    Args:
        img (Image.Image): PIL Image 物件。
        block_size (int): 用於馬賽克化處理的單元格大小。

    Returns:
        tuple: (list, Image.Image, int) 
               包含 HSB 資訊的列表、馬賽克化後的 PIL Image、總像素數。
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
            
            if block.size == 0:
                continue
                
            avg_color = block.mean(axis=(0, 1)).astype(np.uint8)
            downsampled_np[y, x] = avg_color
            
    # 3. 提取 HSB 像素列表 
    hsv_data = []
    total_pixels = downsampled_np.shape[0] * downsampled_np.shape[1]
    
    for rgb_tuple in downsampled_np.reshape(-1, 3):
        r, g, b = rgb_tuple
        
        # colorsys 預期 (0-1) 尺度
        h, s, v = colorsys.rgb_to_hsv(r/255.0, g/255.0, b/255.0) 
        
        # 轉換為常用的 H:0-360, S:0-100, B(Value):0-100 尺度
        h_deg = int(h * 360)
        s_perc = int(s * 100)
        v_perc = int(v * 100)
        
        # 儲存 HSB 資訊，以及原始 RGB (0-255) 資訊
        # 格式: ( (R, G, B), H, S, B )
        hsv_data.append(((r, g, b), h_deg, s_perc, v_perc))
        
    return hsv_data, Image.fromarray(downsampled_np), total_pixels


# --- 核心分析函數：計算主色調（包含飽和度過濾） ---
def analyze_colors_from_hsv(hsv_data: list, total_pixels: int, num_colors: int = 10, min_saturation: int = 10):
    """
    從 HSB 像素列表中計算主要顏色，並忽略低飽和度（灰階）像素。

    Args:
        hsv_data (list): 包含 [(R,G,B), H, S, B] 資訊的像素列表。
        total_pixels (int): 圖片中的總像素數。
        num_colors (int): 要返回的顏色數量。
        min_saturation (int): 最小飽和度閾值 (0-100)。低於此值的像素將被忽略。

    Returns:
        pandas.DataFrame: 包含排名前 num_colors 的顏色資訊。
    """
    
    # 1. 根據飽和度過濾無色調 (灰階) 像素
    # 飽和度 S (索引 2) 必須大於等於 min_saturation
    colorful_pixels = [
        (rgb, h, s, v) for rgb, h, s, v in hsv_data 
        if s >= min_saturation
    ]
    
    if not colorful_pixels:
        return pd.DataFrame()
    
    # 2. 統計最常見的顏色
    # 我們根據馬賽克化後的 RGB 值進行計數
    rgb_tuples = [item[0] for item in colorful_pixels]
    
    # 使用 Counter 統計每個 RGB 顏色出現的頻率
    color_counts = Counter(rgb_tuples)
    
    # 3. 選擇前 N 個最常見的顏色
    most_common_colors = color_counts.most_common(num_colors)
    
    results = []
    current_rank = 1
    
    # 處理計數結果
    for rgb_tuple, count in most_common_colors:
        r, g, b = rgb_tuple
        
        # 重新計算 HSB (使用 colorsys 0-1 尺度)
        h, s, v = colorsys.rgb_to_hsv(r/255.0, g/255.0, b/255.0)
        
        # 格式化輸出
        hex_color = f"#{r:02x}{g:02x}{b:02x}"
        
        results.append({
            "排名": current_rank,
            "色相 (H)": int(h * 360), # 0-360°
            "飽和度 (S)": int(s * 100), # 0-100%
            "亮度 (B)": int(v * 100), # 0-100% (Value / Brightness)
            # 比例基於該主色在**原始總像素**中的佔比
            "比例 (%)": round(count / total_pixels * 100, 2),
            "顏色代碼 (RGB)": hex_color
        })
        current_rank += 1
            
    return pd.DataFrame(results)


# --- Streamlit 應用介面 ---
def app():
    st.set_page_config(layout="wide", page_title="圖片主色調分析")
    st.title("圖片主色調分析工具 (飽和度過濾)")
    st.markdown("本工具對**多張圖片**進行馬賽克化後，會**統一計算**出整體主要顏色。您可以調整**飽和度閾值**來忽略無色調（灰階、黑、白）像素的影響。")

    # 關鍵修改：允許上傳多個檔案
    uploaded_files = st.file_uploader(
        "上傳一張或多張圖片 (JPG, PNG)", 
        type=["jpg", "png", "jpeg"],
        accept_multiple_files=True # <--- 允許上傳多張
    )

    # 檢查是否有檔案上傳
    if uploaded_files:
        
        # --- 側邊欄：參數設定 ---
        with st.sidebar:
            st.header("分析參數設定")
            
            # 1. 馬賽克塊大小
            block_size = st.slider(
                "馬賽克區塊大小 (降採樣級別)",
                min_value=4,
                max_value=64,
                value=16,
                step=4,
                help="數值越大，細節越少，顏色數量越少，分析速度越快。"
            )
            
            # 2. 飽和度過濾閾值
            min_saturation = st.slider(
                "最小飽和度閾值 (%)",
                min_value=0,
                max_value=30,
                value=10, 
                step=1,
                help="設定飽和度 S < 此值的像素將被視為灰階或近無色調，不納入主色計算。建議設在 5% - 15% 之間。"
            )
            
            # 3. 顏色數量
            num_colors = st.slider(
                "要顯示的主色數量",
                min_value=1,
                max_value=20,
                value=10,
                step=1
            )
            
        # --- 執行分析 ---
        
        # 用於累積所有圖片的像素數據
        all_hsv_data = []
        grand_total_pixels = 0
        
        # 設定輸出佈局
        col_img, col_results = st.columns([1, 1.5])
        
        with col_img:
            st.markdown(f"#### 圖片降採樣結果概覽 (共 {len(uploaded_files)} 張)")
            
        
        image_count = 0
        for uploaded_file in uploaded_files:
            try:
                img = Image.open(uploaded_file)
                
                # 1. 馬賽克化和像素提取 (針對單張圖片)
                hsv_data, pixelated_img, total_pixels = pixelate_and_extract_hsv(img, block_size)
                
                # 累積數據
                all_hsv_data.extend(hsv_data)
                grand_total_pixels += total_pixels
                image_count += 1
                
                # 在左側欄顯示每張圖片的降採樣結果
                with col_img:
                    st.image(pixelated_img, 
                             caption=f"圖 {image_count}: {uploaded_file.name}", 
                             width=300) # 限制寬度以堆疊顯示
                
            except Exception as e:
                st.error(f"處理圖片 {uploaded_file.name} 時發生錯誤: {e}")
        
        # 檢查是否成功處理了任何圖片
        if not all_hsv_data:
            st.warning("所有圖片處理失敗或上傳為空。")
            return

        # 2. 計算主色調 (包含飽和度過濾，使用累積數據)
        color_df = analyze_colors_from_hsv(all_hsv_data, grand_total_pixels, num_colors, min_saturation)

        # 顯示結果
        with col_results:
            if color_df.empty:
                st.error(f"根據您設定的飽和度閾值 ({min_saturation}%)，所有圖片中沒有足夠的「有色調」像素進行分析。請嘗試降低閾值。")
            else:
                st.markdown(f"#### 🏆 總體主色調分析結果 ({image_count} 張圖片)")
                st.markdown(f"---")
                
                st.markdown(f"**分析基數**: 總計 **{grand_total_pixels}** 個馬賽克像素點。")
                st.markdown(f"**過濾條件**: 飽和度 < {min_saturation}% 的像素已被排除。")
                
                # 3. 視覺化輸出
                st.markdown("##### 顏色視覺化")
                color_html = ""
                for index, row in color_df.iterrows():
                    hex_color = row['顏色代碼 (RGB)']
                    h, s, b = row['色相 (H)'], row['飽和度 (S)'], row['亮度 (B)']
                    
                    # 使用 HTML 建立顏色塊，提供更好的視覺效果
                    color_html += f"""
                    <div style="display: inline-block; margin: 10px; text-align: center; border: 1px solid #ccc; padding: 5px; min-width: 120px; border-radius: 8px; box-shadow: 2px 2px 5px rgba(0,0,0,0.1);">
                        <div style="width: 100px; height: 100px; background-color: {hex_color}; margin: auto; border-radius: 5px;"></div>
                        <p style="margin-top: 5px; font-size: 14px; font-weight: bold;">No. {row['排名']}</p>
                        <p style="font-size: 12px; margin: 0;">HEX: {hex_color.upper()}</p>
                        <p style="font-size: 12px; margin: 0;">HSB: ({h}°, {s}%, {b}%)</p>
                        <p style="font-size: 12px; margin: 0;">比例: {row['比例 (%)']}%</p>
                    </div>
                    """

                st.markdown(color_html, unsafe_allow_html=True)

                # 4. 顯示詳細數據表格
                st.markdown("---")
                st.markdown("##### 詳細數據表格")
                st.dataframe(color_df, use_container_width=True)

    elif uploaded_files is None or len(uploaded_files) == 0:
        st.info("請上傳一張或多張圖片以開始分析。")

if __name__ == "__main__":
    app()
