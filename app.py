import streamlit as st
import tempfile
import os
import time
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras.models import load_model
import wave
import io
import pyaudio

# 页面配置
st.set_page_config(
    page_title="支气管炎风险检测系统",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS样式
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .stTitle {
        color: white;
        text-align: center;
        font-size: 2.5rem !important;
        margin-bottom: 1rem;
    }
    .risk-high {
        background: linear-gradient(45deg, #FF416C, #FF4B2B);
        color: white;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .risk-medium {
        background: linear-gradient(45deg, #FF9800, #FF5722);
        color: white;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .risk-low {
        background: linear-gradient(45deg, #00b09b, #96c93d);
        color: white;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .stButton button {
        width: 100%;
        border-radius: 10px;
        height: 50px;
        font-weight: bold;
        font-size: 16px;
    }
    .recording-status {
        text-align: center;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        font-weight: bold;
    }
    .recording-active {
        background: linear-gradient(45deg, #FF416C, #FF4B2B);
        color: white;
        animation: pulse 1.5s infinite;
    }
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.7; }
        100% { opacity: 1; }
    }
</style>
""", unsafe_allow_html=True)

class BronchitisPredictor:
    def __init__(self, model_path="models/bronchitis_model.h5"):
        try:
            if os.path.exists(model_path):
                self.model = load_model(model_path)
                self.label_encoder = np.load("models/label_encoder.npy", allow_pickle=True)
                self.max_pad_len = 174
            else:
                st.error("❌ 模型文件未找到！请确保模型文件位于 models/ 目录下")
                self.model = None
        except Exception as e:
            st.error(f"❌ 模型加载失败: {str(e)}")
            self.model = None

    def extract_features(self, audio_path):
        try:
            audio, sample_rate = librosa.load(audio_path, sr=22050, duration=3.0)
            mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
            pad_width = self.max_pad_len - mfccs.shape[1]
            if pad_width < 0:
                mfccs = mfccs[:, :self.max_pad_len]
            else:
                mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
            return mfccs
        except Exception as e:
            st.error(f"❌ 音频处理错误: {str(e)}")
            return None

    def predict(self, audio_path):
        if self.model is None:
            st.error("❌ 模型未正确加载，无法进行分析")
            return "Error: Model not loaded", 0.0
        
        features = self.extract_features(audio_path)
        if features is None:
            return "Error: Could not process audio file", 0.0

        features = features[np.newaxis, ..., np.newaxis]
        prediction = self.model.predict(features, verbose=0)
        predicted_index = np.argmax(prediction)
        predicted_label = self.label_encoder[predicted_index]
        confidence = np.max(prediction)

        # 计算支气管炎风险概率
        bronchitis_prob = 0.0
        if predicted_label == "bronchitis":
            bronchitis_prob = confidence
        elif predicted_label == "healthy_breath":
            bronchitis_prob = 1 - confidence
        elif predicted_label == "healthy_voice":
            bronchitis_idx = np.where(self.label_encoder == "bronchitis")[0][0]
            bronchitis_prob = prediction[0][bronchitis_idx]

        return predicted_label, float(bronchitis_prob)

def record_audio(duration=5, sample_rate=44100):
    """录制音频"""
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    
    audio = pyaudio.PyAudio()
    
    stream = audio.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=sample_rate,
        input=True,
        frames_per_buffer=CHUNK
    )
    
    frames = []
    
    # 显示录音状态
    status_placeholder = st.empty()
    progress_bar = st.progress(0)
    
    # 录音过程
    for i in range(0, int(sample_rate / CHUNK * duration)):
        data = stream.read(CHUNK)
        frames.append(data)
        
        # 更新进度
        progress = (i + 1) / int(sample_rate / CHUNK * duration)
        progress_bar.progress(progress)
        status_placeholder.markdown(
            f'<div class="recording-status recording-active">'
            f'🎙️ 录音中... {int(progress * 100)}%'
            f'</div>', 
            unsafe_allow_html=True
        )
    
    status_placeholder.markdown(
        '<div class="recording-status" style="background: #00C851; color: white;">'
        '✅ 录音完成！'
        '</div>', 
        unsafe_allow_html=True
    )
    
    # 停止流
    stream.stop_stream()
    stream.close()
    audio.terminate()
    
    # 保存到内存中的WAV文件
    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(sample_rate)
        wf.writeframes(b''.join(frames))
    
    wav_buffer.seek(0)
    return wav_buffer.getvalue()

def display_results(label, risk):
    """显示分析结果"""
    st.markdown("---")
    st.header("📋 检测结果")
    
    # 创建结果列
    col1, col2, col3 = st.columns(3)
    
    # 音频类型
    with col1:
        label_display = label.replace('_', ' ').title()
        st.metric("🎵 音频类型", label_display)
    
    # 风险概率
    with col2:
        risk_percentage = f"{risk:.2%}"
        st.metric("📊 风险概率", risk_percentage)
    
    # 风险等级
    with col3:
        if risk > 0.7:
            risk_level = "高风险"
            risk_icon = "🔴"
        elif risk > 0.4:
            risk_level = "中风险" 
            risk_icon = "🟡"
        else:
            risk_level = "低风险"
            risk_icon = "🟢"
        st.metric("📈 风险等级", f"{risk_level} {risk_icon}")
    
    # 详细评估
    st.subheader("📝 详细评估")
    
    if risk > 0.7:
        st.markdown(f'<div class="risk-high">', unsafe_allow_html=True)
        st.write("""
        **🔴 高风险评估**
        
        **建议措施:**
        - 立即咨询呼吸科医生
        - 进行详细医学检查
        - 遵循专业治疗方案
        - 注意休息和隔离防护
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    elif risk > 0.4:
        st.markdown(f'<div class="risk-medium">', unsafe_allow_html=True)
        st.write("""
        **🟡 中风险评估**
        
        **建议措施:**
        - 密切观察呼吸道症状
        - 避免吸烟和空气污染
        - 考虑预约医生咨询
        - 加强免疫力补充营养
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="risk-low">', unsafe_allow_html=True)
        st.write("""
        **🟢 低风险评估**
        
        **建议措施:**
        - 继续保持健康生活习惯
        - 定期锻炼增强体质
        - 注意季节变化防护
        - 均衡饮食充足睡眠
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # 技术信息
    with st.expander("🔬 查看技术详情"):
        st.write(f"""
        **分析信息:**
        - 检测类别: {label_display}
        - 置信度: {risk:.4f}
        - 风险概率: {risk_percentage}
        - 评估时间: {time.strftime('%Y-%m-%d %H:%M:%S')}
        """)

def analyze_audio_file(audio_path):
    """分析音频文件"""
    with st.spinner("🔬 分析音频中，请稍候..."):
        try:
            # 加载预测器
            predictor = BronchitisPredictor()
            
            # 检查模型是否加载成功
            if predictor.model is None:
                st.error("❌ 无法进行分析，模型加载失败")
                return
            
            # 进行预测
            label, risk = predictor.predict(audio_path)
            
            # 显示结果
            display_results(label, risk)
            
        except Exception as e:
            st.error(f"❌ 分析过程中出现错误: {str(e)}")

def analyze_uploaded_file(uploaded_file):
    """分析上传的音频文件"""
    # 保存临时文件
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        audio_path = tmp_file.name
    
    try:
        analyze_audio_file(audio_path)
    finally:
        # 清理临时文件
        if os.path.exists(audio_path):
            os.unlink(audio_path)

def analyze_recorded_audio(audio_data):
    """分析录制的音频"""
    # 保存临时文件
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(audio_data)
        audio_path = tmp_file.name
    
    try:
        analyze_audio_file(audio_path)
    finally:
        # 清理临时文件
        if os.path.exists(audio_path):
            os.unlink(audio_path)

def main():
    st.title("🏥 支气管炎风险检测系统")
    st.markdown("---")
    
    # 侧边栏
    with st.sidebar:
        st.header("🔍 检测选项")
        detection_method = st.radio(
            "选择检测方式:",
            ["🎤 实时录音分析", "📁 上传音频文件分析"]
        )
        
        st.markdown("---")
        st.header("ℹ️ 使用说明")
        st.info("""
        **录音说明:**
        - 请录制清晰的呼吸声或语音
        - 建议在安静环境下录音
        - 录音时长3-10秒
        
        **支持格式:** WAV, MP3, M4A, FLAC
        **检测原理:** 基于深度学习的音频特征分析
        """)
        
        st.markdown("---")
        st.header("📊 风险等级")
        st.success("🟢 低风险: 0-40%")
        st.warning("🟡 中风险: 40-70%")  
        st.error("🔴 高风险: 70-100%")

    # 主内容区域
    col1, col2 = st.columns([2, 1])

    with col1:
        if detection_method == "🎤 实时录音分析":
            st.header("🎤 实时录音检测")
            
            # 录音设置
            col_setting1, col_setting2 = st.columns(2)
            with col_setting1:
                duration = st.slider("录音时长(秒)", min_value=3, max_value=10, value=5)
            with col_setting2:
                sample_rate = st.selectbox("采样率", [16000, 22050, 44100], index=1)
            
            # 录音说明
            st.info("""
            **录音提示:**
            1. 点击下方"开始录音"按钮
            2. 请对着麦克风正常呼吸或说话
            3. 系统会自动分析录音内容
            4. 请确保麦克风权限已开启
            """)
            
            # 录音按钮
            if st.button("🎙️ 开始录音", type="primary", use_container_width=True):
                try:
                    # 录音
                    audio_data = record_audio(duration, sample_rate)
                    
                    # 显示录制的音频
                    st.audio(audio_data, format='audio/wav')
                    
                    # 分析录音
                    analyze_recorded_audio(audio_data)
                    
                except Exception as e:
                    st.error(f"❌ 录音失败: {str(e)}")
                    st.info("""
                    **录音问题解决方案:**
                    1. 检查麦克风是否连接
                    2. 确保已授予麦克风权限
                    3. 尝试重新启动应用
                    4. 如仍无法录音，请使用文件上传功能
                    """)
        
        else:  # 上传文件分析
            st.header("📁 上传音频文件分析")
            
            uploaded_file = st.file_uploader(
                "选择音频文件",
                type=['wav', 'mp3', 'm4a', 'flac'],
                help="请上传呼吸声或语音录音"
            )
            
            if uploaded_file is not None:
                # 显示文件信息
                file_details = {
                    "文件名": uploaded_file.name,
                    "文件大小": f"{uploaded_file.size / 1024:.1f} KB",
                    "文件类型": uploaded_file.type
                }
                
                col1a, col1b = st.columns(2)
                with col1a:
                    st.audio(uploaded_file.getvalue())
                with col1b:
                    st.json(file_details)
                
                # 分析按钮
                if st.button("🔍 开始分析", type="primary", use_container_width=True):
                    analyze_uploaded_file(uploaded_file)

    with col2:
        st.header("💡 健康建议")
        
        advice_col = st.container()
        with advice_col:
            st.success("""
            **🟢 低风险建议:**
            - 保持良好生活习惯
            - 定期锻炼增强免疫力
            - 注意呼吸道防护
            """)
            
            st.warning("""
            **🟡 中风险建议:**
            - 密切观察症状变化
            - 避免吸烟和污染环境
            - 考虑就医咨询
            """)
            
            st.error("""
            **🔴 高风险建议:**
            - 立即就医检查
            - 遵循医生治疗方案
            - 注意休息和营养
            """)
        
        st.markdown("---")
        st.header("🔧 系统状态")
        
        # 检查模型文件是否存在
        model_exists = os.path.exists("models/bronchitis_model.h5")
        encoder_exists = os.path.exists("models/label_encoder.npy")
        pyaudio_available = True
        
        try:
            import pyaudio
            p = pyaudio.PyAudio()
            p.terminate()
        except:
            pyaudio_available = False
        
        if model_exists and encoder_exists:
            st.success("✅ 模型文件就绪")
        else:
            st.error("❌ 模型文件缺失")
            if not model_exists:
                st.error("缺少: models/bronchitis_model.h5")
            if not encoder_exists:
                st.error("缺少: models/label_encoder.npy")
        
        if pyaudio_available:
            st.success("✅ 录音功能就绪")
        else:
            st.warning("⚠️ 录音功能不可用")
            st.info("如需使用录音功能，请安装PyAudio")

if __name__ == "__main__":
    main()