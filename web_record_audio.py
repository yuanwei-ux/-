import streamlit as st
import pyaudio
import wave
import threading
import time
import io

def record_audio_web(duration=5, sample_rate=44100):
    """
    网页版录音功能
    """
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    
    audio = pyaudio.PyAudio()
    
    # 创建流
    stream = audio.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=sample_rate,
        input=True,
        frames_per_buffer=CHUNK
    )
    
    st.info("🎙️ 录音中... 请说话或呼吸")
    
    frames = []
    
    # 创建进度条
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # 录音过程
    for i in range(0, int(sample_rate / CHUNK * duration)):
        data = stream.read(CHUNK)
        frames.append(data)
        
        # 更新进度
        progress = (i + 1) / int(sample_rate / CHUNK * duration)
        progress_bar.progress(progress)
        status_text.text(f"录音进度: {int(progress * 100)}%")
    
    status_text.text("✅ 录音完成！")
    
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