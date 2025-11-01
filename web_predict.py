import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras.models import load_model
import os
import streamlit as st

class BronchitisPredictor:
    def __init__(self, model_path="models/bronchitis_model.h5"):
        try:
            self.model = load_model(model_path)
            self.label_encoder = np.load("models/label_encoder.npy", allow_pickle=True)
            self.max_pad_len = 174
            st.success("✅ 模型加载成功！")
        except Exception as e:
            st.error(f"❌ 模型加载失败: {str(e)}")
            raise e

    def extract_features(self, audio_path):
        try:
            audio, sample_rate = librosa.load(audio_path, res_type='kaiser_fast')
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
        features = self.extract_features(audio_path)
        if features is None:
            return "Error: Could not process audio file", 0.0

        features = features[np.newaxis, ..., np.newaxis]
        prediction = self.model.predict(features)
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
            bronchitis_prob = prediction[0][np.where(self.label_encoder == "bronchitis")[0][0]]

        return predicted_label, bronchitis_prob