import speech_recognition as sr
from pydub import AudioSegment
import os

def load_audio(audio_path):
    """
    MP3/WAV 음성 파일을 로드하고 STT 모델이 처리할 수 있도록 WAV로 변환
    """
    audio = AudioSegment.from_file(audio_path)  # 파일 로드 (MP3, WAV 지원)
    
    # WAV로 변환 (16kHz, 단일 채널)
    processed_audio_path = "processed_audio.wav"
    audio = audio.set_frame_rate(16000).set_channels(1)
    audio.export(processed_audio_path, format="wav")
    
    return processed_audio_path

def speech_to_text(audio_path):
    """
    가공된 WAV 파일을 STT 모델에 입력하여 텍스트로 변환
    """
    recognizer = sr.Recognizer()
    
    with sr.AudioFile(audio_path) as source:
        audio_data = recognizer.record(source)  # 오디오 데이터 로드
        text = recognizer.recognize_google(audio_data, language="en")  # 구글 STT 사용
    return text

# 테스트 실행
if __name__ == "__main__":
    audio_file = "test.mp3"  # 테스트용 MP3 파일
    processed_audio = load_audio(audio_file)  # 오디오 변환
    result_text = speech_to_text(processed_audio)  # STT 실행
    
    print("원본 음성 파일:", audio_file)
    print("변환된 텍스트:", result_text)

    # 변환된 WAV 파일 삭제
    os.remove(processed_audio)