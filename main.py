from pipeline import Pipeline
from data.datacontroller import audio_loader, noise_reduction
from modules.review import review_llm 
from modules.translate import translation
from modules.txtextraction import speech_to_text


class MainController:
    """음성 → 번역 → 검토까지 실행하는 컨트롤러"""

    def __init__(self):
        self.pipeline = Pipeline([
            audio_loader,
            noise_reduction,
            speech_to_text,
            translation,
            review_llm
        ])

    def process_audio(self, audio_path):
        """오디오를 처리하는 파이프라인 실행"""
        return self.pipeline.run(audio_path)

# 실행 예시
if __name__ == "__main__":
    controller = MainController()
    result = controller.process_audio("input.mp3")
    print("\n📌 최종 번역 결과:\n", result)