from transformers import pipeline

def translation(text):
    """텍스트 번역"""
    translator = pipeline("translation", model="Helsinki-NLP/opus-mt-en-ko")
    return translator(text)[0]['translation_text']

class Translation:
    """SeamlessM4T를 이용한 텍스트 번역 컨트롤러 (예: 영어→한국어)"""
    
    def __init__(self, source_lang="en", target_lang="ko"):
        self.pipe = pipeline(
            "translation", 
            model="facebook/seamless-m4t-medium", 
            src_lang=source_lang, 
            tgt_lang=target_lang
        )
    
    def process(self, text):
        result = self.pipe(text)
        return result[0]['translation_text']