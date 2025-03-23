import torch
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer

def review_llm(text):
    """번역 검토 (LLM 활용)"""
    model = pipeline("text-generation", model="meta-llama/Llama-2-7b-chat-hf")
    prompt = f"Review this translation and improve if needed:\n{text}"
    return model(prompt, max_length=200)[0]['generated_text']

class LocalLLMReviewer:
    def __init__(self, model_name="mistralai/Mistral-7B-Instruct", device="auto"):
        """로컬 LLM 모델 로드"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)

        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device
        )

    def review_translation(self, original_text, translated_text, mode="의역"):
        """LLM을 이용한 번역 검토"""
        prompt = f"""
        당신은 전문 번역가입니다. 
        원문의 의미를 유지하면서 더 자연스럽고 정확한 번역으로 개선하세요.

        **번역 스타일:** {mode}
        (의역 / 직역 중 선택 가능)

        **입력 원문:**  
        {original_text}

        **기계 번역된 문장:**  
        {translated_text}

        **검토 및 수정된 번역:**  
        """
        
        result = self.pipe(prompt, max_length=500, do_sample=True)
        return result[0]["generated_text"]
    

if __name__ == "__main__":
    reviewer = LocalLLMReviewer()

    original_text = "Hello, how are you?"
    translated_text = "안녕하세요, 어떻게 지내세요?"

    reviewed_translation = reviewer.review_translation(original_text, translated_text, mode="의역")
    print("검토된 번역 결과:")
    print(reviewed_translation)