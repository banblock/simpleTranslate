from transformers import pipeline

def review_llm(text):
    """번역 검토 (LLM 활용)"""
    model = pipeline("text-generation", model="meta-llama/Llama-2-7b-chat-hf")
    prompt = f"Review this translation and improve if needed:\n{text}"
    return model(prompt, max_length=200)[0]['generated_text']