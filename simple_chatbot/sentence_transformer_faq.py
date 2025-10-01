# import sys
# import io
# sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from sentence_transformers import SentenceTransformer, util
faq_data = {
    "반품 어떻게 해?": "주문 후 7일 이내에 반품 가능합니다.",
    "배송 기간은 얼마나 걸려?": "평균 2~3일 소요됩니다.",
    "고객센터 번호 알려줘": "고객센터 번호는 1234-5678 입니다.",
    "회원가입은 무료야?": "네, 회원가입은 무료입니다.",
    "교환 하고 싶어": "구매매뉴에서 교환 버튼을 눌러주세요"
}

# 1. 모델 불러오기 (다국어 지원 모델)
#model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
model = SentenceTransformer('jhgan/ko-sroberta-multitask')
# 2. FAQ 질문들을 벡터로 변환
questions = list(faq_data.keys())
answers = list(faq_data.values())
question_embeddings = model.encode(questions, convert_to_tensor=True)

# 3. 챗봇 응답 함수
def chatbot_response(user_input):
    user_embedding = model.encode(user_input, convert_to_tensor=True)
    # 코사인 유사도 계산
    scores = util.cos_sim(user_embedding, question_embeddings)[0]
    best_match = scores.argmax().item()
    return answers[best_match]

# 4. 테스트
print("User: 반품하고 싶어")
print("Bot:", chatbot_response("반품하고 싶어"))

print("User: 배송은 언제와?")
print("Bot:", chatbot_response("배송은 언제와?"))

print("User: 교환 신청")
print("Bot:", chatbot_response("교환방법"))