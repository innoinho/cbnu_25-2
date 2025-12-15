## 📄 dual_output_lstm_anomaly_weighted.py

본 파일은 형태소 분석 기반 LSTM 이중 출력(Dual-Output) 모델을 구현한 실험 코드이다.  
한국어 사용자 리뷰 텍스트를 대상으로 감성(Sentiment)과 요구사항(Requirement)을 동시에 분류하며, 
부정 리뷰에 대해 요구사항 학습을 강화하는 Anomaly-Weighted 학습 전략을 검증하는 것을 목적으로 함.
---

### 🔍 핵심 특징

- Mecab 형태소 분석 + LSTM 기반 경량 모델 구조
- 하나의 LSTM 은닉 표현을 공유하는 Dual-Head 분류 구조
- 부정 감성 리뷰(Negative)를 이상치(Anomaly)로 정의
- 요구사항 분류 손실에만 Sample Weight 기반 가중치 적용
- YAKE 기반 핵심 키워드 추출 및 시각화 포함 

---

### ⚙️ 처리 흐름

1. 리뷰 데이터 로드 및 평점 기반 감성 라벨링
2. 키워드 규칙 기반 요구사항 카테고리 라벨링
3. Mecab 형태소 분석을 통한 토큰화 및 시퀀스 변환
4. LSTM 기반 Dual-Output 모델 학습
5. 부정 리뷰 대상 요구사항 가중 학습 및 성능 평가
   
---

### 🧠 모델 개요

- 입력: 형태소 분석 결과 기반 토큰 시퀀스
- 공통 은닉층: 단일 LSTM Layer
- 출력층:
  - Sentiment Head (Softmax)
  - Requirement Head (Softmax)
- 두 출력은 동일한 문맥 표현을 공유하되 독립적으로 학습됨 

---

### 📌 활용 목적

- Transformer 계열 모델 적용 이전의 베이스라인 모델
- 한국어 리뷰 분석을 위한 형태소 기반 접근 방식 검증
- 이상치 가중 학습 전략의 효과 비교 실험
- 요구사항 키워드 해석 및 시각화 실험 코드 
