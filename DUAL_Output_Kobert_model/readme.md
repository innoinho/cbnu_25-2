## 📄 dual_output_kobert_model.py

본 파일은 KoBERT 기반 이중 출력(Dual-Output) 딥러닝 모델을 구현한 학습 및 평가 스크립트이다.  
온라인 사용자 리뷰 텍스트를 입력으로 받아 감성(Sentiment)과 요구사항(Requirement)을 하나의 모델에서 동시에 예측하는 것을 목표로 한다 

---

### 🔍 주요 기능

- KoBERT(`skt/kobert-base-v1`)를 공통 인코더로 사용
- 감성 분류와 요구사항 분류를 병렬적으로 수행하는 Dual-Output 구조
- 부정 감성 리뷰를 이상치(Anomaly)로 정의하고,  
  요구사항 분류 손실에 가중치를 적용하는 Anomaly Weighted Loss 전략 구현
- 학습, 검증, 평가 및 예측 예제까지 포함된 end-to-end 파이프라인 제공

---

### ⚙️ 처리 흐름

1. 환경 설정 및 라이브러리 로드
   - TensorFlow, HuggingFace Transformers, KoBERT 등 사용
   - 실험 재현성을 위한 랜덤 시드 고정

2. 데이터 로드 및 라벨링
   - 리뷰 내용(`content`)과 평점(`score`) 사용
   - 평점 기준 감성 라벨링  
     - 1~2점: Negative  
     - 3점: Neutral  
     - 4~5점: Positive
   - 키워드 기반 요구사항 카테고리 모의 라벨링  
     (Delivery, UI/UX, Service, Price, Packaging)

3. KoBERT 토크나이징
   - SentencePiece 기반 토크나이저 사용
   - `input_ids`, `attention_mask` 생성

4. 데이터 분할
   - 학습/검증 데이터 8:2 분할
   - Negative 감성 리뷰를 기준으로 이상치 마스크 생성 

---

### 🧠 모델 구조

- 공통 인코더
  - `TFBertModel (KoBERT)`
  - `[CLS]` 토큰 벡터를 문장 대표 임베딩으로 사용

- 출력 헤드
  - Sentiment Head: 감성 분류 (Softmax)
  - Requirement Head: 요구사항 분류 (Softmax)

- 하나의 인코더를 공유함으로써  
  감성과 요구사항 간 문맥적 연관성을 공동 학습하도록 설계 
---

### 📉 손실 함수 및 학습 전략

- 기본 손실 함수: Categorical Cross Entropy
- 요구사항 분류 손실에 대해
  - 부정 감성 리뷰(Anomaly)에 가중치 λ 적용
- 목적:
  - 단순 다수 클래스 중심 학습을 방지
  - 실제 사용자 불만 리뷰에 대한 요구사항 탐지 성능 강화 

---

### 📊 평가 및 예측

- 검증 데이터 기반 성능 평가 수행
- 임의 문장 입력에 대해
  - 감성 예측 결과
  - 요구사항 예측 결과
  출력 예제 포함

---

### 📌 활용 목적

- 학위논문 실험 코드
- 캡스톤 프로젝트 구현
- KoBERT 기반 멀티태스크 학습 구조 예제
- 사용자 리뷰 분석 서비스의 프로토타입 모델 
