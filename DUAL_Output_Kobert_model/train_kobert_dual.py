# -*- coding: utf-8 -*-
# Dual-Output KoBERT (TF) with Anomaly-Weighted Loss
# - Compatible with Spyder / .py scripts (no IPython magics)
# - Uses: skt/kobert-base-v1 + TFBertModel(from_pt=True)
# - Tasks: 3-class Sentiment + 5-class Requirement (mock labels)

import os
import sys
import random
import logging
import warnings
from typing import List, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from transformers import AutoTokenizer, TFBertModel

# ----------------------------
# 0) Logging & GPU config
# ----------------------------
logging.getLogger("transformers").setLevel(logging.ERROR)
warnings.filterwarnings("once")

# List and enable GPU if available
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"[GPU] Detected {len(gpus)} GPU(s):", gpus)
    except Exception as e:
        print("[GPU] Could not set memory growth:", e)
else:
    print("[GPU] No GPU detected. Training will run on CPU.")

# ----------------------------
# 1) Hyperparameters
# ----------------------------
MAX_LEN = 128
EMBEDDING_DIM = 768
DROPOUT_RATE = 0.1
BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 5e-5
ANOMALY_WEIGHT_LAMBDA = 1.5
RANDOM_SEED = 42

# Reproducibility
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# ----------------------------
# 2) Data Load & Mock Labels
# ----------------------------
# Change this path to your local CSV if needed
FILE_PATH = r"./yogiyo_reviews_30000.csv"

def label_sentiment(score: int) -> str:
    if score <= 2:
        return "Negative"
    elif score == 3:
        return "Neutral"
    else:
        return "Positive"

REQUIREMENT_CATEGORIES = ["Delivery", "UI/UX", "Service", "Price", "Packaging"]
REQ_KEYWORDS = {
    "Delivery":  ["배달", "시간", "기사님", "지연", "늦어", "빨라"],
    "UI/UX":     ["앱", "오류", "버그", "멈춰", "느려", "업데이트", "결제"],
    "Service":   ["상담원", "고객센터", "응대", "친절", "불친절", "취소"],
    "Price":     ["할인", "쿠폰", "배달비", "가격", "비싸", "요기패스"],
    "Packaging": ["포장", "새다", "흘러", "꼼꼼"],
}

def mock_label_requirements(text: str) -> str:
    text_lower = str(text).lower()
    hits = []
    for cat, kws in REQ_KEYWORDS.items():
        if any(kw in text_lower for kw in kws):
            hits.append(cat)
    if not hits:
        return random.choice(REQUIREMENT_CATEGORIES)
    return hits[0]

def load_or_mock_dataframe(path: str) -> pd.DataFrame:
    if os.path.isfile(path):
        df_ = pd.read_csv(path)
        if not set(["content", "score"]).issubset(df_.columns):
            raise ValueError("CSV must contain 'content' and 'score' columns.")
        df_ = df_[["content", "score"]].dropna()
        print(f"[Data] Loaded: {len(df_):,} rows")
    else:
        print(f"[Data] File not found: {path}. Using mock data (100 rows).")
        data = {
            "content": [f"샘플 리뷰 {i}입니다. 배달이 늦어 불만입니다." if i % 5 == 0 else f"샘플 리뷰 {i}입니다."
                        for i in range(100)],
            "score": np.random.randint(1, 6, 100)
        }
        df_ = pd.DataFrame(data)

    # Labels
    df_["sentiment_label"] = df_["score"].apply(label_sentiment)
    df_["requirement_label"] = df_["content"].apply(mock_label_requirements)

    # Encodings
    sentiment_map = {"Negative": 0, "Neutral": 1, "Positive": 2}
    requirement_map = {cat: i for i, cat in enumerate(REQUIREMENT_CATEGORIES)}
    df_["sentiment_encoded"] = df_["sentiment_label"].map(sentiment_map)
    df_["requirement_encoded"] = df_["requirement_label"].map(requirement_map)

    return df_, sentiment_map, requirement_map

df, sentiment_map, requirement_map = load_or_mock_dataframe(FILE_PATH)
print("[Data] Head:\n", df.head())

# ----------------------------
# 3) Tokenization
# ----------------------------
CKPT = "skt/kobert-base-v1"  # SentencePiece-based
tokenizer = AutoTokenizer.from_pretrained(CKPT, use_fast=False)

def encode_texts(tokenizer, texts: List[str], max_len: int) -> Tuple[np.ndarray, np.ndarray]:
    enc = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=max_len,
        return_tensors="np"
    )
    # Ensure int32 dtype for TF
    input_ids = enc["input_ids"].astype("int32")
    attention_mask = enc["attention_mask"].astype("int32")
    return input_ids, attention_mask

X_input_ids, X_attention_masks = encode_texts(tokenizer, df["content"].tolist(), MAX_LEN)
print(f"[Tokenize] input_ids: {X_input_ids.shape}, attention_mask: {X_attention_masks.shape}")

# ----------------------------
# 4) Labels & Split
# ----------------------------
Y_sentiment = to_categorical(df["sentiment_encoded"].values, num_classes=len(sentiment_map))
Y_requirement = to_categorical(df["requirement_encoded"].values, num_classes=len(requirement_map))

Y_anomaly_mask = np.where(df["sentiment_encoded"].values == 0, 1.0, 0.0)  # Negative -> 1
print(f"[Data] Negative (Anomaly) ratio: {Y_anomaly_mask.mean():.2f}")

(X_train_ids, X_test_ids,
 X_train_masks, X_test_masks,
 YS_train, YS_test,
 YR_train, YR_test,
 YM_train, YM_test) = train_test_split(
    X_input_ids, X_attention_masks,
    Y_sentiment, Y_requirement, Y_anomaly_mask,
    test_size=0.2, random_state=RANDOM_SEED
)

print(f"[Split] Train: {len(X_train_ids)}, Test: {len(X_test_ids)}")

# ----------------------------
# 5) Model (Functional API)
# ----------------------------
# Important: KoBERT has PyTorch weights → from_pt=True
base_model = TFBertModel.from_pretrained(CKPT, from_pt=True)

input_ids = tf.keras.layers.Input(shape=(MAX_LEN,), dtype=tf.int32, name="input_ids")
attention_mask = tf.keras.layers.Input(shape=(MAX_LEN,), dtype=tf.int32, name="attention_mask")

bert_outputs = base_model(input_ids, attention_mask=attention_mask)
cls_token = bert_outputs.last_hidden_state[:, 0, :]  # [CLS]

x = tf.keras.layers.Dropout(DROPOUT_RATE)(cls_token)
sentiment_logits = tf.keras.layers.Dense(3, activation="softmax", name="sentiment")(x)
requirement_logits = tf.keras.layers.Dense(5, activation="softmax", name="requirement")(x)

model = tf.keras.Model(
    inputs=[input_ids, attention_mask],
    outputs={"sentiment": sentiment_logits, "requirement": requirement_logits},
    name="kobert_dual_output"
)

losses = {
    "sentiment": tf.keras.losses.CategoricalCrossentropy(),
    "requirement": tf.keras.losses.CategoricalCrossentropy()
}
metrics = {
    "sentiment": [tf.keras.metrics.CategoricalAccuracy(name="acc")],
    "requirement": [tf.keras.metrics.CategoricalAccuracy(name="acc")]
}
optimizer = Adam(learning_rate=LEARNING_RATE)

model.compile(optimizer=optimizer, loss=losses, metrics=metrics)
model.summary()

# ----------------------------
# 6) Sample Weights (Anomaly-weighted)
# ----------------------------
sample_weight_req_train = 1.0 + (ANOMALY_WEIGHT_LAMBDA - 1.0) * YM_train
sample_weight_req_test  = 1.0 + (ANOMALY_WEIGHT_LAMBDA - 1.0) * YM_test
sample_weight_sent_train = np.ones_like(YM_train, dtype=np.float32)
sample_weight_sent_test  = np.ones_like(YM_test, dtype=np.float32)

train_sample_weights = {
    "sentiment": sample_weight_sent_train,
    "requirement": sample_weight_req_train
}
val_sample_weights = {
    "sentiment": sample_weight_sent_test,
    "requirement": sample_weight_req_test
}

# ----------------------------
# 7) Train
# ----------------------------
history = model.fit(
    x={"input_ids": X_train_ids, "attention_mask": X_train_masks},
    y={"sentiment": YS_train, "requirement": YR_train},
    sample_weight=train_sample_weights,
    validation_data=(
        {"input_ids": X_test_ids, "attention_mask": X_test_masks},
        {"sentiment": YS_test, "requirement": YR_test},
        val_sample_weights
    ),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=1
)

# ----------------------------
# 8) Evaluate
# ----------------------------
eval_result = model.evaluate(
    x={"input_ids": X_test_ids, "attention_mask": X_test_masks},
    y={"sentiment": YS_test, "requirement": YR_test},
    sample_weight=val_sample_weights,
    batch_size=BATCH_SIZE,
    return_dict=True,
    verbose=0
)
print("\n[Evaluation]")
for k, v in eval_result.items():
    print(f"{k}: {v:.4f}")

# ----------------------------
# 9) Inference Demo
# ----------------------------
def predict_texts(texts: List[str]) -> None:
    ids, masks = encode_texts(tokenizer, texts, MAX_LEN)
    pred = model.predict({"input_ids": ids, "attention_mask": masks}, verbose=0)

    inv_sent = {v: k for k, v in {"Negative":0,"Neutral":1,"Positive":2}.items()}
    inv_req  = {v: k for k, v in {cat:i for i, cat in enumerate(REQUIREMENT_CATEGORIES)}.items()}

    print("\n[Sample Predictions]")
    for t, ps, pr in zip(texts, pred["sentiment"], pred["requirement"]):
        s_cls = inv_sent[int(np.argmax(ps))]
        r_cls = inv_req[int(np.argmax(pr))]
        print(f'- "{t}": Sentiment={s_cls}, Requirement={r_cls}')

if __name__ == "__main__":
    samples = [
        "배달이 너무 늦어서 화가 났습니다.",
        "앱 결제가 자꾸 오류가 나요.",
        "가격도 괜찮고 포장도 깔끔했어요."
    ]
    predict_texts(samples)
