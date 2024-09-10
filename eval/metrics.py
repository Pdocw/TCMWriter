import re
from typing import List, Optional

from flair.data import Sentence
from flair.nn import Classifier
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertTokenizer, BertForTokenClassification
from transformers import pipeline

# 使用预训练的BERT模型和分词器
model_name = "bert-base-chinese"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForTokenClassification.from_pretrained(model_name)

# 使用pipeline简化预测过程
ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer)


encoder = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')


def card(l):
    encoded_l = encoder.encode(list(l))
    cosine_sim = cosine_similarity(encoded_l)
    soft_count = 1 / cosine_sim.sum(axis=1)

    return soft_count.sum()


def records_soft_recall(golden: str, predicted: str):
    golden_entities = extract_entities_from_string(golden)
    predicted_entities = extract_entities_from_string(predicted)
    g = set(golden_entities)
    p = set(predicted_entities)
    if len(p) == 0:
        return 0
    if len(g) == 0:
        return 1
    card_g = card(g)
    card_p = card(p)
    card_intersection = card_g + card_p - card(g.union(p))
    return card_intersection / card_g


def extract_entities_from_string(text):
    entities = []
    if len(text) == 0:
        return entities

    # 将字符串按句子进行切分
    sentences = text.split('。')  # 根据中文句号切分

    for sent in sentences:
        if len(sent) == 0:
            continue
        ner_results = ner_pipeline(sent)
        entities.extend([entity['word'] for entity in ner_results])

    # 去重
    entities = list(set(entities))

    return entities


def records_entity_recall(golden: str, predicted: str):
    golden_entities = extract_entities_from_string(golden)
    predicted_entities = extract_entities_from_string(predicted)
    g = set(golden_entities)
    p = set(predicted_entities)
    if len(g) == 0:
        return 1
    else:
        return len(g.intersection(p)) / len(g)

