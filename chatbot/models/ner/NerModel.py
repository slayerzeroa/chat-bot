import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Model, load_model
from tensorflow.keras import preprocessing

# 개체명 인식 모델 모듈
class NerModel:
    def __init__(self, model_name, preprocess):
        # 각 모델의 전처리 객체
        self.p = preprocess

        # 각 모델의 keras inference model
        self.model = load_model(model_name)

        # 태그 딕셔너리
        self.index_to_ner = {1: 'O', 2: 'B_DT', 3: 'B_FOOD', 4: 'I', 5: 'B_OG', 6: 'B_PS', 7: 'B_LC', 8: 'NNP', 9: 'B_TI', 0: 'PAD'}

    # 개체명 클래스 예측
    def predict(self, query):
        pos = self.p.pos(query)
        keywords = self.p.get_keywords(pos, without_tag=True)
        sequences = [self.p.get_wordidx_sequence(keywords)]
        max_len = 40
        padded_seqs = preprocessing.sequence.pad_sequences(sequences, padding='post', value=0, maxlen=max_len)

        predict = self.model.predict(np.array([padded_seqs[0]]))
        predict_class = tf.math.argmax(predict, axis=-1)
        tags = [self.index_to_ner[i] for i in predict_class.numpy()[0]]

        return list(zip(keywords, tags))

    # 키워드별 개체명 예측
    def predict_tags(self, query):
        pos = self.p.pos(query)
        keywords = self.p.get_keywords(pos, without_tag=True)
        sequences = [self.p.get_wordidx_sequence(keywords)]
        max_len = 40
        padded_seqs = preprocessing.sequence.pad_sequences(sequences, padding='post', value=0, maxlen=max_len)

        predict = self.model.predict(np.array([padded_seqs[0]]))
        predict_class = tf.math.argmax(predict, axis=-1)
        tags = []
        for tag_idx in predict_class.numpy()[0]:
            if tag_idx == 1: continue
            tags.append(self.index_to_ner[tag_idx])

            if len(tags) == 0:
                return None
            return tags