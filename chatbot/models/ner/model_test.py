import os, sys
from tensorflow.keras.models import Model, load_model
from tensorflow.keras import preprocessing
import numpy as np
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))
from utils.Preprocess import Preprocess

p = Preprocess(word2index_dic='../../train_tools/dict/chatbot_dict.bin',
               userdic='../../utils/user_dic.tsv')
# 토크나이저 정의
tag_tokenizer = preprocessing.text.Tokenizer(lower=False) # 태그 정보는 lower=False 소문자로 변환하지 않는다.

# 새로운 유형의 문장 NER 예측
new_sentence = '한국철도기술연구원 오늘'
pos = p.pos(new_sentence)
keywords = p.get_keywords(pos, without_tag=True)
new_seq = p.get_wordidx_sequence(keywords)
print("새로운 유형의 시퀀스 : ", new_seq)

new_padded_seqs = preprocessing.sequence.pad_sequences([new_seq], padding='post', value=0, maxlen=40)
print("새로운 유형의 시퀀스 : ", new_seq)
print("새로운 유형의 시퀀스 : ", new_padded_seqs)

# from tensorflow.keras.models import Model, load_model
# NER 예측
model = load_model('ner_model_test.h5')
mp = model.predict(np.array([new_padded_seqs[0]]))
mp = np.argmax(mp, axis=-1)   # 예측된 NER 인덱스값 추출

print("{:10} {:5}".format("단어", "예측된 NER"))
print("-"*50)
index_to_ner = {1: 'O', 2: 'B_DT', 3: 'B_FOOD', 4: 'I', 5: 'B_OG', 6: 'B_PS', 7: 'B_LC', 8: 'NNP', 9: 'B_TI', 0: 'PAD'}
for w, pred in zip(keywords, mp[0]):
    print("{:10} {:5}".format(w, index_to_ner[pred]))