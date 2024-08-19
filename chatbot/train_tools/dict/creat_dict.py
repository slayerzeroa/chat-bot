# creat_dict.py

# 단어 사전 파일 생성
# 챗봇에 사용하는 사전 파일

import os
from konlpy.tag import Komoran

from utils.Preprocess import Preprocess
from tensorflow.keras import preprocessing
import pickle
import pandas as pd

# 말뭉치 데이터 읽어오기
test_data = pd.read_csv('test_data.csv')
test_data.dropna(inplace=True)  # NaN 데이터 제거, inplace: 원본을 변경할지의 여부
corpus_data = list(test_data['Query']) + list(test_data['Answer'])

# 전처리 객체 생성
p = Preprocess(userdic='./user_dic.tsv')

# 말뭉치 데이터로부터 사전 리스트 생성
dict = []
for c in corpus_data:
    pos = p.pos(c)
    for k in pos:
        dict.append(k[0])
print(dict)

# # 사전에 사용될 word2index 생성
# # 사전의 첫 번째 인덱스에는 OVV 사용
# tokenizer = preprocessing.text.Tokenizer(oov_token='OOV')#, num_words=100000)
# tokenizer.fit_on_texts(dict)
# word_index = tokenizer.word_index

# # 사전 파일 생성
# f = open("chatbot_dict_test_v0.1.1.bin", "wb")
# try:
#     pickle.dump(word_index, f)
# except Exception as e:
#     print(e)
# finally:
#     f.close()

# import json
# with open('chatbot_dict_test_v0.1.1.json', 'w', encoding='utf-8') as f:
#     json.dump(word_index, f, indent=4, ensure_ascii=False)