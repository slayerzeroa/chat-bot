import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # 0 = all messages are logged (default behavior), 1 = INFO messages are not printed, 2 = INFO and WARNING messages are not printed, 3 = INFO, WARNING, and ERROR messages are not printed
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from konlpy.tag import Komoran

from utils.Preprocess import Preprocess

data = pd.read_csv('models/intent/train_data.csv')

# 전처리 객체 생성
p = Preprocess(userdic='./utils/user_dic.tsv')

data_tokenized = [p.pos(text_) for text_ in data['query']]
# pos = data_tokenized[0]
# keywords = p.get_keywords(pos, without_tag=True)

data_list = []
for i in range(len(data_tokenized)):
    pos = data_tokenized[i]
    keywords = p.get_keywords(pos, without_tag=True)
    data_list.append(keywords)

num_tokens = [len(tokens) for tokens in data_list]
num_tokens = np.array(num_tokens)

# 평균값, 최댓값, 표준편차
print(f"토큰 길이 평균: {np.mean(num_tokens)}")
print(f"토큰 길이 최대: {np.max(num_tokens)}")
print(f"토큰 길이 표준편차: {np.std(num_tokens)}")

# plt.title('all text length')
# plt.hist(num_tokens, bins=100)
# plt.xlabel('length of samples')
# plt.ylabel('number of samples')
# plt.show()

select_length = 17

def below_threshold_len(max_len, nested_list):
    cnt = 0
    for s in nested_list:
        if(len(s) <= max_len):
            cnt = cnt + 1
        
    print('전체 샘플 중 길이가 %s 이하인 샘플의 비율: %0.2f%%'%(max_len, (cnt / len(nested_list) * 100)))
    
below_threshold_len(select_length, data_list)