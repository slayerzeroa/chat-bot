# test/model_intent_test.py

from utils.Preprocess import Preprocess
from models.intent.IntentModel import IntentModel
import pandas as pd

p = Preprocess(word2index_dic='./train_tools/dict/chatbot_dict_test_v0.1.bin',
               userdic='./utils/user_dic.tsv')

intent = IntentModel(model_name='./models/intent/intent_model_v.0.1.h5', proprocess=p)

train_file = './models/intent/train_data.csv'
data = pd.read_csv(train_file)
query = data['query'].tolist()

for sentence in query:
    predict = intent.predict_class(sentence)
    predict_label = intent.labels[predict]
    print("="*30)
    print(sentence)
    print("의도 예측 클래스 : ", predict)
    print("의도 예측 레이블 : ", predict_label)

# query = "미입회로 형식시험을 진행할 수 있나요?"
# predict = intent.predict_class(query)
# predict_label = intent.labels[predict]
# # print(query, predict_label)
# print("="*30)
# print(query)
# print("의도 예측 클래스 : ", predict)
# print("의도 예측 레이블 : ", predict_label)

# query = "동일한 철도차량의 계약 건마다 새로운 형식승인을 받아야 하나요?"
# predict = intent.predict_class(query)
# predict_label = intent.labels[predict]
# print("="*30)
# print(query)
# print("의도 예측 클래스 : ", predict)
# print("의도 예측 레이블 : ", predict_label)

# query = "철도용품 입찰에 참여하기 위해서는 형식승인과 제작자승인을 모두 받아야 하며 비용은 제작자가 부담해야 하나요?"
# predict = intent.predict_class(query)
# predict_label = intent.labels[predict]
# print("="*30)
# print(query)
# print("의도 예측 클래스 : ", predict)
# print("의도 예측 레이블 : ", predict_label)

# query = "운행승인 신청시 준비해야할 안전성 검토자료는 구체적으로 무엇인가요?"
# predict = intent.predict_class(query)
# predict_label = intent.labels[predict]
# print("="*30)
# print(query)
# print("의도 예측 클래스 : ", predict)
# print("의도 예측 레이블 : ", predict_label)