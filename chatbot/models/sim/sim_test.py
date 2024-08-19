import pandas as pd
from sentence_transformers import SentenceTransformer, util
import pymysql
from DatabaseConfig import *
from Database import *

# # connect to mysql
# db = pymysql.connect(
#                     host='127.0.0.1',
#                     port=3306,
#                     user='root',
#                     passwd='1234',
#                     db='ascl_chatbot',
#                     charset='utf8'
#                     )
#
# # create cursor
# cursor = db.cursor()

db = Database(
    host=DB_HOST, user=DB_USER, password=DB_PASSWORD, port=DB_PORT, db_name=DB_NAME, charset=DB_CHARSET)

db.connect()
sql = "SELECT * FROM chatbot_train_data"
result = db.select_all(sql)

# conn = pymysql.connect(
#             host=DB_HOST,
#             user=DB_USER,
#             password=DB_PASSWORD,
#             port=DB_PORT,
#             db=DB_NAME,
#             charset=DB_CHARSET
#         )
#
#
# cursor = conn.cursor()
#
# # get data from mysql
# sql = "SELECT * FROM chatbot_train_data"
#
# cursor.execute(sql)
# result = cursor.fetchall()

# print(db.select_all(sql))

# print(result)

query_list = []
ans_list = []
for row in result:
    query_list.append(row['query'])
    ans_list.append(row['answer'])

# print(query_list)

test_text = '형식시험에 꼭 참여해야 하나요?'
# test_text = '제작사에서 자체적으로 발급한 자체 인증서도 인증으로써 효력이 있을까요?'    # 제작사의 자체 인증서도 인증되나요?
# test_text = '형식승인 절차를 수행하기 위해 필요한 시험비와 검사 수수료, 기타 비용 등은 얼마정도 예상할 수 있을까요?'    # 예상되는 시험비 및 검사 수수료는 얼마인가요?
# test_text = '기존에 소모품으로 활용되거나, 유지보수에만 활용되는 소규모의 철도용품들도 형식승인 대상인가요?'    # 유지보수에만 소량으로 사용되는 철도용품도 형식승인 대상인가요?

# model = SentenceTransformer('sentence-transformers/paraphrase-distilroberta-base-v1')
# model = SentenceTransformer('jhgan/ko-sroberta-multitask')
model = SentenceTransformer('snunlp/KR-SBERT-V40K-klueNLI-augSTS')

temp_doc = []
for i in query_list:
    i = i.replace(' ', '')
    temp_doc.append(i)
temp_test = test_text.replace(' ', '')

query_embeddings = model.encode(temp_doc, convert_to_tensor=True)
input_embeddings = model.encode(temp_test, convert_to_tensor=True)
cosine_scores = util.pytorch_cos_sim(input_embeddings, query_embeddings)

# org = 0
# temp = cosine_scores[org]
# temp.argsort(descending=True)[0:5]

# for i in temp.argsort(descending=True)[0:5]:
#     print(f"{i}. {test_text} <> {doc_list[i]} \nScore: {cosine_scores[org][i]:.2f}")


# doc_list = ['미입회로 형식시험을 진행할 수 있나요?', '형식시험에 꼭 참여해야 하나요?']
# embeddings = model.encode(doc_list, convert_to_tensor=True)
# cosine_scores = util.pytorch_cos_sim(embeddings, embeddings)

# print(f"{doc_list[0]} <> {doc_list[1]} \nScore: {cosine_scores[0][1]:.2f}")

org = 0
temp = cosine_scores[org]
temp.argsort(descending=True)[0:5]

for i in temp.argsort(descending=True)[0:5]:
    print(f"{test_text} <> {i}. {query_list[i]} \nScore: {cosine_scores[org][i]:.2f}")

print('')
a = list(cosine_scores[org])
print('선택된 질문: ')
print(query_list[a.index(max(a))])
print('선택된 답변: ')
print(ans_list[a.index(max(a))])