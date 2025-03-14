from config.DatabaseConfig import *
from utils.Database import Database
from utils.Preprocess import Preprocess

# 전처리 객체 생성
p = Preprocess(word2index_dic='../train_tools/dict/chatbot_dict.bin',
                userdic='../utils/user_dic.tsv')

# 질문/답변 학습 디비 연결 객체 생성
db = Database(
    host=DB_HOST, user=DB_USER, password=DB_PASSWORD, db_name=DB_NAME
)
db.connect()

# 원문
query = input("질문을 입력해 주세요: ")

# 의도 파악
from models.intent.IntentModel import IntentModel
intent = IntentModel(model_name='../models/intent/intent_model.h5', preprocess=p)
predict = intent.predict_class(query)
intent_name = intent.labels[predict]

# 개체명 파악
from models.ner.NerModel import NerModel
ner = NerModel(model_name='../models/ner/ner_model_test_v.0.2.h5', preprocess=p)
predicts = ner.predict(query)
ner_tags = ner.predict_tags(query)


print("질문 : ", query)
print("=" * 100)
print("의도 파악 : ", intent_name)
print("개체명 파악 : ", predicts)
print("답변 검색에 필요한 NER 태그 : ", ner_tags)
print("=" * 100)

# 답변 검색
from utils.FindAnswer import FindAnswer

try:
    f = FindAnswer(db)
    answer_text = f.search(intent_name, ner_tags)
    answer = f.tag_to_word(predicts, answer_text)
except:
    answer = "죄송해요 무슨 말인지 모르겠어요"

print("답변 : ", answer)

# 질문과 답변을 디비에 저장
import datetime
now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
sql = """
    INSERT INTO chatbot_log(
        `query`, `intent`, `ner`, `answer`, `answer_image`, `date`)
    VALUES(
        '%s', '%s', '%s', '%s', '%s', '%s'
    )
""" % (query, intent_name, ner_tags, answer, '', now)

db.close()
print("저장 완료")

