import threading
import json

from config.DatabaseConfig import *
from utils.Database import Database
from utils.BotServer import BotServer
from utils.Preprocess import Preprocess
from models.intent.IntentModel import IntentModel
from models.ner.NerModel import NerModel
from utils.FindAnswer import FindAnswer

# 전처리 객체 생성
p = Preprocess(word2index_dic='./train_tools/dict/chatbot_dict.bin',
                userdic='./utils/user_dic.tsv')

# 의도 파악 모델
intent = IntentModel(model_name='./models/intent/intent_model.h5', preprocess=p)

# 개체명 인식 모델
ner = NerModel(model_name='./models/ner/ner_model_test_v.0.2.h5', preprocess=p)

def to_client(conn, addr, params):
    db = Database(
        host=DB_HOST,
        user=DB_USER,
        password=DB_PASSWORD,
        db_name =DB_NAME,
        port=DB_PORT,
        charset=DB_CHARSET
    )
    try:
        db.connect()

        # 데이터 수신
        read = conn.recv(2048)
        print("========================================================")
        print("Connection from: %s" % str(addr))

        if read is None or not read:
            print("클라이언트 접속 종료")
            exit(0)

        # json 데이터로 변환
        recv_json_data = json.loads(read.decode())
        print("데이터 수신 : ", recv_json_data)
        query = recv_json_data['Query']

        # 의도 파악
        intent_predict = intent.predict_class(query)
        intent_name = intent.labels[intent_predict]
        print(intent_name)
        # 개체명 파악
        ner_predicts = ner.predict(query)
        ner_tags = ner.predict_tags(query)
        print(ner_tags)
        # 답변 검색
        try:
            f = FindAnswer(db)
            # answer_text = f.search(intent_name, ner_tags)
            # print(answer_text)
            # answer = f.tag_to_word(ner_predicts, answer_text)

            answer = f.find(query)
            # print(answer)
        except:
            answer = "죄송해요 무슨 말인지 모르겠어요"
            answer_image = None

        send_json_data_str = {
            "Query" : query,
            "Answer" : answer,
            # "AnswerImageUrl" : answer_image,
            "Intent" : intent_name,
            "NER" : str(ner_predicts)
        }
        message = json.dumps(send_json_data_str) # json 객체를 전송 가능한 문자열로 변환
        conn.send(message.encode()) # 응답 전송

    except Exception as e:
        print(e)

    finally:
        if db is not None:
            db.close()
        conn.close()
        print("데이터 송수신 종료")

if __name__ == "__main__":
    db = Database(
        host=DB_HOST,
        user=DB_USER,
        password=DB_PASSWORD,
        db_name = DB_NAME,
        port=DB_PORT,
        charset=DB_CHARSET
    )
    print("DB 접속")

    # 봇 서버 동작
    port = 5050
    listen = 100
    bot = BotServer(port, listen)
    bot.create_sock()
    print("bot start")

    while True:
        conn, addr = bot.ready_for_client()
        params = {
            'db' : db
        }
        client = threading.Thread(target=to_client, args=(conn, addr, params))
        client.start()