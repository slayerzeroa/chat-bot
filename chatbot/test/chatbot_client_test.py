import socket
import json

# 챗봇 엔진 서버 접속 정보
host = "127.0.0.1"
port = 5050

# 클라이언트 프로그램 시작
while True:
    print("질문 : ")
    query = input()
    if (query == "exit"):
        exit(0)
    print("-" * 40)

    # 챗봇 엔진 서버 연결
    mySock = socket.socket()
    mySock.connect((host, port))

    # 챗봇 엔진 질의 요청
    json_data = {
        "Query" : query,
        "BotType" : "MyService"
    }
    message = json.dumps(json_data)
    mySock.send(message.encode())

    # 챗봇 엔진 답변 출력
    data = mySock.recv(2048).decode()
    json_data = json.loads(data)
    print("답변 : ", json_data["Answer"])
    print("-" * 40)

# 챗봇 엔진 서버 연결 소켓 닫기
mySock.close()