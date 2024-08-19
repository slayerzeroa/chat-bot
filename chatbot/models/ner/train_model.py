import sys, os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import preprocessing
from sklearn.model_selection import train_test_split
import numpy as np
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))
from utils.Preprocess import Preprocess

# 학습 파일 불러오기
def read_file(file_name):
    sents = []
    with open(file_name, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for idx, l in enumerate(lines):
            if l[0] == ';' and lines[idx + 1][0] == '$':
                this_sent = []
            elif l[0] == '$' and lines[idx - 1][0] == ';':
                continue
            elif l[0] == '\n':
                sents.append(this_sent)
            else:
                this_sent.append(tuple(l.split()))
    return sents

p = Preprocess(word2index_dic='../../train_tools/dict/chatbot_dict_test_v0.1.bin',
               userdic='../../utils/user_dic.tsv')

# 학습용 말뭉치 데이터를 불러옴
corpus = read_file('ner_train.txt')

# 말뭉치 데이터에서 단어와 BIO 태그만 불러와 학습용 데이터셋 생성
sentences, tags = [], []
for t in corpus:
    tagged_sentence = []
    sentence, bio_tag = [], []
    for w in t:
        tagged_sentence.append((w[1], w[3]))
        sentence.append(w[1])
        bio_tag.append(w[3])
    
    sentences.append(sentence)
    tags.append(bio_tag)


print("샘플 크기 : \n", len(sentences))
print("0번 째 샘플 단어 시퀀스 : \n", sentences[0])
print("0번 째 샘플 bio 태그 : \n", tags[0])
print("샘플 단어 시퀀스 최대 길이 :", max(len(l) for l in sentences))
print("샘플 단어 시퀀스 평균 길이 :", (sum(map(len, sentences))/len(sentences)))

# 토크나이저 정의
# p = preprocessing.text.Tokenizer(oov_token='OOV')
# p.fit_on_texts(sentences)
tag_tokenizer = preprocessing.text.Tokenizer(lower=False) # 태그 정보는 lower=False 소문자로 변환하지 않는다.
tag_tokenizer.fit_on_texts(tags)

# 단어사전 및 태그 사전 크기
vocab_size = len(p.word_index) + 1
tag_size = len(tag_tokenizer.word_index) + 1
print("BIO 태그 사전 크기 :", tag_size)
print("단어 사전 크기 :", vocab_size)

# 학습용 단어 시퀀스 생성
x_train = [p.get_wordidx_sequence(sent) for sent in sentences]
# x_train = p.texts_to_sequences(sentences)
y_train = tag_tokenizer.texts_to_sequences(tags)

index_to_ner = tag_tokenizer.index_word # 시퀀스 인덱스를 NER로 변환 하기 위해 사용
index_to_ner[0] = 'PAD'

# 시퀀스 패딩 처리
max_len = 40
x_train = preprocessing.sequence.pad_sequences(x_train, padding='post', maxlen=max_len)
y_train = preprocessing.sequence.pad_sequences(y_train, padding='post', maxlen=max_len)

# 학습 데이터와 테스트 데이터를 8:2의 비율로 분리
x_train, x_test, y_train, y_test = train_test_split(x_train, y_train,
                                                    test_size=.2,
                                                    random_state=1234)

# 출력 데이터를 one-hot encoding
y_train = tf.keras.utils.to_categorical(y_train, num_classes=tag_size)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=tag_size)

print("학습 샘플 시퀀스 형상 : ", x_train.shape)
print("학습 샘플 레이블 형상 : ", y_train.shape)
print("테스트 샘플 시퀀스 형상 : ", x_test.shape)
print("테스트 샘플 레이블 형상 : ", y_test.shape)


# 모델 정의 (Bi-LSTM)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional
from tensorflow.keras.optimizers import Adam

model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=30, input_length=max_len, mask_zero=True))
model.add(Bidirectional(LSTM(200, return_sequences=True, dropout=0.50, recurrent_dropout=0.25)))
model.add(TimeDistributed(Dense(tag_size, activation='softmax')))
model.compile(loss='categorical_crossentropy', optimizer=Adam(0.01), metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=128, epochs=10)

print("평가 결과 : ", model.evaluate(x_test, y_test)[1])
# models.save('ner_model_test.h5')

# 시퀀스를 NER 태그로 변환
def sequences_to_tag(sequences):  # 예측값을 index_to_ner를 사용하여 태깅 정보로 변경하는 함수.
    result = []
    for sequence in sequences:  # 전체 시퀀스로부터 시퀀스를 하나씩 꺼낸다.
        temp = []
        for pred in sequence:  # 시퀀스로부터 예측값을 하나씩 꺼낸다.
            pred_index = np.argmax(pred)  # 예를 들어 [0, 0, 1, 0 ,0]라면 1의 인덱스인 2를 리턴한다.
            temp.append(index_to_ner[pred_index].replace("PAD", "O"))  # 'PAD'는 'O'로 변경
        result.append(temp)
    return result

# 테스트 데이터셋의 NER 예측
y_predicted = model.predict(x_test)
pred_tags = sequences_to_tag(y_predicted) # 예측된 NER
test_tags = sequences_to_tag(y_test)    # 실제 NER

# F1 스코어 계산을 위해 사용
from seqeval.metrics import f1_score, classification_report
print(classification_report(test_tags, pred_tags))
print("F1-score: {:.1%}".format(f1_score(test_tags, pred_tags)))

model = tf.keras.models.load_model('ner_model_test.h5')

# 새로운 유형의 문장 NER 예측
word_to_index = tag_tokenizer.word_index
new_sentence = '오늘 13시 30분 짜장면 주문할게요.'
pos = p.pos(new_sentence)
keywords = p.get_keywords(pos, without_tag=True)
new_seq = p.get_wordidx_sequence(keywords)
print("새로운 유형의 시퀀스 : ", new_seq)

new_x = []
for w in keywords:
    try:
        new_x.append(word_to_index.get(w, 1))
    except KeyError:
        new_x.append(word_to_index['OOV'])
new_padded_seqs = preprocessing.sequence.pad_sequences([new_x], padding='post', value=0, maxlen=max_len)
print("새로운 유형의 시퀀스 : ", new_x)
print("새로운 유형의 시퀀스 : ", new_padded_seqs)

# from tensorflow.keras.models import Model, load_model
# NER 예측
# models = load_model('ner_model_test.h5')
mp = model.predict(np.array([new_padded_seqs[0]]))
mp = np.argmax(mp, axis=-1)   # 예측된 NER 인덱스값 추출

print("{:10} {:5}".format("단어", "예측된 NER"))
print("-"*50)
for w, pred in zip(keywords, mp[0]):
    print("{:10} {:5}".format(w, index_to_ner[pred]))