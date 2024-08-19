from konlpy.tag import Komoran
import pickle

class Preprocess:
    # 생성자
	def __init__(self, word2index_dic="", userdic=None): 
		# 단어 인덱스 사전 불러오기
		if word2index_dic != "":
			f = open(word2index_dic, "rb")
			self.word_index = pickle.load(f)
			f.close()
		else:
			self.word_index = None

		# 형태소 분석기 초기화
		self.komoran = Komoran(userdic=userdic)

		# 제외할 품사
		# 참조: https://docs.komoran.kr/firststep/postypes.html
		self.exclusion_tags = [
            "VV", "VA", "VX", "VCP", "VCN", # 용언 제거
			"JKS", "JKC", "JKG", "JKO", "JKB", "JKV", "JKQ", "JX", "JC", # 관계언 제거
			"SF", "SP", "SS", "SE", "SO", # 기호 제거
			"EP", "EF", "EC", "ETN", "ETM", # 어미 제거
			"XSN", "XSV", "XSA", "XPN" # 접미사 제거 
		]

	# 형태소 분석기 POS tagger (래퍼 함수)
	def pos(self, sentence):
		return self.komoran.pos(sentence)

	# 불용어 제거 후 필요한 품사 정보만 가져오기
	def get_keywords(self, pos, without_tag=False):
		f = lambda x: x in self.exclusion_tags
		word_list = []
		for p in pos:
			if f(p[1]) is False: # 불용어 리스트에 없는 경우에만 저장
				word_list.append(p if without_tag is False else p[0])
		return word_list

	# 키워드를 단어 인덱스 시퀀스로 변환
	def get_wordidx_sequence(self, keywords):
		if self.word_index is None:
			return []
		w2i = []
		for word in keywords:
			try:
				w2i.append(self.word_index[word])
			except KeyError:
				# 해당 단어가 사전에 없는 경우 OOV 처리
				w2i.append(self.word_index["OOV"])
		return w2i

import pandas as pd
from sentence_transformers import SentenceTransformer, util

train_data = pd.read_csv('train_data.csv')
test_data = pd.read_csv('test_data.csv')

doc_list = train_data["document"].to_list()
ans_list = test_data["Answer"].to_list()

test_text = '형식시험에 꼭 참여해야 하나요?'	# 미입회로 형식시험을 진행할 수 있나요?
# test_text = '제작사에서 자체적으로 발급한 자체 인증서도 인증으로써 효력이 있을까요?'    # 제작사의 자체 인증서도 인증되나요?
# test_text = '형식승인 절차를 수행하기 위해 필요한 시험비와 검사 수수료, 기타 비용 등은 얼마정도 예상할 수 있을까요?'    # 예상되는 시험비 및 검사 수수료는 얼마인가요?
# test_text = '기존에 소모품으로 활용되거나, 유지보수에만 활용되는 소규모의 철도용품들도 형식승인 대상인가요?'    # 유지보수에만 소량으로 사용되는 철도용품도 형식승인 대상인가요?
p = Preprocess()
# train_tmp = p.get_keywords(p.pos(train_data), without_tag=True)
# train_tmp = ' '.join(train_tmp)
train_tmp = []
for sentence in doc_list:
    pos = p.pos(sentence)
    keywords = p.get_keywords(pos, without_tag=True)
    keywords = ' '.join(keywords)
    train_tmp.append(keywords)

test_tmp = p.get_keywords(p.pos(test_text), without_tag=True)
test_tmp = ' '.join(test_tmp)
# print(train_tmp)
print(test_tmp)

# model = SentenceTransformer('sentence-transformers/paraphrase-distilroberta-base-v1')
# model = SentenceTransformer('jhgan/ko-sroberta-multitask')
model = SentenceTransformer('snunlp/KR-SBERT-V40K-klueNLI-augSTS')

# query_embeddings = model.encode(doc_list, convert_to_tensor=True)
# input_embeddings = model.encode(test_text, convert_to_tensor=True)
# cosine_scores = util.pytorch_cos_sim(input_embeddings, query_embeddings)
train_tmp_embeddings = model.encode(train_tmp, convert_to_tensor=True)
test_tmp_embeddings = model.encode(test_tmp, convert_to_tensor=True)
cosine_scores = util.pytorch_cos_sim(test_tmp_embeddings, train_tmp_embeddings)
# print(cosine_scores[0])


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
    print(f"{test_text} <> {i}. {doc_list[i]} \nScore: {cosine_scores[org][i]:.2f}")

print('')
a = list(cosine_scores[org])
print('선택된 질문: ')
print(doc_list[a.index(max(a))])
print('선택된 답변: ')
print(ans_list[a.index(max(a))])