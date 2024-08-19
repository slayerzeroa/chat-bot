from sentence_transformers import SentenceTransformer, util

class FindAnswer:
  def __init__(self, db):
    self.db = db

  # 검색 쿼리 생성
  def _make_query(self, intent_name, ner_tags):
    sql = "select * from chatbot_train_data"
    if intent_name != None and ner_tags == None:
      # print(intent_name, ner_tags)
      sql = sql + f' where intent="{intent_name}"'
    elif intent_name != None and ner_tags != None:
      where = f' where intent="{intent_name}"'
      if (len(ner_tags) > 0):
        where += ' and ('
        for ne in ner_tags:
          where += f'ner like "{ne}" or'
        where = where[:-3] + ')'
      sql = sql + where
    # 동일한 답변이 2개 이상인 경우 랜덤으로 선택
    sql = sql + " order by rand() limit 1"
    return sql


  # 답변 검색
  def search(self, intent_name, ner_tags):
    #의도명과 개체명으로 답변 검색
    sql = self._make_query(intent_name, ner_tags)
    answer = self.db.select_one(sql)
    # print(answer)
    #검색되는 답변이 없으면 의도명만 검색
    if answer is None:
      sql = self._make_query(intent_name, None)
      answer = self.db.select_one(sql)

    return (answer['answer'])

  # NER 태그를 실제 입력된 단어로 변환
  def tag_to_word(self, ner_predicts, answer):
    for word, tag in ner_predicts:
      # 변환해야 하는 태그가 있는 경우 추가
      if tag == 'B_FOOD':
        answer = answer.replace(tag, word)

    answer = answer.replace('{', '')
    answer = answer.replace('}', '')
    return answer

  def find(self, query):
    self.db.connect()
    # get data from mysql
    sql = "SELECT * FROM chatbot_train_data"
    result = self.db.select_all(sql)
    # print(result)

    query_list = []
    ans_list = []
    for row in result:
      query_list.append(row['query'])
      ans_list.append(row['answer'])

    model = SentenceTransformer('snunlp/KR-SBERT-V40K-klueNLI-augSTS')

    temp_doc = []
    for i in query_list:
      i = i.replace(' ', '')
      temp_doc.append(i)
    temp_test = query.replace(' ', '')

    query_embeddings = model.encode(temp_doc, convert_to_tensor=True)
    input_embeddings = model.encode(temp_test, convert_to_tensor=True)
    cosine_scores = util.pytorch_cos_sim(input_embeddings, query_embeddings)

    org = 0
    temp = cosine_scores[org]
    temp.argsort(descending=True)[0:5]

    a = list(temp)
    return (ans_list[a.index(max(a))])
