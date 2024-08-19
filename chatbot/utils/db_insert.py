import pymysql
import pandas as pd
import random
# read xlsx file
test = pd.read_csv('./../../data/test_data.csv')
train = pd.read_csv('./../../data/train_data.csv')

# print(test, train)

ner_list = ['O', 'B_DT', 'B_FOOD', 'I', 'B_OG', 'B_PS', 'B_LC', 'NNP', 'B_TI', 'PAD']
ins_ner = []
for i in range(len(test)):
    ins_ner.append(random.choice(ner_list))

data = pd.DataFrame({'id': [idx for i, idx in enumerate(range(len(test)))], 'intent': test.Intent, 'ner': ins_ner, 'query': test.Query, 'answer': test.Answer})



# connect to mysql
db = pymysql.connect(
                    host='127.0.0.1',
                    port=3306,
                    user='root',
                    passwd='1234',
                    db='ascl_chatbot',
                    charset='utf8'
                    )

# create cursor
cursor = db.cursor()

# insert sql
sql = "INSERT INTO chatbot_train_data(id, intent, ner, query, answer) VALUES('%s', '%s','%s','%s','%s')"

# execute sql
for i, row in data.iterrows():
    cursor.execute(sql % (row['id'], row['intent'], row['ner'], row['query'], row['answer']))

# commit
db.commit()

# close
db.close()
