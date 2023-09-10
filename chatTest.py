import numpy as np
import pandas as pd
from numpy import dot
from numpy.linalg import norm
import urllib.request
from sentence_transformers import SentenceTransformer


train_data = pd.read_pickle ("my_data.pkl")

train_data.head()

model = SentenceTransformer('sentence-transformers/xlm-r-100langs-bert-base-nli-stsb-mean-tokens')

def cos_sim(A, B):
  return dot(A, B)/(norm(A)*norm(B))

def return_answer(question):
    embedding = model.encode(question)
    train_data['score'] = train_data.apply(lambda x: cos_sim(x['embedding'], embedding), axis=1)
    return train_data.loc[train_data['score'].idxmax()]['A']

   
def addQA(question, label, answer):

  global train_data
   
  dfAdd = pd.DataFrame({'Q' : [question], 'A' : [answer], 'label' : [label]})
  dfAdd['embedding'] = dfAdd.apply(lambda row: model.encode(row.Q), axis = 1)

  train_data = pd.concat([train_data,dfAdd], ignore_index=True)

#중복질문삭제
  train_data1 = train_data.drop_duplicates(['Q'], keep='first')

  train_data1.to_pickle ("my_data.pkl")

  train_data = pd.read_pickle ("my_data.pkl")


return_answer('기분이 안좋아')

