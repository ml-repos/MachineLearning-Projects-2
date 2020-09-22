import numpy as np
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns; sns.set()
from sklearn.datasets import fetch_20newsgroups
# data = fetch_20newsgroups()
categories = ['alt.atheism','comp.graphics','comp.os.ms-windows.misc','comp.sys.ibm.pc.hardware','comp.sys.mac.hardware',
 'comp.windows.x','misc.forsale','rec.autos','rec.motorcycles','rec.sport.baseball','rec.sport.hockey','sci.crypt',
 'sci.electronics','sci.med','sci.space','soc.religion.christian','talk.politics.guns','talk.politics.mideast','talk.politics.misc',
 'talk.religion.misc']
# with open('./20newsdata.pkl', 'wb') as f:
#     pickle.dump(data, f)
with open('./20newsdata.pkl', 'rb') as f:
    data =pickle.load(f)

train = fetch_20newsgroups(subset='train', categories=categories,
                                shuffle=True)

test = fetch_20newsgroups(subset='test', categories=categories,
                               shuffle=True)

print(len(train.data))

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

model = make_pipeline(TfidfVectorizer(), MultinomialNB())
model.fit(train.data, train.target)
labels = model.predict(test.data)

def srch(s,train=train, model=model):
    pred=model.predict([s])
    return train.target_names[pred[0]]

print(srch('glock'))
