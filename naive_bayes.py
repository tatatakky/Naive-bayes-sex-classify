# -*- coding:utf-8 -*-
import numpy as np, sys
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB


txtbody = open('./data/names_hiragana.txt', encoding='utf-8')
jnames = np.array([x.split() for x in txtbody], dtype='U12')
names_train, gender_train, = jnames[:, 1], jnames[:, 0]

name = sys.argv[1]

# ひらがなの読みを2文字ごとに分割
def split_in_2words(name):
    return [name[i:i+2] for i in range(len(name)-1)]

bow_t = CountVectorizer(analyzer=split_in_2words).fit(names_train)

names_bow = bow_t.transform(names_train)

# TF-IDFを用いてデータの重み付けと正規化を行う
tfidf_t = TfidfTransformer().fit(names_bow)

# 文字列の重み付けと正規化を行う
names_tfidf = tfidf_t.transform(names_bow)

# 学習
namegender_detector = MultinomialNB().fit(names_tfidf, gender_train)

# 性別判定
def predict_gender(name):
    bow = bow_t.transform([name])
    n_tfidf = tfidf_t.transform(bow)
    return namegender_detector.predict(n_tfidf)[0]

print("Name : {}".format(name))
print("Sex  : {}".format(predict_gender(name)))
