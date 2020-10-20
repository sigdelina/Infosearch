import gensim
import pandas as pd
import pickle as pkl
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec, KeyedVectors


# получение матриц для tf-idf, word_to_vec на векторах
# для bm25 лучше использовать rank_bm25 - она работает намного быстрее, чем функция,
# которую я написала сама в дз2 --> импортировать ее буду в app.py
# для word_to_vec по экспериментальному способу тоже лучше использовать в app.py



RANDOM_STATE = 35


def got_data():
    answers = pd.read_csv('answers_all_data.csv')
    queries = pd.read_csv("queries_all_data.csv")
    queries = queries.fillna(0)
    for_queries = queries[['Номер связки\n', 'Текст вопроса',
                           'вопросы_препроцессинговые',
                           'natasha', 'deeppavlov']]
    for_queries = for_queries.rename(columns={'Номер связки\n': 'Номер связки',
                                              'Текст вопроса': 'Текст вопросов',
                                              'natasha': 'natasha',
                                              'deeppavlov': 'deeppavlov'})
    for_answers = answers[['Номер связки', 'Текст вопросов',
                           'вопросы_препроцессинговые',
                           'natasha', 'deeppavlov']]

    train_df, test = train_test_split(for_queries, test_size=0.3,
                                      random_state=RANDOM_STATE)

    train = pd.concat([for_answers, train_df])

    return train, test


def tfidf_ind(train):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(train['вопросы_препроцессинговые'])
    matr_answers = X.toarray()
    with open('tfidf_ind.pkl', 'wb') as f:
        pkl.dump(matr_answers, f)


def model_vec():
    model_file = './araneum_none_fasttextcbow_300_5_2018/araneum_none_fasttextcbow_300_5_2018.model'
    model = KeyedVectors.load(model_file)
    return model


def normalize_vec(v):
    return v / np.sqrt(np.sum(v ** 2))


def doc_vector(text, model):
    lemmas = text.split()
    lemmas_vectors = np.zeros((len(lemmas), model.vector_size))
    vec = np.zeros((model.vector_size,))
    for idx, lemma in enumerate(lemmas):
        if lemma in model:
            lemmas_vectors[idx] = model[lemma]
    if lemmas_vectors.shape[0] is not 0:
        vec = np.mean(lemmas_vectors, axis=0)

    return vec


def fin_wtv_ind(train, model):
    for_wth_pr = []
    for index, row in train.iterrows():
        fin = doc_vector(row['вопросы_препроцессинговые'], model)
        for_wth_pr.append(normalize_vec(fin))
    matr = np.array(for_wth_pr)
    with open('wtv_ind.pkl', 'wb') as f:
        pkl.dump(matr, f)


def main():
    train, test = got_data()
    # записываем в файл для удобной работы с индексами
    train.to_csv('train_data.csv', sep="\t", index=False)
    # матрица tf-idf
    tf = tfidf_ind(train)
    model = model_vec()
    # матрица word to vec класс. способ
    vc_wtv = fin_wtv_ind(train, model)


main()
