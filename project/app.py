import pickle
import re
import string
import pandas as pd
import numpy as np
from rank_bm25 import BM25Okapi
from pymorphy2 import MorphAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec, KeyedVectors
from flask import Flask, render_template, request


# все переменные, которые необходимо использовать
# инициализирцем сразу, чтобы не не тратить время flask

morph = MorphAnalyzer()


# для препроцессинга поступающего текста
def preprocessing(data):
    print(str(data))
    punct = ['^', "\\", '.', '?', '!', '\'', ';', '—', '-',
            '<', '>', '\"', '(', ')', '{', '}', '\'', '[', ']',
             '*', '&', '#', '@', '+', '=']
    line = str(data)
    for el in punct:
        line = line.replace(el, '')
    strng = ''
    for word in line.split(' '):
        wrd = morph.parse(word)[0]
        strng += wrd.normal_form + ' '
    return strng


# загружаем данные
def got_data():
    train = pd.read_csv('train_data.csv', sep='\t')
    answers = pd.read_csv('answers_all_data.csv')
    with open('tfidf_ind.pkl', 'rb') as f:
        tf_mat = pickle.load(f)
    with open('wtv_ind.pkl', 'rb') as f2:
        wtv_mat = pickle.load(f2)
    return train, tf_mat, wtv_mat, answers


train, tf_mat, wtv_mat, answers = got_data()
n_conn = train['Номер связки'].tolist()
n_answ = train['Текст вопросов'].tolist()


# tf-idf
def tfidf_ind(train):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(train['вопросы_препроцессинговые'])
    return vectorizer


vectorizer = tfidf_ind(train)


# bm25
def bm25_func():
    corpus1 = train['вопросы_препроцессинговые'].tolist()
    tokenized_corpus1 = [str(doc).split(" ") for doc in corpus1]
    bm251 = BM25Okapi(tokenized_corpus1)
    return bm251


bm25 = bm25_func()


# модель для word to vec
def model_vec():
    model_file = './araneum_none_fasttextcbow_300_5_2018/araneum_none_fasttextcbow_300_5_2018.model'
    model = KeyedVectors.load(model_file)
    return model


model = model_vec()


def normalize_vec(v):
    return v / np.sqrt(np.sum(v ** 2))


# для word to vec матриц
def create_doc_matrix(text):
    lemmas = text.split()
    lemmas_vectors = np.zeros((len(lemmas), model.vector_size))
    vec = np.zeros((model.vector_size,))

    for idx, lemma in enumerate(lemmas):
        if lemma in model.wv:
            lemmas_vectors[idx] = normalize_vec(model.wv[lemma])

    return lemmas_vectors


# для дальнейшего поиска собираем массив матриц
def for_doc_wtv():
    vec_train = []
    for index, row in train.iterrows():
        text1 = create_doc_matrix(row['вопросы_препроцессинговые'])
        vec_train.append(text1)
    return vec_train


vec_wtv_tr = for_doc_wtv()


# word to vec вектора
def doc_vector(text):
    lemmas = text.split()
    lemmas_vectors = np.zeros((len(lemmas), model.vector_size))
    vec = np.zeros((model.vector_size,))
    for idx, lemma in enumerate(lemmas):
        if lemma in model:
            lemmas_vectors[idx] = model[lemma]
    if lemmas_vectors.shape[0] is not 0:
        vec = np.mean(lemmas_vectors, axis=0)

    return vec


# функции поиска ближайших документов по номеру связки


# функция возвращения самого близкого ответа
def ranging_simple(text):
    cnt = np.argmax(text)
    numb = n_conn[cnt]
    if int(numb) == 0:
        res = 'Номер связки 0. Ответа на Ваш вопрос нет'
    else:
        answ = answers[answers['Номер связки'] == int(numb)]['Текст ответа'].tolist()
        res = 'Топ запрос: ' + str(numb) + '. Ответ: ' + str(answ[0])[:350] + "..."
    return res


# функция возвращения самых близк ранжированных запросов
def sorting_results(ndar):
    fin_dict = []
    lst = pd.Series(ndar)
    lar = lst.nlargest(25)
    ind = lar.index.values.tolist()
    for i in ind:
        d = {}
        num_s = n_conn[i]
        ans = n_answ[i]
        an_fin = str(ans)[:350] + "..."
        d['result'] = {'index': str(num_s), 'query': str(an_fin)}
        fin_dict.append(d)
    return fin_dict


# word to vec матрицы
def search_wtv(query_m, reduce_func=np.max, axis=0):
    sims = []
    query = create_doc_matrix(str(query_m))
    for doc in vec_wtv_tr:
        sim = doc.dot(query.T)
        sim = reduce_func(sim, axis=axis)
        sims.append(sim.sum())
    return sims


# word to vec вектора
def sim_vectors(query):
  doc = normalize_vec(doc_vector(str(query)))
  sim = wtv_mat.dot(doc.T)
  return sim


# tf-idf
def text_to_vec(text):
    new_doc = vectorizer.transform([str(text)]).toarray()
    new_matr = tf_mat.dot(new_doc[0])
    return new_matr


# bm25
def search_bm25(text):
    tokenized_query = str(text).split(" ")
    bm_res = bm25.get_scores(tokenized_query)
    print(bm_res)
    print(np.argmax(bm_res))
    return bm_res




app = Flask(__name__)



@app.route('/')
def search_every():
    if request.args:
        text = request.args['text']
        search_method = request.args['search_method']
        if search_method == 'TF-IDF':
            tf_text = text_to_vec(preprocessing(text))
            search_result_an = ranging_simple(tf_text)
            search_result= sorting_results(tf_text)
        elif search_method == 'BM-25':
            bm_text = search_bm25(preprocessing(text))
            search_result_an = ranging_simple(bm_text)
            search_result = sorting_results(bm_text)
        elif search_method == 'WordToVec Mean Vec':
            mean_text = sim_vectors(preprocessing(text))
            search_result_an = ranging_simple(mean_text)
            search_result = sorting_results(mean_text)
        elif search_method == 'WordToVec Matrix':
            matr_text =search_wtv(preprocessing(text))
            search_result_an = ranging_simple(matr_text)
            search_result = sorting_results(matr_text)
        else:
            raise TypeError('unsupported search method')
        return render_template('new.html', search_result_an=search_result_an,
                               search_result=search_result)

    return render_template('index.html')




if __name__ == '__main__':
    app.run(debug=True)
