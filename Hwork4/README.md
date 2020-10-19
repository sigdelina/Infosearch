Файлы: sem4_semantics.ipynb - дз с изначальным решением


Код можно апгрейдить для векторов: делаем так, что не только в матрице были нормированные вектора, но и нормированный вектор умножался на матрицу. Тогда функция будет выглядеть так:



```
def sim_vectors(matr, query):
    doc = normalize_vec(doc_vector(query))
    sim = matr.dot(doc.T)
    return sim
```

