### Проект по инфопоиску

----

В репозитории лежит 2 файла: app.py - собственно flask-приложение и matrix.py - где приведен код сбора матриц. Также приложены 2 html-шаблона, которые лежат в папке templates.

По **[ссылке](https://drive.google.com/drive/folders/1LzmAF_Bpp6tKc6sKstMfTVWY0BhOknPi?usp=sharing)** .zip, в котором лежит полностью проект:


* матрицы: tfidf_ind.pkl и wtv_ind.pkl (WordToVec). Обучены на сгенерированном датасете train, по предложениям, прошедшим препроцессинг


* датасеты: answers_all_data.csv - все данные по ответам, queries_all_data.csv - все данные по запросам, train_data.csv - датасет с данными, на которых обучались классификаторы


* .py файлы: описаны выше


* директории: static (пустая, изображений в проекте нет), templates (html-шаблоны), venv и директория с моделью, используемой для WordToVec: araneum_none_fasttextcbow_300_5_2018


----


Для запуска на локальном хосте:


* необходимо скачать .zip файл


* распаковать его


* установить все пакеты, которые не установлены, для работы с проектом (pycharm предлагает установку, к сожалению, pip freeze отказался работать и файл requirements.txt сделать не вышло из-за pip-a)


* убедиться, что все описанные выше файлы лежат в директории с app.py


* запустить flask
