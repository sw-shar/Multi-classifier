import sqlite3
import re
import pandas as pd
import numpy as np
import math
from itertools import zip_longest
from collections import defaultdict
import torch
import time
import random

import matplotlib.pyplot as plt
from matplotlib import rc
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from collections import defaultdict
import pandas as pd
import numpy as np
import math
from itertools import zip_longest
from collections import defaultdict
import torch
import time
import torch
import torch.nn as nn
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import warnings

warnings.filterwarnings("ignore")


class ConvNet(nn.Module):
    def __init__(
        self,
        number_count_,
        n_tokens=88,
        emb_size=20,
        kernel_sizes=[2, 3, 4],
    ):
        super().__init__()
        # создаем эмбеддинги собственные
        self.embeddings = torch.nn.Embedding(n_tokens, emb_size)

        # или берта
        # self.enconder = bertmodel
        # создаем сверточные слои
        convs = [
            nn.Conv1d(in_channels=emb_size, out_channels=100, kernel_size=kernel_size)
            for kernel_size in kernel_sizes
        ]

        self.conv_modules = nn.ModuleList(
            convs
        )  # лист модулей по которым можно делать цикл
        # и в этом цикли ты будешь применять одни и теже операции для каждой свертки
        # и добавлять их в лист фичей feature_list
        self.drop = nn.Dropout()
        self.linear = nn.Linear(
            3 * 100, number_count_
        )  # линейный слой, 100 выходных каналов у каждого свертоного слоя, и мы их будем конкретенировать
        self.softmax = nn.Softmax()

    def forward(self, batch):
        embeddings = self.embeddings(torch.LongTensor(batch["text"]))
        embeddings = embeddings.transpose(
            1, 2
        )  # (batch_size, wordvec_size, sentence_length)

        feature_list = []
        for conv in self.conv_modules:
            feature_map = torch.nn.functional.relu(conv(embeddings))
            max_pooled, argmax = feature_map.max(dim=2)
            feature_list.append(max_pooled)

        features = torch.cat(feature_list, dim=1)  # конкретенируем фичи
        features = self.drop(features)
        linear = self.linear(features)
        return linear

    def predict(self, batch):
        return self.softmax(self.forward(batch))


import sys


def set_module_var(module_name, variable_name, value):
    # sys.modules["__main__"].__dict__[name] = value
    setattr(sys.modules[module_name], variable_name, value)


set_module_var("__main__", "ConvNet", ConvNet)

"""## Функции"""

# Commented out IPython magic to ensure Python compatibility.
def image_graph(history):
    #   %config InlineBackend.figure_format='retina'
    sns.set(style="whitegrid", palette="muted", font_scale=1.2)
    HAPPY_COLORS_PALETTE = [
        "#01BEFE",
        "#FFDD00",
        "#FF7D00",
        "#FF006D",
        "#ADFF02",
        "#8F00FF",
    ]
    sns.set_palette(sns.color_palette(HAPPY_COLORS_PALETTE))
    rcParams["figure.figsize"] = 12, 8

    plt.plot([i.cpu() for i in history["train_acc"]], label="train accuracy")
    plt.plot([i.cpu() for i in history["val_acc"]], label="validation accuracy")
    # plt.plot(history['val_acc'].cpu(), label='validation accuracy')

    plt.title("Training history")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend()
    plt.ylim([0, 1])


def show_confusion_matrix(confusion_matrix):
    hmap = sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues")
    hmap.yaxis.set_ticklabels(hmap.yaxis.get_ticklabels(), rotation=0, ha="right")
    hmap.xaxis.set_ticklabels(hmap.xaxis.get_ticklabels(), rotation=30, ha="right")
    plt.ylabel("True sentiment")
    plt.xlabel("Predicted sentiment")


def confusion_matrix_print(df, name, y_test, y_pred):
    """
    Рисуем матрицу для конкретного столбца предсказаний
    """
    new_dict_name = {value: key for key, value in enumerate(df[name].unique())}
    cm = confusion_matrix(y_test, y_pred, labels=list(new_dict_name.values()))
    df_cm = pd.DataFrame(cm, index=new_dict_name, columns=new_dict_name)
    show_confusion_matrix(df_cm)


def make_me_small(df):
    """
    Обрезаем датасет по минимальному количеству элементов, чтобы у всех было одинаково
    """
    minimum = df.drop_duplicates().marka_model.value_counts().min()
    a = []
    for i in df.marka_model.value_counts().index.to_list():
        ss = df.loc[df["marka_model"] == i][:minimum].to_numpy()
        for b in ss:
            a.append(b)
    return pd.DataFrame(a, columns=["name", "marka", "model", "marka_model"])


def prepare_df_func(
    number,
    model_do_you_need=True,
    oll_model_do_you_need=[210, 250, 225],
    start_way="/content/drive/MyDrive/учеба/ВЫШКА/ДИПЛОМ/Example/all_frame_model.csv",
    count_out_model=9,
    rovno=False,
):
    """
    Функция для подготовки датасета для обучения моделей
    start_way = путь до датасета с описаниями марок и моделей
    number - количество групп
    model_do_you_need - если нужно выбрать определенные модели
    oll_model_do_you_need - цифры тех модели которые нужны в выборке ([210, 250, 225]) - самые проблемные
    count_out_model - общее количество моделей( и подмоделей) которые мы получим на выходе
    rovno - нужно ли делать одинаковое количество всем моделей
    """
    baza_name_marke_model = pd.read_csv(start_way, index_col=False).drop(
        "Unnamed: 0", axis=1
    )
    if model_do_you_need:
        baza_name_marke_model = baza_name_marke_model[
            baza_name_marke_model["cifra"].isin(oll_model_do_you_need)
        ]

    spare_parts_name = {
        "Насос основной": [
            "гидронасос",
            "станция",
            "гидростанция",
            "насос",
            "main pump",
            "гидравлический насос",
            "основной насос",
            "насос основной",
            "основной гидравлический насоc",
            "главный насос",
            "гидронасос экскаватора",
            "гидронасос на экскаватор",
            "насос сдвоенный",
            "гидронасос сдвоенный",
        ],
        "Гидромотор хода": [
            "гидромотор хода",
            "Г/мотор хода",
            "мотор хода",
            "гидромотор к экскаватору для ходовой части",
            "гидро мотор хода",
        ],
        "Гидромотор поворота": [
            "гидромотор поворота",
            "Г/мотор поворота",
            "мотор поворота",
        ],
        "Редуктор хода": [
            "редуктор хода",
            "бортовая",
            "бортовой редуктор",
            "редуктор хода без мотора",
            "редуктор хода без гидромотора",
            "редуктор хода без г/мотора",
            "ходовой редуктор",
            "бортовой редуктор без гидромотора",
            "редуктор хода бортовой",
            "ходовой редуктор",
        ],
        "Редуктор поворота": [
            "редуктор поворота",
            "поворотный редуктор",
            "редуктор поворота без г/мотора",
            "редуктор поворота без гидромотора",
            "система поворота платформы",
            "механизм поворота",
            "механизм поворота платформы",
            "редуктор поворота башни",
            "редуктор механизма поворота",
            "поворотный редуктор экскаватора",
            "поворотный мотор редуктор",
            "купить поворотный редуктор на экскаватор",
            "поворотная платформа редуктора поворота",
            "редуктора поворота башни ",
        ],
        "Редуктор хода в сборе": [
            "редуктор в сборе",
            "редуктор с мотором",
            "бортовая с мотором",
            "бортовая в сборе",
            "бортовой редуктор с мотором",
            "редуктор хода в сборе",
            "редуктор хода в сборе с гидромотором",
            "редуктор хода в сборе с г/мотором",
            "редуктор хода с гидромотором",
            "редуктор хода с мотором",
            "редуктор хода в сборе с корпусом г/мотора",
            "бортовой редуктор в сборе",
            "ходовой редуктор в сборе",
            "ходовой редуктор с мотором",
            "бортовой редуктор хода в сборе с гидромотором",
            "редуктор хода с гидромотором",
        ],
    }

    test_spare_parts = {
        "Насос основной": [
            "насос на экскаватор",
            "экскаваторный насос",
        ]
    }
    marka_mame = {
        "Hyundai": ["Hyundai", "Huyndai", "Хёндай", "Хендай", "Хундай", "Хенда"],
        "Volvo": [
            "Volvo",
            "Вольво",
        ],
        "Doosan": [
            "Doosan",
            "Дасан",
            "Дусан",
            "Доосан",
            "Досан",
            "Дассан",
        ],
    }

    if len(list(spare_parts_name.keys())) < number:
        number == len(list(spare_parts_name.keys()))

    # получаем наш датасет
    df = tare_me_oll_parts(
        spare_parts_name, baza_name_marke_model, number, longer=False
    )

    # какоче количество исходящих сочетаний марок/моделей нам нужно
    if len(df.marka_model.value_counts()) < count_out_model:
        count_out_model = len(df.marka_model.value_counts())
        list_model = list(df.marka_model.value_counts().index)[:count_out_model]
    else:
        list_model = list(df.marka_model.value_counts().index)[:count_out_model]

    df = df[df["marka_model"].isin(list_model)]

    class_names = {value: key for key, value in enumerate(df["marka_model"].unique())}
    df["marka_model_bert"] = df["marka_model"].map(class_names)

    if rovno:
        df = make_me_small(df)

    return class_names, df


def tare_me_oll_parts(spare_parts_name, baza_name_marke_model, number, longer=False):
    """
    Функция которая готовит нашу базу для обучения
    spare_parts_name - словарь {групппа - варианты синонимов запчастей}
    baza_name_marke_model - df - [marka_,	model,	marka,	model_normal,	cifra]
    longer - брать больше с названиями запчастей (усложняет работу модели) или просто марка модель
    """

    baza_name_marke_model = baza_name_marke_model.dropna()
    spisok_zaprosov = []
    dict_keys = spare_parts_name.keys()
    for name in dict_keys:  # по всем ключам в названиях групп
        for diffferent_name in spare_parts_name[
            name
        ]:  # по всем названиям в одной группе
            for (
                row
            ) in baza_name_marke_model.to_numpy():  # по всем номерам из базы номеров
                marka_of_parts = row[0]
                model_of_parts = row[1]
                model_of_parts_cifra = str(int(row[4]))
                marka_of_parts_normal = row[2]
                model_of_parts_normal = row[3]

                spisok1 = [
                    diffferent_name.lower()
                    + " "
                    + marka_of_parts
                    + " "
                    + model_of_parts
                ] + [marka_of_parts_normal, model_of_parts_normal]
                if marka_of_parts_normal == "Volvo":
                    spisok2 = [
                        diffferent_name
                        + " "
                        + marka_of_parts
                        + " "
                        + model_of_parts_cifra
                    ] + [marka_of_parts_normal, model_of_parts_normal]
                    spisok5 = [
                        marka_of_parts
                        + " "
                        + model_of_parts_cifra
                        + " "
                        + diffferent_name
                    ] + [marka_of_parts_normal, model_of_parts_normal]
                else:
                    spisok2 = [
                        diffferent_name
                        + " "
                        + marka_of_parts
                        + " "
                        + model_of_parts_cifra
                    ] + [marka_of_parts_normal, model_of_parts_cifra]
                    spisok5 = [
                        marka_of_parts
                        + " "
                        + model_of_parts_cifra
                        + " "
                        + diffferent_name
                    ] + [marka_of_parts_normal, model_of_parts_cifra]

                spisok3 = [diffferent_name + " " + model_of_parts] + [
                    marka_of_parts_normal,
                    model_of_parts_normal,
                ]
                spisok4 = [
                    marka_of_parts + " " + model_of_parts + " " + diffferent_name
                ] + [marka_of_parts_normal, model_of_parts_normal]
                spisok6 = [model_of_parts + " " + diffferent_name] + [
                    marka_of_parts_normal,
                    model_of_parts_normal,
                ]

                if longer:
                    spisok_zaprosov.extend([spisok1, spisok2, spisok3])
                else:
                    spisok_zaprosov.extend(
                        [spisok6, spisok2, spisok5, spisok1]
                    )  # spisok2, spisok5
    # вот тут проыерить ту ли он модель береттт...для записи в таргет
    # он наверное для всех моделей берет человеческую запись
    df = pd.DataFrame(spisok_zaprosov, columns=["name", "marka", "model"])
    df["marka_model"] = df["marka"] + "/" + df["model"]
    return df


# df.to_csv('/content/drive/MyDrive/учеба/ВЫШКА/ДИПЛОМ/Example/name_marka_model_400000.csv')


def filtr_baza(df, counts=25):
    """ """
    mm = df.groupby("model")["name"].count() > counts
    aa = mm.loc[mm].index.to_list()
    df = df.query("model in @aa").copy()
    return df


import random


def zapros_example_graph1(df, counts=5):
    """
    Формируем пользовательский запрос
    counts - количество разделений одного запроса
    для обучения марка-модель
    формируем запрос исходя из того, что 50% выборки будет состоять из марки и полной модели
    30 % - из марки и частичной модели
    20% - только из модели
    """
    spisok_zaprosov = []
    for row in df:
        for i in range(counts):
            # нам нужен разномный выбор чтобы разнообразить наше описание товара
            random_choice = random.random()

            # входные данные
            name_of_parts = [row[0]]
            marka_of_parts = [row[1]]
            model_of_parts = [row[2]]

            random_name_parts = random.choices(spare_parts_name[row[0]])

            random_name_marka = random.choices(marka_mame[row[1]])

            # срез из словаря марок-моделей с пользовательском описании
            sr = (
                baza_name_marke_model.loc[baza_name_marke_model.model_normal == row[2]]
                .reset_index(drop=True)
                .copy()
            )
            # рандомно выбираем модель
            random_model = sr.model.tolist()
            # сюда пишем номер (конкретное число) из модели
            random_model_munber = str(int(sr.cifra[0]))

            if len(random_model) == 0:
                print("В базе не хватает модели - ", row[2])

            # выбираем рандомную модель из словаря
            random_name_model = random.choices(random_model)

            # print(random_choice)
            # разделение на 2 ветки в зависимости от вероятности
            if random_choice > 0.5:
                spisok = " ".join(random_name_parts + random_name_marka + [row[2]])
                spisok_zaprosov.append([spisok] + [row[0]] + [row[1]] + [row[2]])
            elif 0.2 < random_choice < 0.5:
                spisok = " ".join(
                    random_name_parts + random_name_marka + [random_model_munber]
                )
                spisok_zaprosov.append(
                    [spisok] + [row[0]] + [row[1]] + [random_model_munber]
                )
            else:
                spisok = " ".join(random_name_parts + [row[2]])
                spisok_zaprosov.append([spisok] + [row[0]] + [row[1]] + [row[2]])

    return spisok_zaprosov


def func_dly_avito(df, stolb, stolb2):
    """
    Функция возвращает номера / или модели - из названия и описания в виде строки без дублей
    """
    df["len"] = df[stolb2].apply(lambda x: len(x))
    df["oll"] = np.where(df["len"] < 4, df[stolb2] + df[stolb], df[stolb])
    return df["oll"].apply(lambda x: ", ".join(str(i) for i in list(set(x))))


def create_triplet_df(df):
    """
    Ввод:
    df : таблица

    В нашем примере данные для каждого товара представленны в виде списка:
    Запрос - Название товара - Марка - Модель - Номер - Общий номер(который включает в себя все одинаковые товары)

    Для Knowledge Graph мы составили структуру:
    Запрос- Отношение(Название товара) - Название
    Номер - Отношение(Название товара) - Название


    Например:

    head                                relation    tail
    K1006550	                          Название	  Насос основной
    Huyndai r180lc-7 механизм поворота	Марка	      Huyndai

    """

    # примеры с номерами
    triplet_example_number = [[row[4], "Общий_номер", row[5]] for row in df]  # +\
    # [[row[4], 'Название', row[1]] for row in df] +\
    # [[row[4], 'Марка', row[2]]for row in df]+\
    # [[row[4], 'Модель', row[3]]for row in df]

    # примеры с запросами
    triplet_example_zapros = (
        [[row[0], "Название", row[1]] for row in df]
        + [[row[0], "Марка", row[2]] for row in df]
        + [[row[0], "Модель", row[3]] for row in df]
        + [[row[0], "Общий_номер", row[5]] for row in df]
    )

    answer = pd.DataFrame(
        triplet_example_number + triplet_example_zapros,
        columns=["head", "relation", "tail"],
    ).drop_duplicates(ignore_index=True)

    return answer


import random


def zapros_example(df, counts=5):
    """
    Формируем пользовательский запрос
    counts - количество разделений одного запроса
    """
    spisok_zaprosov = []
    for row in df:
        for i in range(counts):
            # нам нужен разномный выбор чтобы разнообразить наше описание товара
            random_choice = random.random()

            random_name_parts = random.choices(spare_parts_name[row[0]])
            random_name_marka = random.choices(marka_mame[row[1]])

            # рандомно выбираем модель
            random_model = baza_name_marke_model.loc[
                baza_name_marke_model.model_normal == row[2]
            ].model.tolist()

            if len(random_model) == 0:
                print("В базе не хватает модели - ", row[2])
            random_name_model = random.choices(random_model)

            # разделение на 2 ветки в зависимости от вероятности
            if random_choice > 0.2:
                if random_choice > 0.6:
                    spisok = " ".join(
                        random_name_parts
                        + random_name_marka
                        + random_name_model
                        + [row[3]]
                    )
                else:
                    spisok = " ".join(
                        random_name_parts + random_name_marka + random_name_model
                    )
            else:
                if random_choice > 0.6:
                    spisok = " ".join(
                        random_name_marka
                        + random_name_model
                        + random_name_parts
                        + [row[3]]
                    )
                else:
                    spisok = " ".join(
                        random_name_marka + random_name_model + random_name_parts
                    )

            print([spisok] + row, [spisok], row)
            spisok_zaprosov.append([spisok] + row)
    return spisok_zaprosov


def prerare_exel(df, name=0, group=1, marka=3, model=2, number=3):
    """
    Подготовка датасета под стандарты - одна строка - один запрос
    Разбиаеться если много номеров или моделей
    """
    answer = []
    for i, row in df.iterrows():
        for a in row[model].split(","):
            try:
                for b in row[number].split(","):
                    if len(row) == 3:
                        answer.append(
                            [
                                row[name],
                                a.strip(),
                                b.strip(),
                            ]
                        )
                    elif len(row) == 5:
                        answer.append(
                            [
                                row[name],
                                row[group],
                                row[marka],
                                a.strip(),
                                b.strip(),
                            ]
                        )
                    elif len(row) > 5:
                        answer.append(
                            [row[0], row[1], a.strip(), b.strip(), row[4], row[5]]
                        )
                    else:
                        answer.append([row[0], row[1], a.strip(), b.strip(), row[4]])
            except:
                answer.append([row[name], row[1], a.strip()])

    return answer


def number_found(text):
    # регулярка для номеров ХЕндай Дусан Вольво

    reg_z = r"""
    [ \t\v\r\n\f] \d{2} [a-zA-Z] \d [- ‑]? \w{5}
    | \d{2}[nqNQN]\w{1}[- ‑]?\w{5}
    | \d{8}
    | [kKкК]\d{1}[vV]\w*
    | \d{2,6}-\d{3,5}
    | [Kk]?\d{7}
    | [Xx][a-zA-Z]{3}[-‑]?\d{5}
    | \d{3}-\d{2}-\d{3,5}
    | [ \t\v\r\n\f]\d{2}[a-zA-Z]{2}[- ‑]?\w{5}
    | [ \t\v\r\n\f][a-zA-ZА-Яа-я]{4}[-‑]?\d{5}[ \t\v\r\n\f]
    | [ \t\v\r\n\f]\d{3}[-‑]?\d{5}
    """

    values = [i.lower().strip() for i in re.findall(reg_z, text, flags=re.VERBOSE)]

    return values


def model_found(text):
    # регулярка для номеров ХЕндай Дусан Вольво
    reg_z = r"[ \t\v\r\n\f]\d{3}[ ,-]?[n,N]?[Л,l, L,л,l][c,C,с,С,c][ ,-]?[7,9]?|[ \t\v\r\n\f]\d{3}[ \$\f\n\r]|[ \t\v\r\n\f][r,R]\d{3}[ ,-]?[w,s,W, ,l, L,л]?[c,C,с,С]?[ ,-]?[5,6,7,9]?[\w][ \t\v\r\n\f]?|[E,E,e,е][c,с,С,C][, ]?\d{3}[\w]?|[b,B,б,Б][l,L,л,Л][, ]?\d{2}[\w,\w]?|[S,s,o,l,a,r]{5}[, ]?\d{3}[, ]?[,l, L,л]?[c,C,с,С]?[-]?\w?|[D,d,Д,д, ][x,X,х,Х,S,s]\d{3}[, ]?[,l, L,л]?[c,C,с,С]?[-]?\w?|[ \t\v\r\n\f]\d{3}[L,C,l,c]{2}[ ,-]?[V,v,A,a]|[r,R,р]\d{3}[ \t\v\r\n\f]|[ \t\v\r\n\f]\w{1}\d{3}"
    r = re.compile(reg_z)
    return [i.lower().strip() for i in re.findall(r, text)]


import re


def cifra(text):

    r = re.compile("\d{3,4}")
    russian = re.findall(r, text)
    return russian


# загружаем нашу базу из файла
baza = pd.read_csv(
    "data/baza.csv",
)

"""### Пример работы"""


baza.to_sql("baza", sqlite3.connect("/tmp/db"), index=False, if_exists="replace")
nnmodel = torch.load("data/model_emmbed")


import json

with open("data/class_names") as f:
    class_names = json.load(f)

ix2word = dict(enumerate(class_names))
with open("data/token_to_id") as f:
    token_to_id = json.load(f)
UNK, PAD = "UNK", "PAD"
UNK_IX, PAD_IX = map(token_to_id.get, [UNK, PAD])
import transformers
from transformers import (
    BertModel,
    BertTokenizer,
    AdamW,
    get_linear_schedule_with_warmup,
)

PRE_TRAINED_MODEL_NAME = "bert-base-cased"
tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)


class SentimentClassifier(nn.Module):
    def __init__(self, n_classes):
        super(SentimentClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        output = self.drop(outputs["pooler_output"])
        return self.out(output)


set_module_var("__main__", "SentimentClassifier", SentimentClassifier)
model_name = torch.load("data/model_name", map_location=torch.device("cpu"))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open("data/ix2word_name") as f:
    class_names3 = json.load(f)


# METHOD 1


def dict_factory(cursor, row):
    fields = [column[0] for column in cursor.description]
    return {key: value for key, value in zip(fields, row)}


def exit_sql(zapros, type_of="price", is_marka_model=False):
    """
    Функция ищет запрос в базе данных и возращает цену и дополнительные номера
    type_of может быть / model - получаем все модели
                      / number - получаем дополнительные номера
                      / price - цена
    """
    with sqlite3.connect("/tmp/db") as db:
        db.row_factory = dict_factory
        cursor = db.cursor()
        # возвра
        values = number_found(zapros.lower())
        if not values:
            return None, None
        value = values[0].strip().lower()
        if value is None:
            return None, None
        # формирование запроса

        if type_of == "model":
            sql = "select DISTINCT b.marka, b.model from baza as b \
            where oll_number == (SELECT oll_number from baza where parts_number = ?)"
        elif type_of == "number":
            sql = "select DISTINCT b.parts_number from baza as b \
            where oll_number == (SELECT oll_number from baza where parts_number = ?)"
        else:
            sql = "SELECT  DISTINCT name, marka, model, price, image_url, groups from baza where parts_number = ? "

        cursor.execute(sql, (value,))

        # ответ
        skins = cursor.fetchall()
        cursor.close()
        # вернем распознаный номер и строку из базы
        return value, skins


import sys


# METHOD 2


def exit_sql_marka_model(groups, marka, model):
    """
    >>> r = exit_sql_marka_model('насос/', 'Hyundai', 'r160lc-7')
    >>> len(r)
    1
    >>> r # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
    [{'name': 'Насос основной', 'marka': 'Hyundai', 'model': 'r160lc-7', 
      'price': 150000, ..., 'groups': 'насос/гидравлический/основной'}]

    >>> exit_sql_marka_model('насос/', 'Hyundai', '160')
    """
    # print(groups, marka, model_prefix, model_suffix)
    with sqlite3.connect("/tmp/db") as db:
        db.row_factory = dict_factory
        cursor = db.cursor()

        # вот сюда нужно втыкнуть чтобы он выводил все номера по этому общему номеру

        if "/" in model:
            model_pattern = model.replace('/', '%')
        else:
            model_pattern = f'%{model}%'

        sql_base = "SELECT DISTINCT name,marka,model,price, image_url,groups from baza where marka = ? and model like ?"
        args_base = (marka, model_pattern)

        cursor.execute(f"{sql_base} and groups like ?", args_base + (groups + "%",))
        skins = cursor.fetchall()
        if skins:
            return skins

        cursor.execute(sql_base, args_base)
        skins = cursor.fetchall()
        return skins


ix2word = dict(enumerate(class_names))

# review_text = "210lc гидронасос насос"
def return_marka_model(review_text, max_len=10):
    """
    review_text - запрос
    class_names - словарь название модели - таргет берта
    max_len - максимальная длиннна запроса
    """
    # TODO: return value
    np.random.seed(0)
    random.seed(0)

    stoplist = []
    sent = [word for word in review_text.split() if word not in stoplist]
    row_ix = [token_to_id.get(word, UNK_IX) for word in sent[:max_len]]
    matrix = np.full((1, max_len), np.int32(PAD_IX))
    matrix[0, : len(row_ix)] = row_ix
    predict = np.argmax(nnmodel.predict({"text": matrix}).detach().numpy(), axis=1)[0]
    return ix2word[predict]


from collections import Counter


def predict_marka_prefix_suffix(review_text, max_len=10):
    """
    >>> predict_marka_prefix_suffix('редуктор hyundai')  # doctest: +ELLIPSIS
    ('Hyundai', ...)

    >>> predict_marka_prefix_suffix('редуктор hyundai 210lc 7')  # doctest: +ELLIPSIS
    ('Hyundai', ...)
    """
    # pr = return_marka_model(review_text) # возвращаем что модель смогла получить
    a = []
    for i in range(100):
        a.append(return_marka_model(review_text))

    pr = Counter(a).most_common(1)[0][0]
    marka, model = pr.split("/", maxsplit=1)
    # а если модели не будет?
    return marka, model



"""
def predict_marka_model(review_text, max_len=10):
    marka,prefix, suffix = predict_marka_prefix_suffix(review_text, max_len=10)
    answer = exit_sql_marka_model( marka,prefix, suffix)
"""
# METHOD 3


def predict_group(tokenizer, model_name, review):
    """
    Функция для предсказания группы
    """
    encoding = tokenizer.encode_plus(
        review,
        add_special_tokens=True,
        max_length=100,
        return_token_type_ids=False,
        pad_to_max_length=True,
        truncation=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    outputs = model_name(
        input_ids=encoding["input_ids"].to(device),
        attention_mask=encoding["attention_mask"].to(device),
    )
    _, preds = torch.max(outputs, dim=1)
    print(outputs)
    # ix2word = class_names3
    return class_names3[str(preds[0].cpu().item())]


def predict_group_for_query(query):
    """
    >>> predict_group_for_query("Hyundai r160lc-7")
    >>> predict_group_for_query("насос Hyundai r160lc-7")
    """
    return predict_group(tokenizer, model_name, query)
