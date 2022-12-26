import base64
import binascii
import datetime
import flask
import jinja2
import json
import os
import pathlib
import uuid
import zipfile

import pytz

import microbook

JINJA_ENVIRONMENT = jinja2.Environment(loader=jinja2.FileSystemLoader("templates/"))
INDEX_TEMPLATE = JINJA_ENVIRONMENT.get_template("index.html")

APP = flask.Flask(__name__)


def make_method_rows(query):
    """
    >>> make_method_rows("400914-00212")  # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
    {'groups': 'насос/',
     'method': 'sql',
     'values': '400914-00212',
     'rows': [('Насос основной', 'Doosan', 'dx225', 220000, ...)]}

    >>> make_method_rows("r160lc-7")  # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
    {'groups': 'насос/',
     'method': 'predict',
     'rows': [...]}

    #>>> make_method_rows("r160lc-7 основной насос")  # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS

    >>> make_method_rows("")  # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
    {'groups': 'редуктор/запчасти', 'method': 'predict', 'rows': [(...
    """
    groups = microbook.predict_group_for_query(query)
    # groups = 'редуктор/хода/'
    method = "sql"
    values, rows = microbook.exit_sql(
        query
    )  # ищем каталожный номер и выводим ответ из базы
    if rows:
        return {"groups": groups, "method": method, "values": values, "rows": rows}
    # если не нашел по номеру
    method = "predict"
    marka, model = microbook.predict_marka_prefix_suffix(query)
    # вот тут переделать - чтобы получали предикт не только по марке модели но и по номеру
    rows = microbook.exit_sql_marka_model(groups, marka, model)
    return {
        "groups": groups,
        "method": method,
        "marka": marka,
        "model": model,
        "rows": rows,
    }


def make_answer(query):
    try:
        method_rows = make_method_rows(query)
    except Exception as exc:
        return {"error": str(exc)}

    return method_rows
    """
        return method_rows

    method = method_rows['method']
    rows = method_rows['rows']
    #groups = method_rows['groups']
    values = method_rows.get('values')
    '''
    name = 'e', marka='e', model='e', price ='e', image_url = 'https://cdni.comss.net/logo/google-chrome-2000.png', group = 't'
    '''
    return {"values": values, "rows": rows}
    #name, marka, model, price, image_url, group = rows[0]
    # TODO: all rows
    #print(rows)
    return {
        "values":values, # распознаный запрос
        "name": name, #группа из базы
        "marka": marka, 
        "model": model,
        "price": price,
        "image_url": image_url,
    }
    """


def query_answer_to_log_row(precolumns, answer):
    """
    >>> query_answer_to_log_row(['2022-12-16', '127.0.0.1', 'abc'], {'error': 'oh'})
    '"2022-12-16","127.0.0.1","abc","oh"'

    >>> query_answer_to_log_row(['abc'], {'name': 'imya', 'price': 123})
    '"abc","","imya","123"'
    """
    row = [*precolumns, ""]
    if "error" in answer:
        row[-1] = answer["error"]
    else:
        row += answer.values()
    return ",".join('"' + str(x).replace('"', '""') + '"' for x in row)


def log_answer(query, answer):
    row_str = query_answer_to_log_row(
        [(datetime.datetime.now() + datetime.timedelta(hours=3)).isoformat(), flask.request.remote_addr, query], answer
    )
    with open("app-log.csv", "a") as file:
        file.write(row_str + "\n")


@APP.route("/", methods=["GET", "POST"])
def index():
    query = None
    answer = {}
    if flask.request.method == "POST":
        query = flask.request.form["query"]
        answer = make_answer(query)
        log_answer(query, answer)

        # return str(answer)
    return INDEX_TEMPLATE.render(query=query, **answer)
