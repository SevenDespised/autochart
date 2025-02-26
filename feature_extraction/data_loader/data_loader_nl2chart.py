import json


def load_data_nl2chart(data_file_name = '../data/preserved_data/raw_data_all.csv'):
    print('Loading nl2chart data from %s' % data_file_name)
    with open(data_file_name, "r", encoding="utf-8") as file:
        data = json.load(file)
    return data

def data_format_trans(data):
    res = {}
    order = 0
    type = data['chart']
    for x in data['x_data']:
        res['x:' + str(order)] = {}
        res['x:' + str(order)]['uid'] = 'x#' + str(order)
        res['x:' + str(order)]['order'] = order
        res['x:' + str(order)]['data'] = [a for a in x]
        order += 1
    order = 0
    for y in data['y_data']:
        res['y:' + str(order)] = {}
        res['y:' + str(order)]['uid'] = 'y#' + str(order)
        res['y:' + str(order)]['order'] = order
        res['y:' + str(order)]['data'] = [a for a in y]
        order += 1
    return res
    