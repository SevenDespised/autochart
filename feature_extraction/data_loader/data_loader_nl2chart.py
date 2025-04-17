import json


def load_data_nl2chart(data_file_name = '../data/preserved_data/raw_data_all.csv'):
    print('Loading nl2chart data from %s' % data_file_name)
    with open(data_file_name, "r", encoding="utf-8") as file:
        data = json.load(file)
    return data

def data_format_trans(data, fid):
    res = {}
    order = 0
    type = data['chart']
    for x in data['x_data']:
        res[fid + str(order)] = {}
        res[fid + str(order)]['uid'] = fid + str(order)
        res[fid + str(order)]['order'] = order
        res[fid + str(order)]['data'] = [a for a in x]
        res[fid + str(order)]['is_x_or_y'] = 'x'
        order += 1
    for y in data['y_data']:
        res[fid + str(order)] = {}
        res[fid + str(order)]['uid'] = fid + str(order)
        res[fid + str(order)]['order'] = order
        res[fid + str(order)]['data'] = [a for a in y]
        res[fid + str(order)]['is_x_or_y'] = 'y'
        order += 1
    return res
    