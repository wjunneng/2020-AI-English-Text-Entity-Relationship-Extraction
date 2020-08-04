# -*- coding:utf-8 -*-
import os
import sys
from pathlib import Path

project_dir = str(Path(__file__).parent.parent.parent)

sys.path.append(project_dir)
os.chdir(sys.path[0])


class PATH(object):
    DATA_DIR = os.path.join(project_dir, 'data')
    INPUT_DIR = os.path.join(DATA_DIR, 'input')
    OUPUT_DIR = os.path.join(DATA_DIR, 'output')

    CHINESE_WWM_PYTORCH_DIR = os.path.join(DATA_DIR, 'model', 'uncased_L-12_H-768_A-12')

    train_csv_path = os.path.join(INPUT_DIR, 'train.csv')
    test_csv_path = os.path.join(INPUT_DIR, 'test.csv')

    train_json_path = os.path.join(OUPUT_DIR, 'train.json')
    dev_json_path = os.path.join(OUPUT_DIR, 'dev.json')
    test_json_path = os.path.join(OUPUT_DIR, 'test.json')

    rel2id_json_path = os.path.join(OUPUT_DIR, 'rel2id.json')

    submit_csv_path = os.path.join(INPUT_DIR, 'submit.csv')


for item in PATH.__dict__:
    if 'DIR' in item:
        if not os.path.exists(PATH().__getattribute__(item)):
            os.makedirs(PATH().__getattribute__(item))
