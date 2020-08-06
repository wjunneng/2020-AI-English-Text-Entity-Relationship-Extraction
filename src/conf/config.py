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
    OUTPUT_DIR = os.path.join(DATA_DIR, 'output')

    PRE_TRAIN_MODEL_DIR = os.path.join(DATA_DIR, 'model', 'uncased_L-12_H-768_A-12')

    train_csv_path = os.path.join(INPUT_DIR, 'train.csv')
    test_csv_path = os.path.join(INPUT_DIR, 'test.csv')
    test_file_full_txt_path = os.path.join(INPUT_DIR, 'TEST_FILE_FULL.TXT')
    train_file_txt_path = os.path.join(INPUT_DIR, 'TRAIN_FILE.TXT')

    train_json_path = os.path.join(OUTPUT_DIR, 'train.json')
    dev_json_path = os.path.join(OUTPUT_DIR, 'dev.json')
    test_json_path = os.path.join(OUTPUT_DIR, 'test.json')

    rel2id_json_path = os.path.join(OUTPUT_DIR, 'rel2id.json')

    submit_csv_path = os.path.join(INPUT_DIR, 'submit.csv')
    recall_submit_csv_path = os.path.join(INPUT_DIR, 'recall_submit.csv')


for item in PATH.__dict__:
    if 'DIR' in item:
        if not os.path.exists(PATH().__getattribute__(item)):
            os.makedirs(PATH().__getattribute__(item))
