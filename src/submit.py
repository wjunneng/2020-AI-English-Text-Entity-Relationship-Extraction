# -*- coding:utf-8 -*-
import os
import sys
from pathlib import Path

# project_dir = str(Path(__file__).parent.parent.parent)
project_dir = '/'

sys.path.append(project_dir)
os.chdir(sys.path[-1])

import argparse
from src.core.data_loader import load_data
from src.core.model import E2EModel
from src.core.utils import get_tokenizer, metric_new
from src.conf.config import PATH
from src.lib.util import Util

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

parser = argparse.ArgumentParser(description='Model Controller')

args = parser.parse_args()

if __name__ == '__main__':
    util = Util()
    util.generate_test_json()

    # pre-trained bert model config
    bert_config_path = PATH.PRE_TRAIN_MODEL_DIR + '/bert_config.json'
    bert_vocab_path = PATH.PRE_TRAIN_MODEL_DIR + '/vocab.txt'
    bert_checkpoint_path = PATH.PRE_TRAIN_MODEL_DIR + '/bert_model.ckpt'

    train_path = PATH.OUTPUT_DIR + '/train.json'
    dev_path = PATH.OUTPUT_DIR + '/dev.json'
    # test_path = PATH.OUTPUT_DIR + '/test_split_by_num/test_triples_5.json' # ['1','2','3','4','5']
    # test_path = PATH.OUTPUT_DIR + '/test_split_by_type/test_triples_seo.json' # ['normal', 'seo', 'epo']
    test_path = PATH.OUTPUT_DIR + '/test.json'  # overall test
    rel_dict_path = PATH.OUTPUT_DIR + '/rel2id.json'
    save_weights_path = PATH.OUTPUT_DIR + '/best_model.weights'

    LR = 2e-5
    tokenizer = get_tokenizer(bert_vocab_path)
    train_data, dev_data, test_data, id2rel, rel2id, num_rels = load_data(train_path, dev_path, test_path,
                                                                          rel_dict_path)
    subject_model, object_model, hbt_model = E2EModel(bert_config_path, bert_checkpoint_path, LR, num_rels)

    hbt_model.load_weights(save_weights_path)
    test_result_path = PATH.OUTPUT_DIR + '/test_result.json'
    test_submit_path = PATH.submit_csv_path
    isExactMatch = False
    if isExactMatch:
        print("Exact Match")
    else:
        print("Partial Match")
    precision, recall, f1_score = metric_new(object_model, test_data, id2rel, tokenizer, isExactMatch,
                                             test_result_path, test_submit_path)
    print(f'{precision}\t{recall}\t{f1_score}')
