# -*- coding:utf-8 -*-
import os
import sys
from pathlib import Path

# project_dir = str(Path(__file__).parent.parent.parent)
project_dir = '/home/wjunneng/Ubuntu/2020-AI-English-Text-Entity-Relationship-Extraction'

sys.path.append(project_dir)
os.chdir(sys.path[-1])

import argparse
from src.core.data_loader import data_generator, load_data
from src.core.model import E2EModel, Evaluate
from src.core.utils import get_tokenizer, metric, metric_new
from src.conf.config import PATH

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

parser = argparse.ArgumentParser(description='Model Controller')
parser.add_argument('--train', default=True, type=bool, help='to train the HBT model, python run.py --train=True')

args = parser.parse_args()

if __name__ == '__main__':
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

    LR = 1e-5
    tokenizer = get_tokenizer(bert_vocab_path)
    train_data, dev_data, test_data, id2rel, rel2id, num_rels = load_data(train_path, dev_path, test_path,
                                                                          rel_dict_path)
    subject_model, object_model, hbt_model = E2EModel(bert_config_path, bert_checkpoint_path, LR, num_rels)

    if args.train:
        BATCH_SIZE = 8
        EPOCH = 100
        MAX_LEN = 96
        STEPS = len(train_data) // BATCH_SIZE
        data_manager = data_generator(train_data, tokenizer, rel2id, num_rels, MAX_LEN, BATCH_SIZE)
        evaluator = Evaluate(subject_model, object_model, tokenizer, id2rel, dev_data, save_weights_path)
        hbt_model.fit_generator(data_manager.__iter__(), steps_per_epoch=STEPS, epochs=EPOCH, callbacks=[evaluator])
    else:
        hbt_model.load_weights(save_weights_path)
        test_result_path = PATH.OUTPUT_DIR + '/test_result.json'
        test_submit_path = PATH.OUTPUT_DIR + '/submit.csv'
        isExactMatch = False
        if isExactMatch:
            print("Exact Match")
        else:
            print("Partial Match")
        precision, recall, f1_score = metric_new(subject_model, object_model, test_data, id2rel, tokenizer, isExactMatch,
                                             test_result_path, test_submit_path)
        print(f'{precision}\t{recall}\t{f1_score}')
