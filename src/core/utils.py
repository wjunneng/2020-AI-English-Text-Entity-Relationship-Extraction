#! -*- coding:utf-8 -*-
import keras.backend as K
from keras_bert import Tokenizer
import numpy as np
import codecs
from tqdm import tqdm
import json
import csv
import unicodedata

BERT_MAX_LEN = 512


class HBTokenizer(Tokenizer):
    def _tokenize(self, text):
        if not self._cased:
            text = unicodedata.normalize('NFD', text)
            text = ''.join([ch for ch in text if unicodedata.category(ch) != 'Mn'])
            text = text.lower()
        spaced = ''
        for ch in text:
            if ord(ch) == 0 or ord(ch) == 0xfffd or self._is_control(ch):
                continue
            else:
                spaced += ch
        tokens = []
        for word in spaced.strip().split():
            tokens += self._word_piece_tokenize(word)
            tokens.append('[unused1]')
        return tokens


def get_tokenizer(vocab_path):
    token_dict = {}
    with codecs.open(vocab_path, 'r', 'utf8') as reader:
        for line in reader:
            token = line.strip()
            token_dict[token] = len(token_dict)
    return HBTokenizer(token_dict, cased=True)


def seq_gather(x):
    seq, idxs = x
    idxs = K.cast(idxs, 'int32')
    batch_idxs = K.arange(0, K.shape(seq)[0])
    batch_idxs = K.expand_dims(batch_idxs, 1)
    idxs = K.concatenate([batch_idxs, idxs], 1)
    return K.tensorflow_backend.tf.gather_nd(seq, idxs)


def extract_items(subject_model, object_model, tokenizer, text_in, id2rel, h_bar=0.5, t_bar=0.5):
    tokens = tokenizer.tokenize(text_in)
    token_ids, segment_ids = tokenizer.encode(first=text_in)
    token_ids, segment_ids = np.array([token_ids]), np.array([segment_ids])
    if len(token_ids[0]) > BERT_MAX_LEN:
        token_ids = token_ids[:, :BERT_MAX_LEN]
        segment_ids = segment_ids[:, :BERT_MAX_LEN]
    sub_heads_logits, sub_tails_logits = subject_model.predict([token_ids, segment_ids])
    sub_heads, sub_tails = np.where(sub_heads_logits[0] > h_bar)[0], np.where(sub_tails_logits[0] > t_bar)[0]
    subjects = []
    for sub_head in sub_heads:
        sub_tail = sub_tails[sub_tails >= sub_head]
        if len(sub_tail) > 0:
            sub_tail = sub_tail[0]
            subject = tokens[sub_head: sub_tail]
            subjects.append((subject, sub_head, sub_tail))
    if subjects:  # [(['indicator'], 6, 7)]
        triple_list = []
        token_ids = np.repeat(token_ids, len(subjects), 0)
        segment_ids = np.repeat(segment_ids, len(subjects), 0)
        sub_heads, sub_tails = np.array([sub[1:] for sub in subjects]).T.reshape((2, -1, 1))
        obj_heads_logits, obj_tails_logits = object_model.predict([token_ids, segment_ids, sub_heads, sub_tails])
        for i, subject in enumerate(subjects):
            sub = subject[0]
            sub = ''.join([i.lstrip("##") for i in sub])
            sub = ' '.join(sub.split('[unused1]'))
            obj_heads, obj_tails = np.where(obj_heads_logits[i] > h_bar), np.where(obj_tails_logits[i] > t_bar)
            for obj_head, rel_head in zip(*obj_heads):
                for obj_tail, rel_tail in zip(*obj_tails):
                    if obj_head <= obj_tail and rel_head == rel_tail:
                        rel = id2rel[rel_head]
                        obj = tokens[obj_head: obj_tail]
                        obj = ''.join([i.lstrip("##") for i in obj])
                        obj = ' '.join(obj.split('[unused1]'))
                        triple_list.append((sub, rel, obj))
                        break
        triple_set = set()
        for s, r, o in triple_list:
            triple_set.add((s, r, o))
        return list(triple_set)
    else:
        return []


def get_start_end_index(tokens, target):
    """
    获取subject/object index
    :param tokens:
    :param target:
    :return:
    """
    token_index = 0
    start_index, end_index = -1, -1
    target_ = target.replace(' ', '[unused1]')

    while target_ != '':
        token_text = tokens[token_index].replace('##', '')
        if token_text == target_[:len(token_text)]:
            if start_index == -1:
                start_index = token_index
            else:
                end_index = token_index

            target_ = target_[len(token_text):]
        else:
            start_index, end_index = -1, -1
            target_ = target.replace(' ', '[unused1]')

        token_index += 1

    if end_index == -1:
        end_index = start_index + 1

    return start_index, end_index


def extract_items_new(subject, object, object_model, tokenizer, text_in, id2rel):
    tokens = tokenizer.tokenize(text_in)
    subject_start_index, subject_end_index = get_start_end_index(tokens=tokens, target=subject)
    object_start_index, object_end_index = get_start_end_index(tokens=tokens, target=object)

    subjects = [(subject, subject_start_index, subject_end_index)]
    token_ids, segment_ids = tokenizer.encode(first=text_in)
    token_ids, segment_ids = np.array([token_ids]), np.array([segment_ids])

    # # [(['indicator'], 6, 7)]
    # if subjects:
    #     triple_list = []
    #     token_ids = np.repeat(token_ids, len(subjects), 0)
    #     segment_ids = np.repeat(segment_ids, len(subjects), 0)
    #     sub_heads, sub_tails = np.array([sub[1:] for sub in subjects]).T.reshape((2, -1, 1))
    #     obj_heads_logits, obj_tails_logits = object_model.predict([token_ids, segment_ids, sub_heads, sub_tails])
    #
    #     heads_index_score_dict = dict(zip(list(np.argsort(-obj_heads_logits[0][object_start_index, :])),
    #                                       [i for i in range(len(obj_heads_logits[0][object_start_index, :]))]))
    #     tails_index_score_dict = dict(zip(list(np.argsort(-obj_tails_logits[0][object_end_index, :])),
    #                                       [i for i in range(len(obj_tails_logits[0][object_end_index, :]))]))
    #
    #     # 按照index进行排序
    #     heads_index_score_dict = dict(sorted(heads_index_score_dict.items(), key=lambda a: a[0]))
    #     # 按照index进行排序
    #     tails_index_score_dict = dict(sorted(tails_index_score_dict.items(), key=lambda a: a[0]))
    #
    #     # index对应的socre进行相加
    #     heads_tails_socre_array = np.sum(
    #         [np.asarray(list(heads_index_score_dict.values())), np.asarray(list(tails_index_score_dict.values()))],
    #         axis=0)
    #     heads_tails_index_socre_dict = dict(zip(list(heads_index_score_dict.keys()), heads_tails_socre_array))
    #     heads_tails_index_socre_dict = sorted(heads_tails_index_socre_dict.items(), key=lambda a: a[1])
    #
    #     rel_index = heads_tails_index_socre_dict[0][0]
    #     rel = id2rel[rel_index]
    #     triple_list.append((subject, rel, object))
    #
    #     triple_set = set()
    #     for s, r, o in triple_list:
    #         triple_set.add((s, r, o))
    #     return list(triple_set)
    # else:
    #     return []

    # [(['indicator'], 6, 7)]
    # 效果较好
    if subjects:
        triple_list = []
        token_ids = np.repeat(token_ids, len(subjects), 0)
        segment_ids = np.repeat(segment_ids, len(subjects), 0)
        sub_heads, sub_tails = np.array([sub[1:] for sub in subjects]).T.reshape((2, -1, 1))
        obj_heads_logits, obj_tails_logits = object_model.predict([token_ids, segment_ids, sub_heads, sub_tails])

        heads_index_score_list = list(np.argsort(-obj_heads_logits[0][object_start_index, :]))

        rel_index = heads_index_score_list[0]
        rel = id2rel[rel_index]
        triple_list.append((subject, rel, object))

        triple_set = set()
        for s, r, o in triple_list:
            triple_set.add((s, r, o))
        return list(triple_set)
    else:
        return []


def partial_match(pred_set, gold_set):
    pred = {(i[0].split(' ')[0] if len(i[0].split(' ')) > 0 else i[0], i[1],
             i[2].split(' ')[0] if len(i[2].split(' ')) > 0 else i[2]) for i in pred_set}
    gold = {(i[0].split(' ')[0] if len(i[0].split(' ')) > 0 else i[0], i[1],
             i[2].split(' ')[0] if len(i[2].split(' ')) > 0 else i[2]) for i in gold_set}
    return pred, gold


def metric(subject_model, object_model, eval_data, id2rel, tokenizer, exact_match=False, output_path=None):
    if output_path:
        F = open(output_path, 'w')
    orders = ['subject', 'relation', 'object']
    correct_num, predict_num, gold_num = 1e-10, 1e-10, 1e-10
    for line in tqdm(iter(eval_data)):
        Pred_triples = set(extract_items(subject_model, object_model, tokenizer, line['text'], id2rel))
        Gold_triples = set(line['triple_list'])

        Pred_triples_eval, Gold_triples_eval = partial_match(Pred_triples, Gold_triples) if not exact_match else (
            Pred_triples, Gold_triples)

        correct_num += len(Pred_triples_eval & Gold_triples_eval)
        predict_num += len(Pred_triples_eval)
        gold_num += len(Gold_triples_eval)

        if output_path:
            result = json.dumps({
                'text': line['text'],
                'triple_list_gold': [
                    dict(zip(orders, triple)) for triple in Gold_triples
                ],
                'triple_list_pred': [
                    dict(zip(orders, triple)) for triple in Pred_triples
                ],
                'new': [
                    dict(zip(orders, triple)) for triple in Pred_triples - Gold_triples
                ],
                'lack': [
                    dict(zip(orders, triple)) for triple in Gold_triples - Pred_triples
                ]
            }, ensure_ascii=False, indent=4)
            F.write(result + '\n')
    if output_path:
        F.close()

    precision = correct_num / predict_num
    recall = correct_num / gold_num
    f1_score = 2 * precision * recall / (precision + recall)

    print(f'correct_num:{correct_num}\npredict_num:{predict_num}\ngold_num:{gold_num}')
    return precision, recall, f1_score


def metric_new(object_model, eval_data, id2rel, tokenizer, exact_match=False, output_path=None, submit_path=None):
    if output_path:
        F = open(output_path, 'w')
        S = open(submit_path, 'w', newline='')
        S_writer = csv.writer(S)

    orders = ['subject', 'relation', 'object']
    correct_num, predict_num, gold_num = 1e-10, 1e-10, 1e-10
    # {'text': 'The red indicator light on my telephone continues to blink after I have checked and emptied my mailbox.'
    # , 'triple_list': [('light', 'None', 'telephone')]}
    for line_index, line in tqdm(enumerate(iter(eval_data))):
        subject = line['triple_list'][0][0]
        object = line['triple_list'][0][2]
        Pred_triples = set(extract_items_new(subject=subject, object=object, object_model=object_model,
                                             tokenizer=tokenizer, text_in=line['text'], id2rel=id2rel))
        Gold_triples = set(line['triple_list'])

        Pred_triples_eval, Gold_triples_eval = partial_match(Pred_triples, Gold_triples) if not exact_match else (
            Pred_triples, Gold_triples)

        correct_num += len(Pred_triples_eval & Gold_triples_eval)
        predict_num += len(Pred_triples_eval)
        gold_num += len(Gold_triples_eval)

        if output_path:
            result = json.dumps({
                'text': line['text'],
                'triple_list_gold': [dict(zip(orders, triple)) for triple in Gold_triples],
                'triple_list_pred': [dict(zip(orders, triple)) for triple in Pred_triples],
                'new': [dict(zip(orders, triple)) for triple in Pred_triples - Gold_triples],
                'lack': [dict(zip(orders, triple)) for triple in Gold_triples - Pred_triples]}, ensure_ascii=False,
                indent=4)
            F.write(result + '\n')

            if list(Pred_triples)[0][0] != list(Gold_triples)[0][0] or list(Pred_triples)[0][2] != \
                    list(Gold_triples)[0][2]:
                print(Pred_triples, Gold_triples)

            relation = list(Pred_triples)[0][1]
            S_writer.writerow([line_index, relation])

    if output_path:
        F.close()
        S.close()

    precision = correct_num / predict_num
    recall = correct_num / gold_num
    f1_score = 2 * precision * recall / (precision + recall)

    print(f'correct_num:{correct_num}\npredict_num:{predict_num}\ngold_num:{gold_num}')
    return precision, recall, f1_score
