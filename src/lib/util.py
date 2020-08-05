# -*- coding:utf-8 -*-
import os
import sys
from pathlib import Path

project_dir = str(Path(__file__).parent.parent.parent)

sys.path.append(project_dir)
os.chdir(sys.path[0])

import pandas as pd
from collections import defaultdict
import json
import random
import numpy as np
import re

from src.conf.config import PATH

random.seed(42)
np.random.seed(42)


class Util(object):
    def __init__(self):
        self.train_csv_path = PATH.train_csv_path
        self.test_csv_path = PATH.test_csv_path

        self.train_json_path = PATH.train_json_path
        self.dev_json_path = PATH.dev_json_path
        self.test_json_path = PATH.test_json_path

        self.rel2id_json_path = PATH.rel2id_json_path

    def generate_rel2id_json(self):
        """
        生成rel2id json 文件
        :return:
        """
        rel_list = [
            # 因果关系
            'Cause-Effect(e1,e2)', 'Cause-Effect(e2,e1)',
            # 工具-使用者关系
            'Instrument-Agency(e1,e2)', 'Instrument-Agency(e2,e1)',
            # 产品-生产者关系
            'Product-Producer(e1,e2)', 'Product-Producer(e2,e1)',
            # 内容-容器关系
            'Content-Container(e1,e2)', 'Content-Container(e2,e1)',
            # 实体起源于某一个位置或者来源
            'Entity-Origin(e1,e2)', 'Entity-Origin(e2,e1)',
            # 实体往目的地移动
            # Entity-Destination(e2, e1) 在训练集中不存在
            'Entity-Destination(e1,e2)', 'Entity-Destination(e2,e1)',
            # 部分-整体组成关系
            'Component-Whole(e1,e2)', 'Component-Whole(e2,e1)',
            # 成员构成集合的非功能部分
            'Member-Collection(e1,e2)', 'Member-Collection(e2,e1)',
            # 信息与主题
            'Message-Topic(e1,e2)', 'Message-Topic(e2,e1)',
            # 其他
            'Other']

        id2rel = dict(zip([i for i in range(len(rel_list))], rel_list))
        rel2id = {value: key for (key, value) in id2rel.items()}

        with open(self.rel2id_json_path, encoding='utf-8', mode='w') as file:
            json.dump(obj=[id2rel, rel2id], fp=file)

    @staticmethod
    def get_result(df):
        result = []
        for index in range(df.shape[0]):
            text = df.iloc[index, 0]
            label = df.iloc[index, 1]

            sample = {}
            subject = re.split('<e1>|</e1>', text)[1]
            object = re.split('<e2>|</e2>', text)[1]

            text = text.strip('"')
            text = text.replace('<e1>' + subject + '</e1>', subject)
            text = text.replace('<e2>' + object + '</e2>', object)

            sample['text'] = text
            sample['triple_list'] = [[subject, label, object]]

            result.append(sample)

        return result

    def generate_train_dev_test_json(self):
        """
        生成json文件
        :return:
        """
        # 'text', 'label'
        train_df = pd.read_csv(self.train_csv_path, encoding='utf-8')
        # train_df.shape[0]: 8573
        print('train_df.shape[0]: {}'.format(train_df.shape[0]))

        """
            Cause-Effect(e1,e2) 392
            Cause-Effect(e2,e1) 691
            Component-Whole(e1,e2) 492
            Component-Whole(e2,e1) 501
            Content-Container(e1,e2) 416
            Content-Container(e2,e1) 168
            Entity-Destination(e1,e2) 906
            Entity-Origin(e1,e2) 632
            Entity-Origin(e2,e1) 146
            Instrument-Agency(e1,e2) 100
            Instrument-Agency(e2,e1) 448
            Member-Collection(e1,e2) 88
            Member-Collection(e2,e1) 641
            Message-Topic(e1,e2) 564
            Message-Topic(e2,e1) 161
            Other 1478
            Product-Producer(e1,e2) 340
            Product-Producer(e2,e1) 409
        """
        train, dev = None, None
        for label, group_df in train_df.groupby(by=['label']):
            group_df.sample(frac=1).reset_index(drop=True)

            threshold = int(group_df.shape[0] * 0.85)
            if train is None:
                train = group_df[:threshold]
                dev = group_df[threshold:]
            else:
                train = pd.concat([train, group_df[:threshold]], axis=0, ignore_index=True)
                dev = pd.concat([dev, group_df[threshold:]], axis=0, ignore_index=True)

        train.reset_index(inplace=True, drop=True)
        dev.reset_index(inplace=True, drop=True)

        # train.shape[0]: 7279, dev.shape[0]: 1294
        print('train.shape[0]: {}, dev.shape[0]: {}'.format(train.shape[0], dev.shape[0]))

        train_result = Util.get_result(df=train)
        dev_result = Util.get_result(df=dev)

        random.shuffle(train_result)
        random.shuffle(dev_result)

        with open(self.train_json_path, encoding='utf-8', mode='w') as file:
            json.dump(obj=train_result, ensure_ascii=False, fp=file)

        with open(self.dev_json_path, encoding='utf-8', mode='w') as file:
            json.dump(obj=dev_result, ensure_ascii=False, fp=file)

        with open(self.test_json_path, encoding='utf-8', mode='w') as file:
            json.dump(obj=dev_result, ensure_ascii=False, fp=file)

        # [(85, 1), (82, 1), (67, 1), (65, 1), (64, 2), (63, 1), (61, 1), (60, 1), (58, 3), (57, 4), (55, 1), (54, 2),
        # (53, 2), (52, 4), (51, 2), (50, 5), (49, 2), (48, 2), (47, 3), (46, 2), (45, 6), (44, 9), (43, 11), (42, 15),
        # (41, 9), (40, 21), (39, 14), (38, 25), (37, 24), (36, 28), (35, 32), (34, 33), (33, 52), (32, 60), (31, 49),
        # (30, 61), (29, 89), (28, 123), (27, 137), (26, 169), (25, 227), (24, 233), (23, 280), (22, 329), (21, 356),
        # (20, 395), (19, 435), (18, 431), (17, 446), (16, 475), (15, 456), (14, 456), (13, 483), (12, 494), (11, 476),
        # (10, 427), (9, 419), (8, 372), (7, 236), (6, 105), (5, 26), (4, 7), (3, 1)]

    def generate_test_json(self):
        """
        生成测试集结果
        :return:
        """
        test_df = pd.read_csv(self.test_csv_path, encoding='utf-8')
        text_length_count = defaultdict(lambda: 0)

        result = []
        for train_index in range(test_df.shape[0]):
            text = test_df.iloc[train_index, 1]

            sample = {}
            subject = re.split('<e1>|</e1>', text)[1]
            object = re.split('<e2>|</e2>', text)[1]

            text = text.strip('"')
            text = text.replace('<e1>' + subject + '</e1>', subject)
            text = text.replace('<e2>' + object + '</e2>', object)

            sample['text'] = text
            sample['triple_list'] = [[subject, 'None', object]]

            result.append(sample)

            text_length_count[len(text.split(' '))] += 1

        with open(self.test_json_path, encoding='utf-8', mode='w') as file:
            json.dump(obj=result, ensure_ascii=False, fp=file)

        # [(79, 1), (66, 1), (58, 1), (57, 1), (54, 1), (50, 1), (49, 1), (48, 1), (47, 3), (46, 1), (45, 1), (44, 3),
        # (43, 2), (42, 2), (41, 4), (40, 5), (39, 4), (38, 5), (37, 4), (36, 8), (35, 7), (34, 12), (33, 11), (32, 7),
        # (31, 18), (30, 17), (29, 20), (28, 37), (27, 30), (26, 44), (25, 44), (24, 63), (23, 85), (22, 70), (21, 85),
        # (20, 110), (19, 103), (18, 105), (17, 106), (16, 119), (15, 121), (14, 110), (13, 124), (12, 125), (11, 122),
        # (10, 107), (9, 100), (8, 92), (7, 60), (6, 31), (5, 8), (4, 1)]
        text_length_count = sorted(text_length_count.items(), key=lambda a: a[0], reverse=True)
        print('text_length_count: {}'.format(text_length_count))

# if __name__ == '__main__':
#     util = Util()
#
#     util.generate_rel2id_json()
#
#     util.generate_train_dev_test_json()
#
#     util.generate_test_json()
