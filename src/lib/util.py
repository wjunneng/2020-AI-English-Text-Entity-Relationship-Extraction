# -*- coding:utf-8 -*-
import os
import sys
from pathlib import Path

project_dir = str(Path(__file__).parent.parent.parent)

sys.path.append(project_dir)
os.chdir(sys.path[0])

import pandas as pd
from tqdm import tqdm
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

    def generate_train_dev_test_json(self):
        """
        生成json文件
        :return:
        """
        # 'text', 'label'
        train_df = pd.read_csv(self.train_csv_path, encoding='utf-8')

        result = []
        for train_index in range(train_df.shape[0]):
            text = train_df.iloc[train_index, 0]
            label = train_df.iloc[train_index, 1]

            sample = {}
            subject = re.split('<e1>|</e1>', text)[1]
            object = re.split('<e2>|</e2>', text)[1]

            text = text.strip('"')
            text = text.replace('<e1>' + subject + '</e1>', subject)
            text = text.replace('<e2>' + object + '</e2>', object)

            sample['text'] = text
            sample['triple_list'] = [[subject, label, object]]

            result.append(sample)

        random.shuffle(result)

        result_length = len(result)
        train_result = result[:int(result_length * 0.8)]
        dev_result = result[int(result_length * 0.8):]

        with open(self.train_json_path, encoding='utf-8', mode='w') as file:
            json.dump(obj=train_result, ensure_ascii=False, fp=file)

        with open(self.dev_json_path, encoding='utf-8', mode='w') as file:
            json.dump(obj=dev_result, ensure_ascii=False, fp=file)

        with open(self.test_json_path, encoding='utf-8', mode='w') as file:
            json.dump(obj=dev_result, ensure_ascii=False, fp=file)


if __name__ == '__main__':
    util = Util()

    # util.generate_rel2id_json()

    util.generate_train_dev_test_json()
