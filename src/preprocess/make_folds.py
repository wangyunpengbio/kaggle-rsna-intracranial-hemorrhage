import sys
import argparse
import collections
import pickle
from pprint import pprint
import random

import numpy as np
from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input')
    parser.add_argument('--output')
    parser.add_argument('--n-fold', type=int, default=5)
    parser.add_argument('--seed', type=int, default=10)
    return parser.parse_args()

# ！！！这波代码的含义就是，维持同一个病人的图像只出现在一个fold之中，并且保持每个fold的各种疾病数量大致相当
def _make_folds(df, n_fold, seed):
    # 先统计每一类有多少个，键是疾病种类，值是对应数量
    counter_gt = collections.defaultdict(int)
    for labels in df.labels.str.split():
        for label in labels:
            counter_gt[label] += 1

    counter_folds = collections.Counter()

    folds = {}
    random.seed(seed)
    groups = df.groupby('PatientID') # 按病人汇总
    print('making %d folds...' % n_fold)
    for patient_id, group in tqdm(groups, total=len(groups)):

        # labels记录了每个病人的label（注意：此处记录的不是一张图片的label，是一个病人的label）
        labels = []
        for row in group.itertuples():
            for label in row.labels.split():  #row.labels是如同： subarachnoid subdural any这样的形式
                labels.append(label)
        if not labels: # 没有任何疾病的人还是少数的，最后统计下来一共就1w人
            labels = ['']

        count_labels = [counter_gt[label] for label in labels] # count_labels，此处把每个病人的label转成全部数据中，对应疾病的总数量
        min_label = labels[np.argmin(count_labels)] # 找出总数中出现最少的疾病（优先维持较少类别疾病的平衡）
        count_folds = [(f, counter_folds[(f, min_label)]) for f in range(n_fold)] # 统计这种疾病在各个fold之中的数量
        min_count = min([count for f,count in count_folds])# 找出这种最少出现的疾病在最少的fold之中的数量
        fold = random.choice([f for f,count in count_folds if count == min_count])# 随机选其中最少的fold，赋值给那个fold（可能会有多个fold都最少）
        folds[patient_id] = fold

        for label in labels:
            counter_folds[(fold,label)] += 1

    pprint(counter_folds)

    return folds


def main():
    args = get_args()
    with open(args.input, 'rb') as f:
        df = pickle.load(f)

    folds = _make_folds(df, args.n_fold, args.seed)
    df['fold'] = df.PatientID.map(folds)
    with open(args.output, 'wb') as f:
        pickle.dump(df, f)

    print('saved to %s' % args.output)


if __name__ == '__main__':
    print(sys.argv)
    main()
