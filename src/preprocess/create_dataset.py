import sys
import argparse
import collections
import pickle
from pprint import pprint

import pandas as pd
from tqdm import tqdm

from ..utils import misc


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input')
    parser.add_argument('--output')
    return parser.parse_args()


def show_distribution(dataset):
    counter = collections.defaultdict(int)
    for row in dataset.itertuples():
        for label in row.labels.split():
            counter[label] += 1
        if not row.labels: # 如果有疾病，则negative+1
            counter['negative'] += 1
        counter['all'] += 1
    pprint(counter)


def parse_position(df):
    expanded = df.ImagePositionPatient.apply(lambda x: pd.Series(x))  # 图像位置：指示了图像左上角的第一个像素的空间坐标（x,y,z），也就是DICOM文件传输的第一个像素的坐标。
    expanded.columns = ['Position1', 'Position2', 'Position3']
    return pd.concat([df, expanded], axis=1)


def parse_orientation(df):
    expanded = df.ImageOrientationPatient.apply(lambda x: pd.Series(x))
    expanded.columns = ['Orient1', 'Orient2', 'Orient3', 'Orient4', 'Orient5', 'Orient6']
    return pd.concat([df, expanded], axis=1)


def add_adjacent_labels(df):
    df = df.sort_values('PositionOrd')

    records = []
    print('making adjacent labels...')
    for index,group in tqdm(df.groupby('StudyInstanceUID')):

        labels = list(group.labels)
        for j,id in enumerate(group.ID):
            if j == 0:
                left = ''
            else:
                left = labels[j-1]
            if j+1 == len(labels):
                right = ''
            else:
                right = labels[j+1]

            records.append({
                'LeftLabel': left,
                'RightLabel': right,
                'ID': id,
            })
    return pd.merge(df, pd.DataFrame(records), on='ID')


def main():
    args = get_args()
    
    # 之前存的是一个dataframe对象，此刻加载回来
    with open(args.input, 'rb') as f:
        df = pickle.load(f)
    print('read %s (%d records)' % (args.input, len(df)))

    show_distribution(df)

    df = df[df.custom_diff > 60]
    print('removed records by custom_diff (%d records)' % len(df))

    df = parse_position(df) # 处理图像真实位置，生成['Position1', 'Position2', 'Position3']，为后续生成真实位置做准备

    df['WindowCenter'] = df.WindowCenter.apply(lambda x: misc.get_dicom_value(x))
    df['WindowWidth'] = df.WindowWidth.apply(lambda x: misc.get_dicom_value(x))
    # 这行代码相当于计算出图像真实的位置
    df['PositionOrd'] = df.groupby('SeriesInstanceUID')[['Position3']].rank() / df.groupby('SeriesInstanceUID')[['Position3']].transform('count')

    df = add_adjacent_labels(df)
    df = df[['ID', 'labels', 'PatientID', 'WindowCenter', 'WindowWidth', 'RescaleIntercept', 'RescaleSlope', 'Position3', 'PositionOrd', 'LeftLabel', 'RightLabel']]

    df = df.sort_values('ID')
    # 存储最后使用的表
    with open(args.output, 'wb') as f:
        pickle.dump(df, f)

    show_distribution(df)

    print('created dataset (%d records)' % len(df))
    print('saved to %s' % args.output)


if __name__ == '__main__':
    print(sys.argv)
    main()
