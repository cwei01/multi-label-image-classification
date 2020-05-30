import pandas as pd
from pandas.core.frame import DataFrame
import numpy as np

from PIL import Image

from image import build_model

if __name__ == '__main__':
    # y_pred=[[0.884931  , 0.0296324, 0.00693199, 0.00261877 ,0.01287931 ,0.00387509,
    #  0.0078038 , 0.00759728 ,0.00642071, 0.00475496, 0.0032026 , 0.00182845,
    #  0.00134397, 0.00451495, 0.00634398 ,0.01038251, 0.04453781, 0.01372127,
    #  0.0057035],[0.884931, 0.00296324, 0.50693199, 0.00261877, 0.01287931, 0.00387509,
    #      0.0078038, 0.00759728, 0.00642071, 0.00475496, 0.0032026, 0.00182845,
    #      0.00134397, 0.00451495, 0.00634398, 0.01038251, 0.04453781, 0.01372127,
    #      0.6057035]]
    # pre_list = []
    # for i in range(len(y_pred)):
    #     pre_temp=[]
    #     for j in range(len(y_pred[0])):
    #         if y_pred[i][j]>=0.5:
    #             pre_temp.append(str(j+1))
    #     pre_list.append(pre_temp)
    # #pre_list=DataFrame(pre_list)
    # #test = pd.read_csv('train.csv').head(2)
    # test= pd.read_csv(r'../test.csv', encoding='utf-8').head(2)
    # #test.drop(labels=None, axis=1, index=None, columns='Caption', inplace=True)
    #
    # test['Labels']=pre_list
    #
    # for i in range(len(test['Labels'])):
    #     if 'A' in  str(test['Caption'][i]):
    #         test['Labels'][i].append('1')
    #         test['Labels'][i]=set(test['Labels'][i])
    # for i in range(len(test)):
    #     test['Labels'][i] = " ".join(test['Labels'][i])
    # test.drop(labels=None, axis=1, index=None, columns='Caption', inplace=True)
    # print(test)

    # out_path='../submit.csv'
    # test.to_csv(out_path, sep=',', index=False, header=True)
    model=build_model(19)
    model.load_weights('model4.h5')
    #model=model.load_model('../model.h5')
    X_test = np.zeros((10000, 224, 224, 3), dtype=np.uint8)
    i = 0
    for img_path in range(30000, 40000):
        str2 = 'data' + r"/{}.jpg".format(img_path)
        img = Image.open(str2)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img = img.resize((224, 224))
        arr = np.asarray(img)
        X_test[i, :, :, :] = arr
        i += 1

    y_pred = model.predict(X_test)
    #y_pred=model.
    # print(y_pred[0])
    pre_list = []
    for i in range(len(y_pred)):
        pre_temp = []
        for j in range(len(y_pred[0])):
            if y_pred[i][j] >= 0.5:
                t = str(j + 1)
                pre_temp.append(t)
        pre_list.append(pre_temp)
    # pre_list=DataFrame(pre_list)

    test = pd.read_csv(r'test.csv', encoding='utf-8')
    #test.drop(labels=None, axis=1, index=None, columns='Caption', inplace=True)
    test['Labels'] = pre_list
    # print(test)
    for i in range(len(test['Labels'])):
        if 'cat' in str(test['Caption'][i]):
            test['Labels'][i].append('17')
        if 'dog' in str(test['Caption'][i]):
            test['Labels'][i].append('18')
        if 'horse' in str(test['Caption'][i]):
            test['Labels'][i].append('19')
        if 'bench' in str(test['Caption'][i]):
            test['Labels'][i].append('15')
        if 'bird' in str(test['Caption'][i]):
            test['Labels'][i].append('16')
        if 'bicycle' in str(test['Caption'][i]):
             test['Labels'][i].append('2')
        if 'motorcycle' in str(test['Caption'][i]):
             test['Labels'][i].append('4')
        if 'plane' in str(test['Caption'][i]):
             test['Labels'][i].append('5')
        if 'bus' in str(test['Caption'][i]):
                 test['Labels'][i].append('6')
        if 'train' in str(test['Caption'][i]):
             test['Labels'][i].append('7')
        if 'truck' in str(test['Caption'][i]):
             test['Labels'][i].append('8')
        if 'boat' in str(test['Caption'][i]):
             test['Labels'][i].append('9')
        if 'fire' in str(test['Caption'][i]):
            test['Labels'][i].append('11')
        test['Labels'][i] = set(test['Labels'][i])
    for i in range(len(test)):
        if test['Labels'][i] is None:
            test['Labels'][i].append('1')
        test['Labels'][i] = " ".join(test['Labels'][i])
    test.drop(labels=None, axis=1, index=None, columns='Caption', inplace=True)
    # print(test)
    out_path = 'submit.csv'
    test.to_csv(out_path, sep=',', index=False, header=True)