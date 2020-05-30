import re
import pandas as pd
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import *
from keras.models import *
from keras.optimizers import *
from keras.callbacks import *
import keras.backend as K
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from pandas.core.frame import DataFrame
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def recall(y_true, y_pred):
    # Calculates the recall
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def fbeta_score(y_true, y_pred, beta=1):
    # Calculates the F score, the weighted harmonic mean of precision and recall.
    if beta<=0:
     raise ValueError('The lowest choosable beta is zero (only precision).')

    # If there are no true positives, fix the F score at 0 like sklearn.
    if K.sum(K.round(K.clip(y_true, 0, 1))) == 0:
        return 0

    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    bb = beta ** 2
    fbeta_score = (1 + bb) * (p * r) / (bb * p + r + K.epsilon())
    return fbeta_score


def fmeasure(y_true, y_pred):
    # Calculates the f-measure, the harmonic mean of precision and recall.
    return fbeta_score(y_true, y_pred, beta=1)


def bn_prelu(x):
    x = BatchNormalization()(x)
    x = PReLU()(x)
    return x


def build_model(out_dims, input_shape=(224, 224, 3)):
    inputs_dim = Input(input_shape)

    x = Conv2D(16, (3, 3), strides=(2, 2), padding='same')(inputs_dim)
    x = bn_prelu(x)
    x = Conv2D(16, (3, 3), strides=(1, 1), padding='same')(x)
    x = bn_prelu(x)
    x = MaxPool2D(pool_size=(2, 2))(x)

    x = Conv2D(32, (3, 3), strides=(1, 1), padding='same')(x)
    x = bn_prelu(x)
    x = Conv2D(32, (3, 3), strides=(1, 1), padding='same')(x)
    x = bn_prelu(x)
    x = MaxPool2D(pool_size=(2, 2))(x)

    x = Conv2D(64, (3, 3), strides=(1, 1), padding='same')(x)
    x = bn_prelu(x)
    x = MaxPool2D(pool_size=(2, 2))(x)

    x = Conv2D(128, (3, 3), strides=(1, 1), padding='same')(x)
    x = bn_prelu(x)
    x = GlobalAveragePooling2D()(x)

    dp_1 = Dropout(0.5)(x)

    fc2 = Dense(out_dims)(dp_1)
    fc2 = Activation('sigmoid')(fc2)  # sigmoid

    model = Model(inputs=inputs_dim, outputs=fc2)
    return model


if __name__ == '__main__':

    train_label=pd.read_csv(r'train.csv',encoding='utf-8')
    #print(train_label['Labels'])
    #print(train_label)
    label=[]
    for i in range(len(train_label['Labels'])):
        num = re.findall('\d+', train_label['Labels'][i])
        label_list=[0 for i in range(19)]
        for j in range(len(num)):
               label_list[int(num[j])-1]=1
        label.append(label_list)
    label=DataFrame(label)
    print(label.shape)
    nub_train = 30000  # train number
    X_train = np.zeros((nub_train, 224, 224, 3), dtype=np.uint8)
    i = 0

    for img_path in range(30000):
        str1='data'+r"/{}.jpg".format(img_path)
        img = Image.open(str1)
        img = img.resize((224, 224))
        arr = np.asarray(img)
        X_train[i, :, :, :] = arr
        i += 1


    X_train2, X_val, y_train2, y_val = train_test_split(X_train, label, test_size=0.3, random_state=2020)

    model = build_model(19)


    train_datagen = ImageDataGenerator(width_shift_range=0.1,
                                       height_shift_range=0.1,
                                       zoom_range=0.1)
    val_datagen = ImageDataGenerator()

    batch_size = 8

    train_generator = train_datagen.flow(X_train2, y_train2, batch_size=batch_size, shuffle=False)
    val_generator = val_datagen.flow(X_val, y_val, batch_size=batch_size, shuffle=False)


    checkpointer = ModelCheckpoint(filepath='model4.h5',
                                   monitor='val_fmeasure', verbose=1, save_best_only=True, mode='max')
    reduce = ReduceLROnPlateau(monitor='val_fmeasure', factor=0.5, patience=2, verbose=1, min_delta=1e-4)

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy', fmeasure,recall, precision])

    epochs = 5

    history = model.fit_generator(train_generator,
                                  validation_data=val_generator,
                                  epochs=epochs,
                                  callbacks=[checkpointer, reduce],
                                  verbose=1)

    #------------------------------------------test--------------
    # X_test = np.zeros((10000, 224, 224, 3), dtype=np.uint8)
    # i = 0
    # for img_path in range(30000, 40000):
    #     str2 = 'data' + r"/{}.jpg".format(img_path)
    #     img = Image.open(str2)
    #     if img.mode != 'RGB':
    #         img = img.convert('RGB')
    #     img = img.resize((224, 224))
    #     arr = np.asarray(img)
    #     X_test[i, :, :, :] = arr
    #     i += 1
    #
    # y_pred = model.predict(X_test)
    # y_pred=list(y_pred)
    # #print(y_pred[0])
    # pre_list=[]
    # for i in range(len(y_pred)):
    #     pre_temp=[]
    #     for j in range(len(y_pred[0])):
    #         if y_pred[i][j]>=0.5:
    #             t=str(j+1)
    #             pre_temp.append(t)
    #     pre_list.append(pre_temp)
    # #pre_list=DataFrame(pre_list)
    #
    # test= pd.read_csv(r'../test.csv', encoding='utf-8')
    # test.drop(labels=None, axis=1, index=None, columns='Caption', inplace=True)
    # test['Labels']=pre_list
    # #print(test)
    # for i in range(len(test)):
    #     test['Labels'][i]=" ".join(test['Labels'][i])
    # #print(test)
    # out_path='../submit.csv'
    # test.to_csv(out_path, sep=',', index=False, header=True)



