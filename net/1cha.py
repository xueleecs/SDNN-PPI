from matplotlib.cbook import flatten
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.layers import Dense, Input, Dropout, Conv1D, Reshape, MaxPooling1D, ZeroPadding1D, AveragePooling1D
from keras.layers import Add, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, \
    MaxPooling2D, GlobalMaxPooling2D
# from keras.layers import Conv1D
from keras.layers.merge import concatenate
from keras.optimizers import SGD
from keras.optimizers import Adam
from keras.models import Model
from keras.regularizers import l2, l1
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve
from keras.utils import np_utils
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
import sklearn.metrics as metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import matthews_corrcoef, accuracy_score, precision_score, recall_score
from sklearn.manifold import TSNE
from keras.initializers import glorot_uniform

from xgboost import XGBClassifier
import time
from attention import *
from keras import backend as K
start = time.time()
lr = 0.01
epochs = 50
batch_size = 64
dr = 0.12
l2c = 0.001


def define_model():
    
    ########################################################"Channel-1" ########################################################
    
    input_1 = Input(shape=(1146, ), name='Protein_a')
    p11 = Dense(1024, activation='relu', kernel_initializer='glorot_normal', name='ProA_feature_111',
                kernel_regularizer=l2(l2c))(input_1)
    p11 = BatchNormalization(axis=-1)(p11)
    p11 = Dropout(dr)(p11)

    p12 = Dense(512, activation='relu', kernel_initializer='glorot_normal', name='ProA_feature_1',
                kernel_regularizer=l2(l2c))(p11)
    p12 = BatchNormalization(axis=-1)(p12)
    p12 = Dropout(dr)(p12)

    p13 = Dense(256, activation='relu', kernel_initializer='glorot_normal', name='ProA_feature_2',
                kernel_regularizer=l2(l2c))(p12)
    p13 = BatchNormalization(axis=-1)(p13)
    p13 = Dropout(dr)(p13)

    p13 = Reshape((4,64))(p13)

    ##attention
    d10 = p13.get_shape().as_list()
    d10 = d10[2]
    X = Self_Attention(d10)(p13)
    p13 = BatchNormalization(axis=-1)(X)
    p13 = Dropout(dr)(p13)

    p13=Flatten()(p13)

    p14 = Dense(128, activation='relu', kernel_initializer='glorot_normal', name='ProA_feature_3',
                kernel_regularizer=l2(l2c))(p13)
    p14 = BatchNormalization(axis=-1)(p14)
    p14 = Dropout(dr)(p14)

    p15 = Dense(64, activation='relu', kernel_initializer='glorot_normal', name='ProA_feature_4',
                kernel_regularizer=l2(l2c))(p14)
    p15 = BatchNormalization(axis=-1)(p15)
    p15 = Dropout(dr)(p15)

    p16 = Dense(32, activation='relu', kernel_initializer='glorot_normal', name='ProA_feature_14',
                kernel_regularizer=l2(l2c))(p15)
    p16 = BatchNormalization(axis=-1)(p16)
    p16 = Dropout(dr)(p16)
    

    
    ##################################### Prediction Module ##########################################################
    
    pre_output = Dense(32, activation='relu', kernel_initializer='glorot_normal', name='Merged_feature_2')(p16)

    pre_output = Reshape((4,8))(pre_output)

    ##attention
    d30 = pre_output.get_shape().as_list()
    d30 = d30[2]
    X = Self_Attention(d30)(pre_output)
    pre_output = BatchNormalization(axis=-1)(X)
    pre_output = Dropout(dr)(pre_output)

    pre_output=Flatten()(pre_output)

    pre_output = Dense(16, activation='relu', kernel_initializer='he_uniform', name='Merged_feature_3')(pre_output)

    pre_output = Dense(8, activation='relu', kernel_initializer='he_uniform', name='Merged_feature_4')(pre_output)

    pre_output = Dropout(dr)(pre_output)

    output = Dense(1, activation='sigmoid', name='output')(pre_output)
    model = Model(input=input_1, output=output)
   
    sgd = SGD(lr=0.01, momentum=0.9, decay=0.001)
    model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
    return model


##################################### Load Positive and Negative Dataset ##########################################################
    
df_pos = pd.read_csv('/PositiveScore.csv', header=None)
df_neg = pd.read_csv('/NegativeScore.csv', header=None)

df_neg['Status'] = 0
df_pos['Status'] = 1
df_neg=df_neg.sample(n=len(df_pos))

df = pd.concat([df_pos,df_neg])
df = df.reset_index()
df=df.sample(frac=1)
df = df.iloc[:,1:]

X = df.iloc[:,0:1146].values
y = df.iloc[:,1146:].values
Trainlabels=y
scaler = StandardScaler().fit(X)

X = scaler.transform(X)

##################################### Five-fold Cross-Validation ##########################################################
    
kf=StratifiedKFold(n_splits=5)


accuracy1 = []
specificity1 = []
sensitivity1 = []
precision1=[]
recall1=[]

m_coef=[]
dnn_fpr_list=[]
dnn_tpr_list=[]
dnn_auc_list = []
o=0
max_accuracy=float("-inf")
dnn_fpr=None
dnn_tpr=None

for train, test in kf.split(X,y):
    global model
    model=define_model()
    o=o+1

    model.fit([X[train]],y[train],epochs=epochs, batch_size=batch_size,verbose=1)

    y_test=y[test]
    y_score = model.predict(X[test])
    
    fpr, tpr, _= roc_curve(y_test,  y_score)
    auc = metrics.roc_auc_score(y_test, y_score)
    
    dnn_auc_list.append(auc)
    
    y_score=y_score[:,0]
    
    for i in range(0,len(y_score)):
        if(y_score[i]>0.5):
            y_score[i]=1
        else:
            y_score[i]=0
            
    cm1=confusion_matrix(y[test][:,0],y_score)
    acc1 = accuracy_score(y[test][:,0], y_score, sample_weight=None)
    spec1= (cm1[0,0])/(cm1[0,0]+cm1[0,1])
    sens1 = recall_score(y[test][:,0], y_score, sample_weight=None)
    prec1=precision_score(y[test][:,0], y_score, sample_weight=None)
    

    sensitivity1.append(sens1)
    specificity1.append(spec1)
    accuracy1.append(acc1)
    precision1.append(prec1)
    
    coef=matthews_corrcoef(y[test], y_score, sample_weight=None)
    m_coef.append(coef)
    dnn_fpr_list.append(fpr)
    dnn_tpr_list.append(tpr)

    if acc1>max_accuracy:
        max_accuracy=acc1
        dnn_fpr=fpr[:]
        dnn_tpr=tpr[:]

dnn_fpr=pd.DataFrame(dnn_fpr)
dnn_tpr=pd.DataFrame(dnn_tpr)

mean_acc1 = np.mean(accuracy1)
mean_acc1 = np.mean(accuracy1)
std_acc1 = np.std(accuracy1)
var_acc1 = np.var(accuracy1)
print("Accuracy1:" + str(mean_acc1) + " Â± " + str(std_acc1))
print("Accuracy_Var:" + str(mean_acc1) + " Â± " + str(var_acc1))

mean_spec1 = np.mean(specificity1)
std_spec1 = np.std(specificity1)
print("Specificity1:" + str(mean_spec1) + " Â± " + str(std_spec1))

mean_sens1 = np.mean(sensitivity1)
std_sens1 = np.std(sensitivity1)
print("Sensitivity1:" + str(mean_sens1) + " Â± " + str(std_sens1))

mean_prec1 = np.mean(precision1)
std_prec1 = np.std(precision1)
print("Precison1:" + str(mean_prec1) + " Â± " + str(std_prec1))

mean_coef = np.mean(m_coef)
std_coef = np.std(m_coef)
print("MCC1:" + str(mean_coef) + " Â± " + str(std_coef))

print("dnn_auc_list:" + str(dnn_auc_list))
print("AUC1:" + str(np.mean(dnn_auc_list)))

end1 = time.time()
end11 = end1 - start
print(f"Runtime of the program is {end1 - start}")