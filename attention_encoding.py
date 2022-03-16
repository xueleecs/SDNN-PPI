import numpy as np
import pandas as pd
from keras.layers import Dense, Input, Dropout, Conv1D, Reshape, MaxPooling1D, ZeroPadding1D, AveragePooling1D
from keras.layers import  Add, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
# from keras.layers import Conv1D
from keras.layers.merge import concatenate
from keras.optimizers import SGD
from keras.models import Model
from keras.regularizers import l2
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve
from keras.utils import np_utils
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
import sklearn.metrics as metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import matthews_corrcoef,accuracy_score, precision_score,recall_score
from sklearn.manifold import TSNE
from attention import *

from keras.initializers import glorot_uniform

from xgboost import XGBClassifier
import time

start = time.time()

def define_model():
    
    ########################################################"Channel-1" ########################################################
    
    input_1 = Input(shape=(573, ), name='Protein_a')

    p1 = Reshape((3, 191))(input_1)


    ##attention
    d1 = p1.get_shape().as_list()
    d1 = d1[2]
    X = Self_Attention(d1)(p1)

    # 均值池化
    X = AveragePooling1D(pool_size=2, padding='same')(X)

    ########################################################"Channel-2" ########################################################

    input_2 = Input(shape=(573, ), name='Protein_b')

    p2 = Reshape((3, 191))(input_2)

    ##attention
    d10 = p2.get_shape().as_list()
    d10 = d10[2]
    p2 = Self_Attention(d10)(p2)

    # 均值池化
    p2 = AveragePooling1D(pool_size=2, padding='same')(p2)

    ##################################### Merge Abstraction features ##################################################
    
    merged = concatenate([X,p2], name='merged_protein1_2')
    
    ##################################### Prediction Module ##########################################################
    
   
    merged = Flatten()(merged)

    pre_output = Dense(64, activation='relu', kernel_initializer='glorot_normal', name='Merged_feature_1')(merged)
    pre_output = Dense(32, activation='relu', kernel_initializer='glorot_normal', name='Merged_feature_2')(pre_output)
    pre_output = Dense(16, activation='relu', kernel_initializer='he_uniform', name='Merged_feature_3')(pre_output)


    
    pre_output=Dropout(0.2)(pre_output)

    output = Dense(1, activation='sigmoid', name='output')(pre_output)

    model = Model(input=[input_1, input_2], output=output)
   
    sgd = SGD(lr=0.01, momentum=0.9, decay=0.001)

    model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])

    return model


##################################### Load Positive and Negative Dataset ##########################################################
    
df_pos = pd.read_csv('/home/aita/4444/LX/data_method/HP/PositiveHP.csv',header=None)
df_neg = pd.read_csv('/home/aita/4444/LX/data_method/HP/NegativeHP.csv',header=None)

df_neg['Status'] = 0
df_pos['Status'] = 1#在最后添加一列标签
df_neg=df_neg.sample(n=len(df_pos))#推测保证阳性样本和阴性样本数量一样

df = pd.concat([df_pos,df_neg])#上下拼接
df = df.reset_index()# 在第一列加了索引
df=df.sample(frac=1)#frac=1表示取样本的比例，但是顺序会乱。
df = df.iloc[:,1:]#去掉第一列，不太对啊

X = df.iloc[:,0:1146].values
y = df.iloc[:,1146:].values
Trainlabels=y
scaler = StandardScaler().fit(X)
#scaler = RobustScaler().fit(X)
X = scaler.transform(X)

X1_train = X[:, :573]
X2_train = X[:, 573:]

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

    model.fit([X1_train[train],X2_train[train]],y[train],epochs=50,batch_size=64,verbose=1)

    y_test=y[test]

    y_score = model.predict([X1_train[test],X2_train[test]])

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
    # dnn_fpr_list.append(fpr)
    # dnn_tpr_list.append(tpr)

    if acc1>max_accuracy:
        max_accuracy=acc1
        dnn_fpr=fpr[:]
        dnn_tpr=tpr[:]

dnn_fpr=pd.DataFrame(dnn_fpr)
dnn_tpr=pd.DataFrame(dnn_tpr)
dnn_fpr.to_csv('fprDNN.csv',header=False, index=False)
dnn_tpr.to_csv('tprDNN.csv',header=False, index=False)

##########################创建保存文件##############################
import sys  # 需要引入的包
 
# 以下为包装好的 Logger 类的定义
class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")
        # self.log = open(filename, "a", encoding="utf-8")  # 防止编码错误
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    def flush(self):
        pass

#########################保存以下信息###############################
sys.stdout = Logger('/home/aita/4444/LX/NO.1/test1 encoding/A.txt')

print("This is 13")
mean_acc1=np.mean(accuracy1)
std_acc1=np.std(accuracy1)
var_acc1=np.var(accuracy1)
print("Accuracy1:"+str(mean_acc1)+" Â± "+str(std_acc1))
print("Accuracy_Var:"+str(mean_acc1)+" Â± "+str(var_acc1))
mean_spec1=np.mean(specificity1)
std_spec1=np.std(specificity1)
print("Specificity1:"+str(mean_spec1)+" Â± "+str(std_spec1))
mean_sens1=np.mean(sensitivity1)
std_sens1=np.std(sensitivity1)
print("Sensitivity1:"+str(mean_sens1)+" Â± "+str(std_sens1))
mean_prec1=np.mean(precision1)
std_prec1=np.std(precision1)
print("Precison1:"+str(mean_prec1)+" Â± "+str(std_prec1))

mean_coef=np.mean(m_coef)
std_coef=np.std(m_coef)
print("MCC1:"+str(mean_coef)+" Â± "+str(std_coef))

print("AUC1:"+str(np.mean(dnn_auc_list)))


end1 = time.time()
end11=end1 - start
print(f"Runtime of the program is {end1 - start}")

######################################## ROC Curve plot ###############################################

def ROC_dnn():
    plt.figure(figsize=(3,2),dpi=300)
    
    plt.plot(dnn_fpr,dnn_tpr)
    plt.legend()
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')        
    
    plt.title("ROC Curve for DNN")
    plt.show()     

def ROC_dnn_xgb(): # Enter ROC_dnn_xgb() in console to see the roc-auc plot for XGB Classifier
    plt.figure(figsize=(3,2),dpi=300)
    plt.plot(xgb_fpr,xgb_tpr)   
    plt.legend()
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')           
    plt.title("ROC Curve for DNN_XGB Classifier")
    plt.show()
    

######################################## TSNE plot ###############################################

def TSNE_raw():
    global raw_data
    raw_data= pd.concat([df_pos,df_neg])
    raw_data=raw_data.iloc[:,:-1]
    t=TSNE(n_components=2).fit_transform(raw_data)
    pos_t=t[:int(len(t)/2),:]
    neg_t=t[int(len(t)/2):,:]
    plt.scatter(pos_t[:,0],pos_t[:,1],label="Positive",s=4)
    plt.scatter(neg_t[:,0],neg_t[:,1],label="Negative",s=4)
    plt.legend()
    plt.show()

TSNE_raw()

def TSNE_extracted():
    
    pos=extracted_df[extracted_df.iloc[:,-1]==1]
    neg=extracted_df[extracted_df.iloc[:,-1]==0]
    X_feat=pd.concat([pos,neg])
    X_feat=X_feat.iloc[:,:-1]
    t=TSNE(n_components=2).fit_transform(X_feat)
    pos_t=t[:int(len(t)/2),:]
    neg_t=t[int(len(t)/2):,:]
    plt.scatter(pos_t[:,0],pos_t[:,1],label="Positive",s=4)
    plt.scatter(neg_t[:,0],neg_t[:,1],label="Negative",s=4)
    plt.legend()
    plt.show()

TSNE_extracted()