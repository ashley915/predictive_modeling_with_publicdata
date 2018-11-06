from keras.models import Sequential
from keras.layers import Dense, UpSampling2D
from keras.callbacks import ModelCheckpoint, EarlyStopping
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import chardet
from keras.optimizers import SGD
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
import datetime

filename=input("파일명을 입력하세요")

start = datetime.datetime.now()
#seed 값 설정
seed = 0
np.random.seed(seed)
tf.set_random_seed(seed)
dataset = np.loadtxt('./data/mydata_analysis_5years_trainset_rmhcorr_181011.csv',delimiter=',', skiprows=1, encoding='utf-8')



#x,y  train 설정
# x = dataset[:,:-1]
x_train = dataset[:,:-1]
y_train = dataset[:, -1]
# y = dataset[:, -1]
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)


#over and under sampling

# ros = RandomOverSampler(random_state=0)
# ros.fit(x_train, y_train)
rus = RandomUnderSampler(random_state=0)
rus.fit(x_train,y_train)
# x_resampled, y_resampled = ros.fit_sample(x_train, y_train)
x_resampled, y_resampled = rus.fit_sample(x_train, y_train)


#x,y validataion 설정
data = np.loadtxt('./data/mydata_analysis_5years_testset_rmhcorr_181011.csv', delimiter=',', skiprows=1, encoding='utf-8')
x_test = data[:,:-1]
y_test = data[:,-1]
print(x_test)
print(y_test)


#model 쌓기
model=Sequential()

model.add(Dense(5,input_dim=37))
model.add(Dense(3, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


sgd = SGD(lr=0.0000001, decay=1e-2, momentum=0.9)
model.compile(loss='mean_squared_error',
optimizer='adam',
metrics=['accuracy'])

history = model.fit(x_resampled, y_resampled, validation_data=(x_test, y_test), epochs=10, batch_size=10)
output = model.predict_classes(x_test, verbose=1)
output_train = model.predict_classes(x_train, verbose=1)
res = model.predict(x_test)
print(output)
print("\n Accuracy: %.4f" % (model.evaluate(x_test, y_test)[1]))
evaluate = model.evaluate(x_test,y_test)

end = datetime.datetime.now()
fin = end-start
print("\n",fin,"seconds used")

from sklearn.metrics import confusion_matrix
cm1 = confusion_matrix(y_test,output)
print('Confusion Matrix : \n', cm1)

total1=sum(sum(cm1))
#####from confusion matrix calculate accuracy
accuracy1=(cm1[0,0]+cm1[1,1])/total1
print ('Accuracy : ', accuracy1)

sensitivity1 = cm1[0,0]/(cm1[0,0]+cm1[0,1])
print('Sensitivity : ', sensitivity1 )

specificity1 = cm1[1,1]/(cm1[1,0]+cm1[1,1])
print('Specificity : ', specificity1)

# #plot accuracy
# plt.plot(history.history['acc'])
# plt.plot(history.history['val_acc'])
# plt.title('Model accuracy')
# plt.ylabel('Accuracy')
# plt.xlabel('Epoch')
# plt.legend(['Train', 'Test'], loc='upper left')
# plt.show()
#
# #plot loss
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('Model loss')
# plt.ylabel('Loss')
# plt.xlabel('Epoch')
# plt.legend(['Train', 'Test'], loc='upper left')
# plt.show()




if not os.path.exists('./results/'):
    os.makedirs('./results/')

outcome = open('./results/'+filename+'_NN_results.txt','w')
result = open('./results/'+filename+'_trainset_NN_yhat.csv','w')
result_train = open('./results/'+filename+'_testset_NN_yhat.csv','w')
np.savetxt(result, output, fmt='%i', delimiter=',')
np.savetxt(result_train, output_train, fmt='%i', delimiter=',')
outcome.write('Test loss:'+ "%.2f"%evaluate[0])
outcome.write('\n'+'Test accuracy:'+ "%.2f"%evaluate[1])
outcome.write('\n'+'Sensitivity : '+ str(sensitivity1) )
outcome.write('\n'+'Specificity : '+ str(specificity1))
outcome.write('\n'+'time : '+ str(fin)+" seconds used")
result.close()
outcome.close()
print("DONE!!")