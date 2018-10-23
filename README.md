================================

########项目介绍#################
1.Keras多层感知器法识别手写数字
2.keras卷积神经网络识别手写数字

#################################

#######多层感知器法主要过程及相关函数库###############

1.数据预处理

#将features (数字图像特征值)使用reshape 转换
x_train=x_train_image.reshape(60000,784).astype('float32')
x_test=x_test_image.reshape(10000,784).astype('float32')
#标准化
x_train_normalize=x_train/255
x_test_normalize=x_test/255
#one_hot encoding转换
y_train_onehot=np_utils.to_categorical(y_train_label)
y_test_onehot=np_utils.to_categorical(y_test_label)

2.建立模型

#建立Sequential
model=Sequential()
#建立“输入层”“隐藏层”
model.add(Dense(units=256,input_dim=784,kernel_initializer='normal',activation='relu'))
#建立“输出层”
model.add(Dense(units=10,kernel_initializer='normal',activation='softmax'))

3.训练模型

#定义训练方式
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
#开始训练
train_history=model.fit(x=x_train_normalize,y=y_train_onehot,validation_split=0.2,epochs=10,batch_size=200,verbose=2)

4.评估模型准确率

#评估模型准确率
scores=model.evaluate(x_test_normalize,y_test_onehot)

5.进行预测

#进行预测
prediction=model.predict_classes(x_test)
prediction

6.其中通过改变隐藏层神经元，加入dropout和建立两个隐藏层等方式，提高训练效果，解决了过度拟合问题，增加了准确率。

########################################################################################################



###########卷积神经网络法主要过程及相关函数########################################

1.数据预处理

#将features转换为四维矩阵
x_train4d=x_train.reshape(x_train.shape[0],28,28,1).astype('float32')
x_test4d=x_test.reshape(x_test.shape[0],28,28,1).astype('float32')
#将features标准化
x_train4d_normalize=x_train4d/255
x_test4d_normalize=x_test4d/255
#label(数字真实值)以one-hot encoding进行转换
y_trainonehot=np_utils.to_categorical(y_train)
y_testonehot=np_utils.to_categorical(y_test)

2.建立模型

#建立卷积层1
model.add(Conv2D(filters=16,kernel_size=(5,5),padding='same',input_shape=(28,28,1),activation='relu'))
#建立池化层1
model.add(MaxPooling2D(pool_size=(2,2)))
#建立卷积层2
model.add(Conv2D(filters=36,kernel_size=(5,5),padding='same',activation='relu'))
#建立池化层2
model.add(MaxPooling2D(pool_size=(2,2)))
#加入dropout模型
model.add(Dropout(0.25))
#建立平坦层
model.add(Flatten())
#建立隐藏层0
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.5))
#建立输出层
model.add(Dense(10,activation='softmax'))

3.训练模型

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
train_history=model.fit(x=x_train4d_normalize,y=y_trainonehot,validation_split=0.2,epochs=10,batch_size=300,verbose=2)

4.评估模型准确率

#评估模型准确率
scores=model.evaluate(x_test4d_normalize,y_testonehot)
scores[1]

5.进行预测

prediction=model.predict_classes(x_test4d_normalize)
prediction[:10]
