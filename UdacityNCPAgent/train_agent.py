import os
import pandas as pd
import kerasncp as kncp
from kerasncp.tf import LTCCell
from utils import batch_generator
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint 
from keras.callbacks import CSVLogger,TensorBoard
from keras.layers import InputLayer, Conv2D,MaxPool2D,Dropout,Flatten,Dense,RNN,Activation,Reshape,Lambda,Activation,Permute
from keras.optimizers import RMSprop

MODEL_NAME = 'LTCvNCP'

tensorboard = TensorBoard(log_dir='logs/{}'.format(MODEL_NAME))
csv_log = CSVLogger('LTCvNCP_training.log',separator=',',append=False)

dir = os.getcwd()
data_dir = os.path.join(dir,'data')
df = pd.read_csv(os.path.join(data_dir,'driving_log.csv'), names=['center', 'left', 'right', 'steering', 'throttle', 'reverse', 'speed'])
X = df[['center','left','right']].values
y = df[['steering']].values
x_train,x_valid,y_train,y_valid = train_test_split(X,y,test_size=0.2,random_state=0)


def ReshapeLayer(x):
    shape = x.shape
    transpose = Permute((2,1,3))(x)
    reshape = Reshape((shape[1],shape[2]*shape[3]))(transpose)
    return reshape



ncp_wiring = kncp.wirings.NCP(
    inter_neurons=12,
    command_neurons=8, 
    motor_neurons=1, 
    sensory_fanout=4, 
    inter_fanout=4, 
    recurrent_command_synapses=6,
    motor_fanin=4,  
)
ncp_cell = LTCCell(ncp_wiring)

model = Sequential()
model.add(InputLayer(input_shape=(160,320,3)))

model.add(Conv2D(128,(3,3),strides=(2,2),activation='relu',name='conv1'))
model.add(MaxPool2D(pool_size=(3,3),name='pool1'))

model.add(Conv2D(128,(3,3),strides=(2,2),activation='relu',name='conv2'))
model.add(MaxPool2D(pool_size=(3,3),name='pool2'))

model.add(Dropout(0.2))
model.add(Lambda(ReshapeLayer,name='reshape'))
model.add(Dense(100,activation='relu',name='Dense1'))
model.add(Dense(12,activation='relu',name='Dense3'))

model.add(RNN(ncp_cell,name='NCP'))
model.add(Activation('tanh'))

opt = RMSprop(lr=1e-7,decay=1e-10)
model.compile(optimizer=opt,loss='MeanSquaredError',metrics=['accuracy'])


model.summary()

checkpoint = ModelCheckpoint(MODEL_NAME+'.h5',monitor='val_loss',verbose=0,save_best_only=True,mode='auto')

Hist = model.fit_generator(batch_generator(data_dir, x_train, y_train, 50,False),
                        100,
                        20,
                        validation_data=batch_generator(data_dir, x_valid, y_valid, 50,False),
                        validation_steps=10,
                        verbose=1,
                        callbacks=[checkpoint,tensorboard,csv_log])

#model.fit(X,y,epochs=10,batch_size=20,shuffle=True,validation_split=0.3)