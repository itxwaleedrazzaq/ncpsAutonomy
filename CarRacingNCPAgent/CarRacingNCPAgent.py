import numpy as np
import cv2
import os
import gym
import pickle
import seaborn as sns
from stable_baselines3 import PPO
import kerasncp as kncp
from utils import correct_dims
from kerasncp.tf import LTCCell
from keras.callbacks import CSVLogger,TensorBoard,ModelCheckpoint
from keras.models import Sequential 
from keras.optimizers import SGD
from keras.models import load_model
from keras.layers import InputLayer, Conv2D,MaxPool2D,Flatten
from keras.layers import Dense,RNN,Activation,Activation,TimeDistributed


class CarRacingNCPAgent:
    def __init__(
        self,
        INPUT_SHAPE = (None,96,96,3),
        learning_rate =  0.01,
        k = 20,
        n_epoch = 10,
        valid_split = 0.3,

        recorderdir       = 'recorder/memory.zip',
        training_data_file = 'data/data.pickle',
        MODEL_NAME = 'CarRacingNCPAgent',
        MODEL_PATH = 'models/CarRacingNCPAgent.h5',
        
        inter_neurons= 12,
        command_neurons=8, 
        motor_neurons=5,
        sensory_fanout=4,
        inter_fanout=4,
        recurrent_command_synapses=6,
        motor_fanin=4):


        self.k = k
        self.n_epoch = n_epoch
        self.valid_split = valid_split
        self.INPUT_SHAPE = INPUT_SHAPE
        self.learning_rate              = learning_rate
        self.recorderdir                = recorderdir
        self.training_data_file         = training_data_file
        self.MODEL_NAME = MODEL_NAME
        self.MODEL_PATH = MODEL_PATH
        self.inter_neurons              = inter_neurons
        self.command_neurons            =command_neurons 
        self.motor_neurons              =motor_neurons
        self.sensory_fanout             =sensory_fanout
        self.inter_fanout               =inter_fanout
        self.recurrent_command_synapses =recurrent_command_synapses
        self.motor_fanin                =motor_fanin
    

    def record_data(self,env,iter=50):
        training_data = []
        if os.path.exists(self.training_data_file):
            print('----Data is already available in data folder----')
        else:
            print('----Recording Data in data folder----')
            recorder = PPO.load(self.recorderdir,env=env)
            obs = env.reset()
            for i in range(1,iter+1):
                obs = env.reset()
                done = False
                while not done:
                    #env.render()
                    action,_ = recorder.predict(obs)
                    training_data.append([obs,action])
                    obs,_,done,_ = env.step(action)
                print('iteration number {} is recorded'.format(i))
            env.close()
            pickle_out = open(self.training_data_file,'wb')
            pickle.dump(training_data,pickle_out)
            print('---Training Data has been successfully recorded in data---')
    

    def test_recorded_data(self):
        if os.path.exists(self.training_data_file):
            pickle_in = open(self.training_data_file,'rb')
            data = pickle.load(pickle_in)
            for obs,action in data:
                print(action)
                cv2.imshow('Observation Space',obs)
                if cv2.waitKey(1)==27:
                    break
            cv2.destroyAllWindows()    
        else:
            print('Training Data file is not available')


    def NCP(self):
        ncp_wiring = kncp.wirings.NCP(
                        inter_neurons =self.inter_neurons, 
                        command_neurons=self.command_neurons, 
                        motor_neurons=self.motor_neurons, 
                        sensory_fanout=self.sensory_fanout, 
                        inter_fanout=self.inter_fanout,  
                        recurrent_command_synapses=self.recurrent_command_synapses, 
                        motor_fanin=self.motor_fanin,)

        ncp_cell = LTCCell(ncp_wiring)
        return ncp_cell


    def build(self,ncp_cell):
        agent = Sequential([
            InputLayer(input_shape=self.INPUT_SHAPE),
            TimeDistributed(Conv2D(10,(3,3),activation='relu',name='ConvNet1')),
            TimeDistributed(MaxPool2D(pool_size=(2,2),name='MaxPool1')),
            TimeDistributed(Conv2D(20,(5,5),activation='relu',name='ConvNet2')),
            TimeDistributed(MaxPool2D(pool_size=(2,2),name='MaxPool2')),
            TimeDistributed(Conv2D(30,(5,5),activation='relu',name='ConvNet3')),
            TimeDistributed(MaxPool2D(pool_size=(2,2),name='MaxPool3')),
            TimeDistributed(Flatten()),
            TimeDistributed(Dense(units=32,activation='relu',name='FC1')),    
            RNN(ncp_cell,return_sequences=True,name='NCPCell'),
            TimeDistributed(Activation('softmax'))])
        agent.compile(optimizer=SGD(self.learning_rate),loss='sparse_categorical_crossentropy',metrics=['accuracy'])
        return agent


    def summary(self,agent):
        agent.summary()

    def load_data(self):
        X = []
        y = []
        if os.path.exists(self.training_data_file):
            pickle_in = open(self.training_data_file,'rb')
            data = pickle.load(pickle_in)
            for obs,action in data:
                X.append(obs)
                y.append(action)
            return(X,y)
        else:
            print('Training Data file is not available')


    def learn(self,agent,X,y):
        if os.path.exists(('models/'+(self.MODEL_NAME+'.h5'))):
            print('MODEL IS ALREADY PRESENT')
        else:
            tensorboard = TensorBoard(log_dir='logs/{}'.format(self.MODEL_NAME))
            csv_log = CSVLogger((self.MODEL_NAME+'.log'),separator=',',append=False)
            checkpoint = ModelCheckpoint(filepath=('models/'+(self.MODEL_NAME+'.h5')),monitor='val_loss',
                                            verbose=0,save_best_only=True,mode='auto')
            agent.fit(X,y,epochs=self.n_epoch,batch_size=self.k,shuffle=True,
                                                validation_split=self.valid_split,
                                                callbacks=[tensorboard,csv_log,checkpoint])

    def test_agent(self,env,steps):
        if os.path.exists(self.MODEL_PATH): 
            agent = load_model(self.MODEL_PATH,custom_objects={'LTCCell':LTCCell})
            obs = env.reset()
            for i in range(steps):
                done = False
                while not done:
                    env.render()
                    predicted_action = agent.predict(correct_dims(obs))
                    #print(predicted_action)
                    obs,reward,done,_ = env.step(np.argmax(predicted_action))
                obs = env.reset()
            env.close()
        else:
            print('AGENT NOT FOUND -- train using agent.learn()---')



