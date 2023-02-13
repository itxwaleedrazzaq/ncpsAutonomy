import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from keras.models import Model
from tensorflow.keras.preprocessing.image import load_img

def preprocess(X,y):
    X = np.expand_dims(np.array(X),axis=0)
    X = X/255.0
    y = np.array(y)
    return X,y

def saliency_map(agent):
    outputs = [agent.layers[i].output for i in range(0,len(agent.layers))]
    model = Model(inputs=agent.inputs, outputs=outputs)
    img = load_img('figures/test.png', target_size=(96,96))
    feature_maps = model.predict(correct_dims(img))
    title=['Input','conv1','maxpool1','conv2','maxpool2','conv3','maxpool3','FC1','NCP']
    row = 3
    col = 3
    i = 0
    for fmap in feature_maps:
        if len(np.shape(fmap))==5:
            plt.subplot(row,col,i+1)
            plt.imshow(fmap[0,0,:,:,i])
            plt.title(title[i])
            plt.tight_layout()
        elif len(np.shape(fmap))==4:
            plt.subplot(row,col,i+1)
            plt.imshow(fmap[0,:,:,i])
            plt.title(title[i])
            plt.tight_layout()
        elif len(np.shape(fmap))==3:
            plt.subplot(row,col,i+1)
            plt.imshow(fmap[:,:,i])
            plt.title(title[i])
            plt.tight_layout()
        else:
            pass
        i+=1
        
    plt.savefig('figures/saliencymap.png')
    plt.show()

    
def NCP_stacking(ncp_cell):
    sns.set_style('white')
    plt.figure(figsize=(15,15))
    legend_handles = ncp_cell.draw_graph(layout='spiral',neuron_colors={'command':'tab:cyan'})
    plt.legend(handles=legend_handles,loc='upper center',bbox_to_anchor=(1,1))
    sns.despine(left=True, bottom=True)
    plt.tight_layout()
    plt.savefig('figures/NCP_Architecture.png')
    plt.show()

def correct_dims(img):
    return np.expand_dims(np.expand_dims(np.array(img),axis=0),axis=1)

def visualize_history(filename):
    hist = pd.read_csv(filename, sep=',', engine='python')
    plt.plot(hist['accuracy'])
    plt.plot(hist['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('figures/Accuracy.png')

    plt.plot(hist['loss'])
    plt.plot(hist['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('figures/Loss.png')
