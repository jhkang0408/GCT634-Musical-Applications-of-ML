# GCT634 (2018) HW1
#
# Mar-18-2018: initial version
#
# Juhan Nam
#

import sys
import os
import numpy as np
import librosa
from feature_summary import *
from sklearn.metrics import confusion_matrix, plot_confusion_matrix

# classifier
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.ensemble import RandomForestClassifier

# dimension reduction
from sklearn.decomposition import PCA

def train_model(train_X, train_Y, valid_X, valid_Y, model_name, hyper_param1):
    
    if model_name == 'Linear_SVM':        
        clf = SGDClassifier(verbose=0, loss="hinge", alpha=hyper_param1, max_iter=1000, penalty="l2", random_state=0)
    elif model_name == 'Non_Linear_SVM': 
        clf = SVC(C=hyper_param1, kernel='rbf')
    elif model_name == 'MLP':
        clf = MLPClassifier(activation='relu', hidden_layer_sizes=(train_X.shape[1], 10), solver='adam', learning_rate_init=hyper_param1, max_iter=1000) 
    elif model_name == 'KNN':
        clf = KNeighborsClassifier(n_neighbors=hyper_param1)
    elif model_name =='Random_Forest':
        clf = RandomForestClassifier(n_estimators=hyper_param1)
        
    # train
    clf.fit(train_X, train_Y)

    # train
    train_Y_hat = clf.predict(train_X)
    accuracy = np.sum((train_Y_hat == train_Y))/1100.0*100.0
    print('train accuracy = ' + str(accuracy) + ' %')
 
    # validation
    valid_Y_hat = clf.predict(valid_X)
    accuracy = np.sum((valid_Y_hat == valid_Y))/300.0*100.0
    print('validation accuracy = ' + str(accuracy) + ' %\n')
    
    return clf, accuracy

if __name__ == '__main__':

    # load data
    train_X = np.concatenate((mean_std_perceptionbased(dataset='train'), mean_std_deltaperceptionbased(dataset='train'), mean_std_deltamfcc(dataset='train'), codebook_based_feature_summarization(dataset='train')), axis=0)
    valid_X = np.concatenate((mean_std_perceptionbased(dataset='valid'), mean_std_deltaperceptionbased(dataset='valid'), mean_std_deltamfcc(dataset='valid'), codebook_based_feature_summarization(dataset='valid')), axis=0)
    
    # train_X = mean_std_mfcc(dataset='train')
    # valid_X = mean_std_mfcc(dataset='valid')
    
    # label generation
    cls = np.array([1,2,3,4,5,6,7,8,9,10])
    train_Y = np.repeat(cls, 110)
    valid_Y = np.repeat(cls, 30)

    # # feature normalizaiton
    train_X = train_X.T
    train_X_mean = np.mean(train_X, axis=0)
    train_X = train_X - train_X_mean
    train_X_std = np.std(train_X, axis=0)
    train_X = train_X / (train_X_std + 1e-5)
    
    valid_X = valid_X.T
    valid_X = valid_X - train_X_mean
    valid_X = valid_X/(train_X_std + 1e-5)

    # # dimension reduction    
    # dr = PCA(n_components=50, svd_solver='full')
    # dr.fit_transform(train_X)
    # plt.figure()
    # plt.plot(dr.explained_variance_)
    # plt.title('Scree plot of eigenvalues')
    # plt.xlabel('Number of principal components')
    # plt.ylabel('Eigenvalues')
    
    model_name = 'Non_Linear_SVM'
    
    if model_name == 'Linear_SVM':
        alphas = [1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3]
    elif model_name == 'Non_Linear_SVM': 
        alphas = [1e-2, 1e-1, 1, 1e1, 1e2, 1e3]
    elif model_name == 'MLP':
        alphas = [1e-4, 1e-3, 1e-2, 1e-1]
    elif model_name == 'KNN':
        alphas = [1,3,5,7,9]
    elif model_name =='Random_Forest':
        alphas = [500, 1000, 1500, 2000]
    
    model = []
    valid_acc = []
    for a in alphas:
        clf, acc = train_model(train_X, train_Y, valid_X, valid_Y, model_name, a)
        model.append(clf)
        valid_acc.append(acc)
        
    # choose the model that achieve the best validation accuracy
    final_model = model[np.argmax(valid_acc)]

    # now, evaluate the model with the test set
    valid_Y_hat = final_model.predict(valid_X)

    accuracy = np.sum((valid_Y_hat == valid_Y))/300.0*100.0
    print('final validation accuracy = ' + str(accuracy) + ' %')
    print('final conufsion matrix = ')
    print(confusion_matrix(valid_Y, valid_Y_hat))
    
    # confusion matrix plot    
    instruments = ['Bass', 'Brass', 'Flute', 'Guitar', 'Keyboard', 'Mallet', 'Organ', 'Reed', 'String', 'Vocal']
    plot = plot_confusion_matrix(final_model, 
                                 valid_X, valid_Y, 
                                 display_labels=instruments, 
                                 cmap=plt.cm.Blues,
                                 xticks_rotation=45,
                                 normalize=None)
    plot.ax_.set_title('Instrument Confusion Matrix')        
    
    # Dimensionality Reduction PCA Plot
    dr = PCA(n_components=3, svd_solver='full')
    train_X = (dr.fit_transform(train_X))
    plt.figure()  
    plt.xlabel('Principal Component1')
    plt.ylabel('Principal Component2') 
    plt.title('PCA Result (2D)')
    for i in range(10): 
        plt.plot(train_X[:,0][train_Y==i], train_X[:,1][train_Y==i], 'o', ms=4.5, label=instruments[i], alpha=.5)    
    plt.legend(loc=1, prop={'size': 8})      

    fig = plt.figure() 
    ax = fig.add_subplot(111, projection='3d')  
    for i in range(10): 
        ax.scatter(train_X[:,0][train_Y==i], train_X[:,1][train_Y==i], train_X[:,2][train_Y==i], label=instruments[i],alpha=0.5, s=1) 
    ax.set_xlabel('Principal Component1')
    ax.set_ylabel('Principal Component2')
    ax.set_zlabel('Principal Component3')        
    plt.legend()       
    plt.title("PCA Result (3D)") 
    plt.legend(loc=1, prop={'size': 7})
    plt.draw()         
        