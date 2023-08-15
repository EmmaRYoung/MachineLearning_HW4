import numpy as np
from sklearn.svm import SVC

import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import numpy.linalg as LA
import scipy.io
mat = scipy.io.loadmat('mnist_all.mat')


#read in and prepare data

#read in and prepare training data
keep0 = 1000 #Training data has same number of samples for each class (easier)
train0 = mat["train0"]
train0 = train0[0:keep0,:]
train0 = train0/255  

train1 = mat["train1"]
train1 = train1[0:keep0,:]
train1 = train1/255  

train2 = mat["train2"]
train2 = train2[0:keep0,:]
train2 = train2/255  

train3 = mat["train3"]
train3 = train3[0:keep0,:]
train3 = train3/255  

train4 = mat["train4"]
train4 = train4[0:keep0,:]
train4 = train4/255  

train5 = mat["train5"]
train5 = train5[0:keep0,:]
train5 = train5/255  

train6 = mat["train6"]
train6 = train6[0:keep0,:]
train6 = train6/255  

train7 = mat["train7"]
train7 = train7[0:keep0,:]
train7 = train7/255  

train8 = mat["train8"]
train8 = train8[0:keep0,:]
train8 = train8/255  

train9 = mat["train9"]
train9 = train9[0:keep0,:]
train9 = train9/255  

#read in and prepare testing data
keep = 500
test0 = mat["test0"]
test0 = test0[0:keep,:]
test0 = test0/255  

test1 = mat["test1"]
test1 = test1[0:keep,:]
test1 = test1/255  

test2 = mat["test2"]
test2 = test2[0:keep,:]
test2 = test2/255  

test3 = mat["test3"]
test3 = test3[0:keep,:]
test3 = test3/255  

test4 = mat["test4"]
test4 = test4[0:keep,:]
test4 = test4/255  

test5 = mat["test5"]
test5 = test5[0:keep,:]
test5 = test5/255  

test6 = mat["test6"]
test6 = test6[0:keep,:]
test6 = test6/255  

test7 = mat["test7"]
test7 = test7[0:keep,:]
test7 = test7/255  

test8 = mat["test8"]
test8 = test8[0:keep,:]
test8 = test8/255  

test9 = mat["test9"]
test9 = test9[0:keep,:]
test9 = test9/255  

#Prepare data for 1 vs all
train0Vall = np.vstack((train0, train1[0:100], train2[0:100], train3[0:100], train4[0:100], train5[0:100], train6[0:100], train7[0:100], train8[0:100], train9[0:100]))
y = np.vstack((np.ones((np.shape(train0)[0],1)), -1*np.ones((900,1))))
#we can use the same ground truth vector for all the samples because the length of all training samples for each class is made equal above

train1Vall = np.vstack((train1, train0[0:100], train2[0:100], train3[0:100], train4[0:100], train5[0:100], train6[0:100], train7[0:100], train8[0:100], train9[0:100]))

train2Vall = np.vstack((train2, train0[0:100], train1[0:100], train3[0:100], train4[0:100], train5[0:100], train6[0:100], train7[0:100], train8[0:100], train9[0:100]))

train3Vall = np.vstack((train3, train0[0:100], train1[0:100], train2[0:100], train4[0:100], train5[0:100], train6[0:100], train7[0:100], train8[0:100], train9[0:100]))

train4Vall = np.vstack((train4, train0[0:100], train1[0:100], train2[0:100], train3[0:100], train5[0:100], train6[0:100], train7[0:100], train8[0:100], train9[0:100]))

train5Vall = np.vstack((train5, train0[0:100], train1[0:100], train2[0:100], train3[0:100], train4[0:100], train6[0:100], train7[0:100], train8[0:100], train9[0:100]))

train6Vall = np.vstack((train6, train0[0:100], train1[0:100], train2[0:100], train3[0:100], train4[0:100], train5[0:100], train7[0:100], train8[0:100], train9[0:100]))

train7Vall = np.vstack((train7, train0[0:100], train1[0:100], train2[0:100], train3[0:100], train4[0:100], train5[0:100], train6[0:100], train8[0:100], train9[0:100]))

train8Vall = np.vstack((train8, train0[0:100], train1[0:100], train2[0:100], train3[0:100], train4[0:100], train5[0:100], train6[0:100], train7[0:100], train9[0:100]))

train9Vall = np.vstack((train9, train0[0:100], train1[0:100], train2[0:100], train3[0:100], train4[0:100], train5[0:100], train6[0:100], train7[0:100], train8[0:100]))




#define fcns
def PCA(X):
    #step1: Calculate mean and std of the dataset (vector)
    [NumSample, NumFeature] = np.shape(X)
    x_mu = np.reshape(np.mean(X, axis=0), (NumFeature,1))
    
    #step2: subtract the mean from the dataset, construct "A" matrix
    A = np.transpose(X) - x_mu #(784 x 60,000)
    
    #Then construct AA' matrix "C": characterizes the scatter of the data
    C = np.zeros((NumFeature, NumFeature)) 
    for i in range(NumSample):
        vect = np.reshape(A[:,i], (NumFeature,1)) #reshape step makes matrix multiplication work
        C = C + vect@np.transpose(vect)
        
    C = C/NumSample #(784 x 784)
    
    #Compute the eigenvalues and eigenvectors of C
    [W, V] = LA.eig(C) #W: eigenvalues V: eigenvectors
    W = np.real(W)
    V = np.real(V)
    #The column v[:,i] is the eigenvector corresponding to 
    #the eigenvector corresponsing to the eigenvalue w[i]
    
    #find cutoff to maintain variance of 90% and 95% of the data
    thresh1 = 0.90
    thresh2 = 0.95
    
    total_eig = np.sum(W)
    eig_store = 0
    found = 0
    for i in range(len(W)):
        eig_store = eig_store + W[i]
        fraction = np.sum(eig_store)/total_eig
        if fraction > thresh1 and found!=1:
            keep90 = i-1
            found = 1 #the found boolean is so we can look for both keep90 and keep95 in one loop
            
        if fraction > thresh2:
            keep95 = i-1

    #Use for dimensionality reduction, basis (eigen vectors) are trunkated at the indices that represents
    #the desired ammount of variation
    
    #Dimensionality reduction is done in another function
    b90 = np.transpose(V[:,0:keep90])
    b95 = np.transpose(V[:,0:keep95])

    return b90, b95

def DimRed(X,basis):
    #reduce the dimensionality of the samples with the basis obtained from PCA
    
    [NumSample, NumFeature] = np.shape(X)
    x_mu = np.reshape(np.mean(X, axis=0), (NumFeature,1))
    A = np.transpose(X) - x_mu 
    
    X_red = basis@A 
    return X_red

def ReportAccuracy(Confusion):
    #obtain useful information from the confusion matrix
    #right now reports accuracy, but can report anything as TP, TN, FN, FP are calculated first
    dim = len(Confusion)
    TPstore = np.zeros((dim,1))
    FNstore = np.zeros((dim,1))
    FPstore = np.zeros((dim,1))
    TNstore = np.zeros((dim,1))
    AccurStore = np.zeros((dim,1))

    for i in range(dim):
        TPstore[i] = Confusion[i,i]
        
        beforei_R = Confusion[i,0:i]
        afteri_R = Confusion[i,i+1:dim]
        beforei_C = Confusion[0:i,i]
        afteri_C = Confusion[i+1:dim,i]
        temp = np.delete(Confusion, obj = i, axis=0)
        AllXcept = np.delete(temp, obj = i , axis=1)
        
        FNstore[i] = np.sum(beforei_R) + np.sum(afteri_R)
        FPstore[i] = np.sum(beforei_C) + np.sum(afteri_C)
        TNstore[i] = np.sum(AllXcept)
        
        AccurStore[i] = (TPstore[i] + TNstore[i])/(TPstore[i] + TNstore[i] + FPstore[i] + FNstore[i])
        
    return AccurStore



#prepare data
trainALL = np.vstack((train0,train1,train2,train3,train4,train5,train6,train7,train8,train9))
Ltr_all = len(trainALL)

testALL = np.vstack((test0,test1,test2,test3,test4,test5,test6,test7,test8,test9))
Lte_all = len(testALL)

ovrALL = np.vstack((train0Vall, train1Vall, train2Vall, train3Vall, train4Vall, train5Vall, train6Vall, train7Vall, train8Vall, train9Vall))

#gives us the basis for dimensionality reduction

[b90, b95] = PCA(trainALL)
'''
#keep basis(eigen vectors) from training data, need to project testing to lower dimensions with the basis

#dimensionality reduction (training and testing)

testALL_90 = np.transpose(DimRed(testALL, b90))
testALL_95 = np.transpose(DimRed(testALL, b95))

testALL_90 = np.transpose(DimRed(testALL, b90))
testALL_95 = np.transpose(DimRed(testALL, b95))

ovrALL_90 = np.transpose(DimRed(ovrALL, b90))
ovrALL_95 = np.transpose(DimRed(ovrALL, b95))
'''
#train support vectors for all cases
#clf = SVC()
#clf.fit(train0Vall, np.ravel(y))
#clf.predict([test0[0,:]])
#print("aa")


kernel_ref = ['linear', 'linear', 'linear', 'linear', 'rbf', 'rbf', 'rbf', 'rbf']
C_ref = [0.1, 1, 10, 100, 0.1, 1, 10, 100]
comboList = np.vstack((kernel_ref, C_ref))

#loop for no dimensionality reduction


Confusion = np.zeros((10,10))
AccuracyALL = np.zeros((len(kernel_ref), 10))
classes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
for i_out in range(len(kernel_ref)):
    print("using kernel and C:")
    kernel_ = comboList[0,i_out]
    print(kernel_)
    C_ = np.float32(comboList[1,i_out])
    print(C_)
    clf0 = SVC(C = C_, kernel = kernel_, decision_function_shape='ovr')
    clf1 = SVC(C = C_, kernel = kernel_, decision_function_shape='ovr')
    clf2 = SVC(C = C_, kernel = kernel_, decision_function_shape='ovr')
    clf3 = SVC(C = C_, kernel = kernel_, decision_function_shape='ovr')
    clf4 = SVC(C = C_, kernel = kernel_, decision_function_shape='ovr')
    clf5 = SVC(C = C_, kernel = kernel_, decision_function_shape='ovr')
    clf6 = SVC(C = C_, kernel = kernel_, decision_function_shape='ovr')
    clf7 = SVC(C = C_, kernel = kernel_, decision_function_shape='ovr')
    clf8 = SVC(C = C_, kernel = kernel_, decision_function_shape='ovr')
    clf9 = SVC(C = C_, kernel = kernel_, decision_function_shape='ovr')
    
    #train all support vectors (no reduced dimensions)
    clf0.fit(train0Vall, np.ravel(y))
    clf1.fit(train1Vall, np.ravel(y))
    clf2.fit(train2Vall, np.ravel(y))
    clf3.fit(train3Vall, np.ravel(y))
    clf4.fit(train4Vall, np.ravel(y))
    clf5.fit(train5Vall, np.ravel(y))
    clf6.fit(train6Vall, np.ravel(y))
    clf7.fit(train7Vall, np.ravel(y))
    clf8.fit(train8Vall, np.ravel(y))
    clf9.fit(train9Vall, np.ravel(y))
    
    begin = 0 #for sorting through testing data
    end = keep

    for i in range(10): #test with each class of data
        testCurr = testALL[begin:end,:] 
        begin = end
        end = begin + keep
        [NumSample, NumFeature] = np.shape(testCurr)
        
        ConfusionTally = np.zeros((1,10))
        
        for j in range(NumSample):
            pred = np.zeros((1,10))
            
            pred[:,0] = clf0.predict([testCurr[j,:]])
            pred[:,1] = clf1.predict([testCurr[j,:]])
            pred[:,2] = clf2.predict([testCurr[j,:]])
            pred[:,3] = clf3.predict([testCurr[j,:]])
            pred[:,4] = clf4.predict([testCurr[j,:]])
            pred[:,5] = clf5.predict([testCurr[j,:]])
            pred[:,6] = clf6.predict([testCurr[j,:]])
            pred[:,7] = clf7.predict([testCurr[j,:]])
            pred[:,8] = clf8.predict([testCurr[j,:]])
            pred[:,9] = clf9.predict([testCurr[j,:]])
            
            maximum = max(pred[0])
            MaxInd = np.where(pred[0] == maximum)
            
            if len(MaxInd[0]) > 1:
                MaxInd = np.random.choice(MaxInd[0])
            else:
                MaxInd = MaxInd[0]
            
            
            ConfusionTally[:,MaxInd] = ConfusionTally[:,MaxInd] + 1
            
        Confusion[i,:] = ConfusionTally #I'm not sure if this is how the confusion matrix is made....   
        
    #save for plotting
    dat_Confusion = pd.DataFrame(Confusion, index = classes, columns = classes)
    plt.figure(figsize = (10,7))
    plt.title("Confusion matrix for Kernel = " + kernel_ + " C=" + str(C_) + " Unreduced Data")
    cfm_plot = sn.heatmap(dat_Confusion, annot=True)
    
    #calculate and store accuracies
    Accuracy = np.transpose(ReportAccuracy(Confusion))
    AccuracyALL[i_out,:] = Accuracy
    
    
    print("test")
    
Accuracy_mean= np.mean(AccuracyALL, axis=1)         

        
#Find basis for data reduction and apply to training and testing data
[b90, b95] = PCA(trainALL)

#reduce dimensionality
testALL_90 = np.transpose(DimRed(testALL, b90))
testALL_95 = np.transpose(DimRed(testALL, b95))

train0Vall_90 = np.transpose(DimRed(train0Vall, b90))
train0Vall_95 = np.transpose(DimRed(train0Vall, b95))

train1Vall_90 = np.transpose(DimRed(train1Vall, b90))
train1Vall_95 = np.transpose(DimRed(train1Vall, b95))

train2Vall_90 = np.transpose(DimRed(train2Vall, b90))
train2Vall_95 = np.transpose(DimRed(train2Vall, b95))

train3Vall_90 = np.transpose(DimRed(train3Vall, b90))
train3Vall_95 = np.transpose(DimRed(train3Vall, b95))

train4Vall_90 = np.transpose(DimRed(train4Vall, b90))
train4Vall_95 = np.transpose(DimRed(train4Vall, b95))

train5Vall_90 = np.transpose(DimRed(train5Vall, b90))
train5Vall_95 = np.transpose(DimRed(train5Vall, b95))

train6Vall_90 = np.transpose(DimRed(train6Vall, b90))
train6Vall_95 = np.transpose(DimRed(train6Vall, b95))

train7Vall_90 = np.transpose(DimRed(train7Vall, b90))
train7Vall_95 = np.transpose(DimRed(train7Vall, b95))

train8Vall_90 = np.transpose(DimRed(train8Vall, b90))
train8Vall_95 = np.transpose(DimRed(train8Vall, b95))

train9Vall_90 = np.transpose(DimRed(train9Vall, b90))
train9Vall_95 = np.transpose(DimRed(train9Vall, b95))


Confusion_90 = np.zeros((10,10))
Confusion_95 = np.zeros((10,10))
AccuracyALL_90 = np.zeros((len(kernel_ref), 10))
AccuracyALL_95 = np.zeros((len(kernel_ref), 10))
classes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
for i_out in range(len(kernel_ref)):
    print("using kernel and C:")
    kernel_ = comboList[0,i_out]
    print(kernel_)
    C_ = np.float32(comboList[1,i_out])
    print(C_)
    #initialize (??? whatever this is) support vectors
    clf0_90 = SVC(C = C_, kernel = kernel_, decision_function_shape='ovr')
    clf0_95 = SVC(C = C_, kernel = kernel_, decision_function_shape='ovr')
    
    clf1_90 = SVC(C = C_, kernel = kernel_, decision_function_shape='ovr')
    clf1_95 = SVC(C = C_, kernel = kernel_, decision_function_shape='ovr')
    
    clf2_90 = SVC(C = C_, kernel = kernel_, decision_function_shape='ovr')
    clf2_95 = SVC(C = C_, kernel = kernel_, decision_function_shape='ovr')
    
    clf3_90 = SVC(C = C_, kernel = kernel_, decision_function_shape='ovr')
    clf3_95 = SVC(C = C_, kernel = kernel_, decision_function_shape='ovr')
    
    clf4_90 = SVC(C = C_, kernel = kernel_, decision_function_shape='ovr')
    clf4_95 = SVC(C = C_, kernel = kernel_, decision_function_shape='ovr')
    
    clf5_90 = SVC(C = C_, kernel = kernel_, decision_function_shape='ovr')
    clf5_95 = SVC(C = C_, kernel = kernel_, decision_function_shape='ovr')
    
    clf6_90 = SVC(C = C_, kernel = kernel_, decision_function_shape='ovr')
    clf6_95 = SVC(C = C_, kernel = kernel_, decision_function_shape='ovr')
    
    clf7_90 = SVC(C = C_, kernel = kernel_, decision_function_shape='ovr')
    clf7_95 = SVC(C = C_, kernel = kernel_, decision_function_shape='ovr')
    
    clf8_90 = SVC(C = C_, kernel = kernel_, decision_function_shape='ovr')
    clf8_95 = SVC(C = C_, kernel = kernel_, decision_function_shape='ovr')
    
    clf9_90 = SVC(C = C_, kernel = kernel_, decision_function_shape='ovr')
    clf9_95 = SVC(C = C_, kernel = kernel_, decision_function_shape='ovr')
    
    #train all support vectors (90 and 95 reduced dimensions)
    clf0_90.fit(train0Vall_90, np.ravel(y))
    clf0_95.fit(train0Vall_95, np.ravel(y))
    
    clf1_90.fit(train1Vall_90, np.ravel(y))
    clf1_95.fit(train1Vall_95, np.ravel(y))
    
    clf2_90.fit(train2Vall_90, np.ravel(y))
    clf2_95.fit(train2Vall_95, np.ravel(y))
    
    clf3_90.fit(train3Vall_90, np.ravel(y))
    clf3_95.fit(train3Vall_95, np.ravel(y))
    
    clf4_90.fit(train4Vall_90, np.ravel(y))
    clf4_95.fit(train4Vall_95, np.ravel(y))
    
    clf5_90.fit(train5Vall_90, np.ravel(y))
    clf5_95.fit(train5Vall_95, np.ravel(y))
    
    clf6_90.fit(train6Vall_90, np.ravel(y))
    clf6_95.fit(train6Vall_95, np.ravel(y))
    
    clf7_90.fit(train7Vall_90, np.ravel(y))
    clf7_95.fit(train7Vall_95, np.ravel(y))
    
    clf8_90.fit(train8Vall_90, np.ravel(y))
    clf8_95.fit(train8Vall_95, np.ravel(y))
    
    clf9_90.fit(train9Vall_90, np.ravel(y))
    clf9_95.fit(train9Vall_95, np.ravel(y))
    
    begin = 0 #for sorting through testing data
    end = keep

    for i in range(10): #test with each class of data
        testCurr_90 = testALL_90[begin:end,:] 
        testCurr_95 = testALL_95[begin:end,:] 
        begin = end
        end = begin + keep
        [NumSample, NumFeature] = np.shape(testCurr_90)
        
        ConfusionTally_90 = np.zeros((1,10))
        ConfusionTally_95 = np.zeros((1,10))
        
        for j in range(NumSample):
            pred_90 = np.zeros((1,10))
            pred_95 = np.zeros((1,10))
            
            pred_90[:,0] = clf0_90.predict([testCurr_90[j,:]])
            pred_90[:,1] = clf1_90.predict([testCurr_90[j,:]])
            pred_90[:,2] = clf2_90.predict([testCurr_90[j,:]])
            pred_90[:,3] = clf3_90.predict([testCurr_90[j,:]])
            pred_90[:,4] = clf4_90.predict([testCurr_90[j,:]])
            pred_90[:,5] = clf5_90.predict([testCurr_90[j,:]])
            pred_90[:,6] = clf6_90.predict([testCurr_90[j,:]])
            pred_90[:,7] = clf7_90.predict([testCurr_90[j,:]])
            pred_90[:,8] = clf8_90.predict([testCurr_90[j,:]])
            pred_90[:,9] = clf9_90.predict([testCurr_90[j,:]])
            
            pred_95[:,0] = clf0_95.predict([testCurr_95[j,:]])
            pred_95[:,1] = clf1_95.predict([testCurr_95[j,:]])
            pred_95[:,2] = clf2_95.predict([testCurr_95[j,:]])
            pred_95[:,3] = clf3_95.predict([testCurr_95[j,:]])
            pred_95[:,4] = clf4_95.predict([testCurr_95[j,:]])
            pred_95[:,5] = clf5_95.predict([testCurr_95[j,:]])
            pred_95[:,6] = clf6_95.predict([testCurr_95[j,:]])
            pred_95[:,7] = clf7_95.predict([testCurr_95[j,:]])
            pred_95[:,8] = clf8_95.predict([testCurr_95[j,:]])
            pred_95[:,9] = clf9_95.predict([testCurr_95[j,:]])
            
            maximum_90 = max(pred_90[0])
            MaxInd_90 = np.where(pred_90[0] == maximum_90)
            if len(MaxInd_90[0]) > 1:
                MaxInd_90 = np.random.choice(MaxInd_90[0])
            else:
                MaxInd_90 = MaxInd_90[0]
                
            maximum_95 = max(pred_95[0])
            MaxInd_95 = np.where(pred_95[0] == maximum_95)
            if len(MaxInd_95[0]) > 1:
                MaxInd_95 = np.random.choice(MaxInd_95[0])
            else:
                MaxInd_95 = MaxInd_95[0]
            
            
            ConfusionTally_90[:,MaxInd_90] = ConfusionTally_90[:,MaxInd_90] + 1
            ConfusionTally_95[:,MaxInd_95] = ConfusionTally_95[:,MaxInd_95] + 1
            
        Confusion_90[i,:] = ConfusionTally_90   
        Confusion_95[i,:] = ConfusionTally_95 
        
    #save for plotting
    dat_Confusion_90 = pd.DataFrame(Confusion_90, index = classes, columns = classes)
    dat_Confusion_95 = pd.DataFrame(Confusion_95, index = classes, columns = classes)
    
    plt.figure(figsize = (10,7))
    plt.title("Confusion matrix for Kernel = " + kernel_ + " C=" + str(C_) + " 90% Reduced Data")
    cfm_plot = sn.heatmap(dat_Confusion_90, annot=True)
    
    plt.figure(figsize = (10,7))
    plt.title("Confusion matrix for Kernel = " + kernel_ + " C=" + str(C_) + " 95% Reduced Data")
    cfm_plot = sn.heatmap(dat_Confusion_95, annot=True)
    
    #calculate and store accuracies
    Accuracy_90 = np.transpose(ReportAccuracy(Confusion_90))
    AccuracyALL_90[i_out,:] = Accuracy_90
    
    Accuracy_95 = np.transpose(ReportAccuracy(Confusion_95))
    AccuracyALL_95[i_out,:] = Accuracy_95
    
    
    print("test")
    
#first 4: linear kernel, second 4: RBF kernel    
Accuracy_mean_90 = np.mean(AccuracyALL_90, axis=1)  
Accuracy_mean_95 = np.mean(AccuracyALL_95, axis=1)  

#plot accuracy against C.
barWidth = 0.25 
br1 = np.arange(4)
br2 = [x + barWidth for x in br1]
br3 = [x + barWidth for x in br2]
br4 = [x + barWidth for x in br3]

plt.figure(figsize = (12,8))
plt.title("Comparison of Kernel Selection and C Value - Linear Kernel")
plt.bar(br1, Accuracy_mean[0:4], color = 'r', width = barWidth, edgecolor = 'grey', label = 'Unreduced')
plt.bar(br2, Accuracy_mean_90[0:4], color = 'b', width = barWidth, edgecolor = 'grey', label = '90% reduced')
plt.bar(br3, Accuracy_mean_95[0:4], color = 'g', width = barWidth, edgecolor = 'grey', label = '95% reduced')
plt.xlabel('C', fontweight = 'bold', fontsize = 15)
plt.ylabel('Average Classification Accuracy for All Classes')
plt.xticks([r + barWidth for r in range(4)], ['0.1', '1.0', '10', '100'])
plt.legend()

plt.figure(figsize = (12,8))
plt.title("Comparison of Kernel Selection and C Value - RBF Kernel")
plt.bar(br1, Accuracy_mean[4:8], color = 'r', width = barWidth, edgecolor = 'grey', label = 'Unreduced')
plt.bar(br2, Accuracy_mean_90[4:8], color = 'b', width = barWidth, edgecolor = 'grey', label = '90% reduced')
plt.bar(br3, Accuracy_mean_95[4:8], color = 'g', width = barWidth, edgecolor = 'grey', label = '95% reduced')
plt.xlabel('C', fontweight = 'bold', fontsize = 15)
plt.ylabel('Average Classification Accuracy for All Classes')
plt.xticks([r + barWidth for r in range(4)], ['0.1', '1.0', '10', '100'])
plt.legend()

'''
plt.plot(np.log10(C_ref[0:4]), Accuracy_mean[0:4], color = "r", linestyle = "solid", label = "Full data, Linear Kernel")
plt.scatter(np.log10(C_ref[0:4]), Accuracy_mean[0:4], color = "r")
plt.plot(np.log10(C_ref[0:4]), Accuracy_mean_90[0:4], color = "r", linestyle = "dashdot", label = "90% data, Linear Kernel")
plt.scatter(np.log10(C_ref[0:4]), Accuracy_mean_90[0:4], color = "r")
plt.plot(np.log10(C_ref[0:4]), Accuracy_mean_95[0:4], color = "r", linestyle = "dotted", label = "95% data, Linear Kernel")
plt.scatter(np.log10(C_ref[0:4]), Accuracy_mean_95[0:4], color = "r")

plt.plot(np.log10(C_ref[0:4]), Accuracy_mean[4:8], color = "b", linestyle = "solid", label = "Full data, RBF Kernel")
plt.scatter(np.log10(C_ref[0:4]), Accuracy_mean[4:8], color = "b")
plt.plot(np.log10(C_ref[0:4]), Accuracy_mean_90[4:8], color = "b", linestyle = "dashdot", label = "90% data, RBF Kernel")
plt.scatter(np.log10(C_ref[0:4]), Accuracy_mean_90[4:8], color = "b")
plt.plot(np.log10(C_ref[0:4]), Accuracy_mean_95[4:8], color = "b", linestyle = "dotted", label = "95% data, RBF Kernel")
plt.scatter(np.log10(C_ref[0:4]), Accuracy_mean_95[4:8], color = "b")

plt.xlabel("log10(C)")
plt.ylabel("Average Classification Accuracy for All Classes")
plt.legend()
'''

