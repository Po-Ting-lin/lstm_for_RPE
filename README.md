## Input Data

X.shape: (samples, time_steps, features)  
y.shape: (samples, label)  

## Bidirectoinal LSTM

### fine tuning

K-fold cross validation to fine tune hyperparameter.

* batch_size  
![](/images/lstmfinetunebatch.png)  

* units of LSTM  
![](/images/lstmfinetuneunita.png)  

* ratio of dropout  
![](/images/lstmfinetunedropout.png)  


### predict test data
* cell 1
![](/images/c1.png)  
* cell 2
![](/images/c2.png)  
* cell 3
![](/images/c3.png)  

### evaluation

* predict test set 
TN  FP  FN  TP 
30   1   1  24 

Recall(Sensitivity):  0.96
Specificity:  0.968
Accuracy:  0.964
