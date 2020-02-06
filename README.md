## Input Data

X.shape: (samples, time_steps, features)  
y.shape: (samples, label)  

## Bidirectional LSTM

### fine tuning

K-fold cross validation to fine tune hyperparameters.

* batch_size  
![](/images/lstmfinetunebatch.png)  

* units of LSTM  
![](/images/lstmfinetuneunits.png)  

* ratio of dropout  
![](/images/lstmfinetunedropout.png)  

### evaluation

* learning curve  
![](/images/loss.png)   

* ROC curve  
![](/images/roc.png)   



### predict test data
* cell 1
![](/images/c1.png)  
* cell 2
![](/images/c2.png)  
* cell 3
![](/images/c3.png)  



