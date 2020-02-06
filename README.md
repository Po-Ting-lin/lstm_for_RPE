## Input Data

X.shape: (samples, time_steps, features)  
y.shape: (samples, label)  

## Bidirectional LSTM
```
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv1d_1 (Conv1D)            (None, 33, 12)            444       
_________________________________________________________________
bidirectional_1 (Bidirection (None, 40)                5280      
_________________________________________________________________
dropout_1 (Dropout)          (None, 40)                0         
_________________________________________________________________
dense_1 (Dense)              (None, 1)                 41        
=================================================================
Total params: 5,765
Trainable params: 5,765
Non-trainable params: 0
_________________________________________________________________
```

### fine tuning

K-fold cross validation to fine tune hyperparameters.

* batch_size  
![](/images/lstmfinetunebatch.png)  

* units of LSTM  
![](/images/lstmfinetuneunits.png)  

* ratio of dropout  
![](/images/lstmfinetunedropout.png)  

* initial learning rate  
![](/images/lrdemo.png)

* step-based learning rate decay  
![](/images/lrstep.png)

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



