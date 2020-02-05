## Input Data

X.shape: (samples, time_steps, features)  
y.shape: (samples, label)  

## Bidirectoinal LSTM

### fine tuning

K-fold cross validation
* batch_size
![](/images/lstmfinetunebatch.png)
* units of LSTM
![](/images/lstmfinetuneunits.png)
* ratio of dropout
![](/images/lstmfinetunedropout.png)

### predict test data

![](/images/c1.png)

![](/images/c2.png)

![](/images/c3.png)

### evaluation


