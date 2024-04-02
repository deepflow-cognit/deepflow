# `deepflow package v1.6`
deepflow is cognit's main neural networking package made for layers,activations,optimizers and sequential containers

> [!IMPORTANT]
> Cognit has not been published to pip or any other packager, so its best
> if you use git to clone the repo and use it in your current directory


```
import deepflow

mnist = deepflow.dataset.mnist
(x_train,y_train) = mnist.load()

model = deepflow.sequential([
    
    deepflow.layers.flatten(input_shape=(28,28)),
    deepflow.layers.dense(12,activation="relu"),
    deepflow.layers.dropout(rate=0.2,input_shape=(2,1)),
    deepflow.layers.dense(10,1,activation="softmax")
])

model.train_data(optimiser="adam",X=x_train,y=y_train,layers_=model,loss_calc="sce",epochs=100)
```
 

