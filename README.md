# sam-tensorflow
A Tensorflow implement of "Predicting Human Eye Fixations via an LSTM-based Saliency Attentive Model"



## Training
```
1. cd sam-tensorflow
2. CUDA_VISIABLE_DEVICES=0 python main.py train -d "data_name" -p "data_path"
```

## Testing
```
1. cd sam-tensorflow
2. CUDA_VISIABLE_DEVICES=0 python main.py test -d "data_name" -p "data_path"

```

