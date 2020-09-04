# sam-tensorflow
A Tensorflow implement of "Predicting Human Eye Fixations via an LSTM-based Saliency Attentive Model"



## Training
```
1. Download the pretrained vgg networrk and put it in the ./sam-tensorflow/weights floder

2. cd sam-tensorflow
3. CUDA_VISIABLE_DEVICES=0 python main.py train -d "data_name" -p "data_path"
```

## Testing
```
cd sam-tensorflow
CUDA_VISIABLE_DEVICES=0 python main.py test -d "data_name" -p "data_path"

```

