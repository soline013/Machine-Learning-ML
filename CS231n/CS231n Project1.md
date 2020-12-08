# CS231n Project1.
**Intel Image Classification - Image Scene Classification of Multiclass.**

## Project Guidance.
https://www.notion.so/1-CNN-image-classification-4829d58a80d14272b36fba5c7a87dc0a

## Project Goal.
  1. Test Accuracy 높이기.
  
     Notebook이 95%로 나와있지만, 실제 Test Score는 91% 정도가 나온다.

  2. Pytorch 코드 이해.
  
     Pytorch로 포팅하거나, Notebook을 돌려보면서 Pytorch 코드 이해하기.

  3. ML 전반의 이해도 향상.
  
     Model, Layer, Learning Rate, Optimizer, Batch Size, etc.

## Notebook 선정.
https://www.kaggle.com/fadilparves/intel-image-multiclass-pytorch-95-acc

Pytorch를 사용한 Notebook.

Torchvision을 Import하여 ResNet34 모델을 사용했다.

## Models.
https://pytorch.org/docs/stable/torchvision/index.html

https://blog.naver.com/another0430/222069431000

ResNet의 요구 사항에 맞게 Transform하는 부분을 살려 ResNet 모델을 선정.

ResNet34, ResNet152, Wide ResNet-101-2, ResNeXt-101-32x8d를 불러오면서 실험.

실제로 기존 모델을 Baseline으로 사용할 때 ResNet을 많이 사용한다고 한다.

### ResNet34.

### ResNet152.

### Wide ResNet-101-2.

### ResNeXt-101-32x8d.

### :heavy_check_mark:
*각 모델의 구조 추가 예정.*

*ResNet의 자세한 설명 추가 예정.*
