# SSUML Team Project 1. - Intel Image Classification.
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
https://www.notion.so/asollie/CS231n-LEC09-f8edf22ab0704e25b6c7bc64bd3300b2

https://pytorch.org/docs/stable/torchvision/index.html

https://blog.naver.com/another0430/222069431000

ResNet의 요구 사항에 맞게 Transform하는 부분을 살려 ResNet 모델을 선정.

ResNet34, ResNet152, Wide ResNet-101-2, ResNeXt-101-32x8d를 불러오면서 실험.

실제로 기존 모델을 Baseline으로 사용할 때 ResNet을 많이 사용한다고 한다.

### ResNet34.
1. Training & Valid Acc.
   
   ![image](https://user-images.githubusercontent.com/66259854/104050301-c3f9d900-5229-11eb-97f0-913c61686f4d.png)
   
   ![image](https://user-images.githubusercontent.com/66259854/104050310-c9efba00-5229-11eb-82b9-882d63b0ec80.png)

2. Test Acc.

   ![image](https://user-images.githubusercontent.com/66259854/104050332-d542e580-5229-11eb-9ccb-94239078e3be.png)

### ResNet152.
1. Training & Valid Acc.

   ![image](https://user-images.githubusercontent.com/66259854/104050345-db38c680-5229-11eb-89cf-7c1ae7e16728.png)

   ![image](https://user-images.githubusercontent.com/66259854/104050360-dffd7a80-5229-11eb-95cf-690244aad777.png)

2. Test Acc.

   ![image](https://user-images.githubusercontent.com/66259854/104050377-e4c22e80-5229-11eb-8239-5cab2a834731.png)

### Wide ResNet-101-2.
1. Training & Valid Acc.
   
   ![image](https://user-images.githubusercontent.com/66259854/104050389-ea1f7900-5229-11eb-8313-d2d76b840759.png)
   
   ![image](https://user-images.githubusercontent.com/66259854/104050406-ee4b9680-5229-11eb-9671-c8cbabc6037f.png)

2. Test Acc.
   
   ![image](https://user-images.githubusercontent.com/66259854/104050412-f277b400-5229-11eb-8786-07218cd7caa7.png)

### ResNeXt-101-32x8d.
1. Training & Valid Acc.
   
   ![image](https://user-images.githubusercontent.com/66259854/104050425-f7d4fe80-5229-11eb-9de5-ab3a0ff83984.png)
   
   ![image](https://user-images.githubusercontent.com/66259854/104050446-fc011c00-5229-11eb-8a43-923566afffa3.png)

2. Test Acc.
   
   ![image](https://user-images.githubusercontent.com/66259854/104050461-00c5d000-522a-11eb-8750-56f75db4fa79.png)

## Optimizer.
https://www.notion.so/asollie/CS231n-LEC07-444af169209a42398e91de92b1d4d2b2

https://pytorch.org/docs/stable/optim.html

https://wiserloner.tistory.com/1032

```Python
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
```

기존 Notebook은 SGD를 사용함.
1. Adam은 Training과 Valid의 격차가 더 늘어남.
2. AdamW도 Adam과 비슷한 성능을 보임.
3. Adam류 Optimizer를 사용해서 다시 한 번 돌려보자.

### :heavy_check_mark:
다시 한 번 Optimizer를 바꾸어가면서 실험한 내용 추가 예정.

## Learning Rate.
https://towardsdatascience.com/learning-rate-schedules-and-adaptive-learning-rate-methods-for-deep-learning-2c8f433990d1

https://www.deeplearningwizard.com/deep_learning/boosting_models_pytorch/lr_scheduling/

https://m.blog.naver.com/PostView.nhn?blogId=nostresss12&logNo=221544987534&proxyReferer=https:%2F%2Fwww.google.co.kr%2F

```Python
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
```

기존 Notebook의 LR은 0.001.
Learning Rate Scheduler를 사용함.

### Question.
Learning Rate Scheduler 북마크에서 SGD가 계속해서 나온다.

SGD를 계속해서 사용하는 이유가 SGD의 Mini-batch와 연관이 있을까?

## Project Notebook.
https://www.kaggle.com/asollie/intel-image-multiclass-pytorch-94-test-acc
