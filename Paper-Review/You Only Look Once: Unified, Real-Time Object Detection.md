# You Only Look Once: Unified, Real-Time Object Detection.

[You Only Look Once: Unified, Real-Time Object Detection](https://arxiv.org/pdf/1506.02640v5.pdf)

## Abstract.

- 기존 Multi-task 문제를 하나의 회귀 문제로 바꾸었다.
- 이미지 전체를 Single Network으로 한 번 계산하여 Bounding Box와 Class Probability를 구한다.
- 모든 Pipeline이 Single Network로 이루어져 있는 End-to-end 방식이다.
- YOLO는 45 FPS, Fast YOLO는 155 FPS로 빠르다.

```
YOLO = One Stage Method
One Stage Method는 Single Network로 끝난다.
YOLO는 Multi-task 문제도 하나의 회귀로 해결했다.

R-CNNs = Two Stage Method
Two Stage Method는 Region Proposal 단계와 Detector 단계로 구분된다.
Detector 단계는 Multi-task 문제로, Localization을 위해 Regression을, Classification을 위해 SVM 등을 거친다.
```

```
YOLO는 Real Time Detection이 가능하다.
```

## 1. Introduction.

1. 기존 모델은 Classifer를 Detector로 바꾸어 사용하는데, 단점이 있다. e.g. DPM, R-CNN
    1. DPN은 Sliding Window 방식을 사용한다.
    2. R-CNN은 Region Proposal 방식을 사용한다.

        Bounding Box를 Classification, Regression, Post-processing 하는 등 복잡하다.

        느리고, 각 단계가 독립적으로 훈련되므로 최적화가 어렵다.

2. 따라서 Object Detection을 하나의 회귀 문제로 보고 YOLO가 나오게 되었다.

    ![YOLO 1](https://user-images.githubusercontent.com/66259854/130081307-1905c2c2-ba06-4032-93a7-1c0abbc02780.png)

    YOLO는 Single Network가 Bounding Box와 Class Probability을 계산하여 여러 장점이 있다.

    1. 매우 빠르다.

        Titan X GPU에서 Batch Processing 없이 45 FPS를 처리하고, Fast YOLO는 155 FPS를 처리한다.

        이는 동영상을 25ms 이하의 지연시간으로 실시간 처리가 가능한 정도이다.

    1. 예측 때 이미지 전체를 본다.

        YOLO는 이미지 전체를 보기 때문에 주변 정보를 처리할 수 있다.

        R-CNN은 주변 정보는 처리가 불가능하여 배경의 반점이나 노이즈를 물체로 인식하는 Background Error가 발생할 수 있다.

        YOLO는 R-CNN에 비해 Background Error가 절반 수준이다.

    2. 물체의 일반적인 부분을 학습한다.

        자연 이미지를 학습하여 그림 이미지로 테스트하면, DPM이나 R-CNN보다 뛰어난 성능을 보인다.

        독특한 부분보다 일반적인 부분을 학습하여 자연이나 그림에서 교집합이 많은 것 같다.

        따라서 훈련에서 보지 못한 새로운 이미지에 강하다.

    3. 마지막은 단점으로, 정확도가 조금 떨어진다.

        보통 속도와 정확성은 Trade-off 관계에 있기 때문에, 기존 모델에 비해 빠르지만 정확도가 다소 낮다.

## 2. Unified Detection.

![YOLO 2](https://user-images.githubusercontent.com/66259854/130081313-29b8d22d-911b-4385-8b8d-f2236a524c52.png)

1. YOLO는 입력 이미지를 S X S Grid로 나눈다.
2. 각각의 Grid Cell은 B개의 Bounding Box, Bounding Box에 대한 Confidence Score를 예측한다.

    Confidence Score는 $\text{Pr}(\text{Object}) * \text{IOU}^{\text{truth}}_{\text{pred}}$이다.

3. 각 Bounding Box는 5개 예측치로 구성된다.

    (x, y(는 Bounding Box의 중심이 Cell 내에서 갖는 상대 좌표이다. (0, 0)에서 (1, 1)의 값을 갖는다.

    (w, h)는 이미지의 너비, 높이가 1일 때, 그에 대한 Bounding Box의 상대 너비, 상대 높이이다. (0, 0)에서 (1, 1)의 값을 갖는다.

    confidence는 Confidence Score이다.

4. 각각의 Grid Cell은 Conditionall Class Probalilties, C를 예측한다.

    C는  $\text{Pr}(\text{Class}_i|\text{Object})$이다.

    B개의 Bounding Box와는 무관하게 하나의 Grid Cell은 하나의 Class만 예측한다.

5. Test에서 Confidence Score와 C를 곱하여 Class-specific Confidence Score를 구한다.

    $$\text{Pr}(\text{Class}_i | \text{Object}) * \text{Pr}(\text{Object}) * \text{IOU}^{\text{truth}}_{\text{pred}}$$
    
    $$= \text{Pr}(\text{Class}_i) * \text{IOU}^{\text{truth}}_{\text{pred}}$$

6. Dataset은 PASCAL VOC를 사용했다.

    S=7, B=2, C=20이고, 최종 예측 텐서 Demension은 (7 X 7 X (2  * 5 + 20))이다. 

### 2.1. Network Design.

![YOLO 3](https://user-images.githubusercontent.com/66259854/130081317-d429f641-9abf-43cb-b688-c0edb9b417cb.png)

![YOLO 0](https://user-images.githubusercontent.com/66259854/130081289-6658802a-86e0-47bb-862d-baf3002c4445.png)

- YOLO의 신경망은 GoogLeNet에서 가져왔다.
- 24개의 Conv Layer와 2개의 FC Layer로 구성되어 있다.
- GoogLeNet의 인셉션 구조 대신, 1 X 1 Convoluion 이후 3 X 3 Convoluion을 사용한다.
- Fast YOLO는 9개의 Conv Layer를 사용한다.

### 2.2. Training.
1. Pretrain.
    - 1,000개의 Class인 ImageNet Dataset으로 Conv 계층을 Pretrain 하였다.
    - 20개의 Conv Layer를 사용하고 뒤에 FC Layer를 연결했다.
    - 1주간 훈련하여 ImageNet 2012 검증 데이터셋에서 88%의 정확도가 나왔다.

2. YOLO Neural Network.
    - Darknet Framework를 사용했다.
    - Pretrain 된 모델을 Object Detection에 맞게 바꾼다.
    - 20개의 Conv Layer를 24개로 늘리고, FC Layer도 2개로 늘린다. 가중치는 임의로 초기화한다.
    - 224 X 224의 해상도를 448 X 448로 늘렸다.
    - 마지막에는 Linear Activaion Function을 적용하고, 나머지는 Leaky ReLU를 사용한다. $\phi(x) = \begin{cases} x, \qquad  \ \text{if} \ x>0 \\ 0.1x, \quad\text{otherwise} \end{cases}$

3. Loss Function.

    $$\lambda_{coord} \sum^{S^2}_{i=0} \sum^{B}_{j=0} 1^{obj}_{ij} [(x_i - \hat{x}_i)^2 + (y_i - \hat{y}_i)^2] \\ + \lambda_{coord} \sum^{S^2}_{i=0} \sum^{B}_{j=0} 1^{obj}_{ij} [(\sqrt{w_i} - \sqrt{\hat{w}_i})^2 + (\sqrt{h_i} - \sqrt{\hat{h}_i})^2] \\ + \sum^{S^2}_{i=0} \sum^{B}_{j=0} 1^{obj}_{ij} (C_i - \hat{C}_i)^2 \\ + \lambda_{noobj} \sum^{S^2}_{i=0} \sum^{B}_{j=0} 1^{noobj}_{ij} (C_i - \hat{C}_i)^2 \\ + \sum^{S^2}_{i=0} 1^{obj}_{i} \sum_{c \, \in \, classes} (p_i(c) - \hat{p}_i(c))^2$$

    - Loss는 SSE(Sum-squared Error)를 사용한다.
    - SSE는 최적화가 쉬워 좋지만, Localization / Classification Loss의 가중치가 같은 단점이 있다.
    - 대부분의 Grid Cell에는 객체가 없기 때문에, 객체가 존재하는 Bounding Box 좌표에 대한 Loss 가중치를 높였다. $\lambda_{coord} = 5, \, \lambda_{noobj} = 0.5$
    - SSE는 큰 Bounding Box와 작은 Bounding Box에 같은 가중치를 두는데, 작은 Bounding Box는 위치 변화에 더 민감하기 때문에, (w, h)에 Square Root를 취한다.

4. Setting & Parameter.
    - VOC 2007, 2012 훈련 및 검증 데이터셋
    - 135 Epochs
    - Batch Size: 64
    - Momentum: 0.9
    - Decay: 0.0005
    - Learning Rate: 0.001 → 0.01, 75 Epoch: 0.01 / 30 Epoch: 0.001 / 30 Epoch: 0.0001
    - Dropout: 0.5
    - Data Augmentation: 20% Random Scaling, Random Translation

### 2.3. Inference.
- Test는 PASCAL VOC로 진행하고, 한 이미지에 98개의 Bounding Box가 나온다.
- YOLO의 Grid Design은 한 객체가 여러 Cell에서 검출되는 다중 검출 문제가 발생할 수 있다.

    따라서 Non-maximal Suppression으로 해결하고, mAP가 2~3% 증가하였다.

    #### 번외. Non-maximal Suppression.

    Object Detection 과정에서 다중 검출 문제를 해결하기 위해 사용된다.

    1. 기준치 e.g. 0.6 이하의 Bounding Box를 모두 제거한다.
    2. 가장 큰 확률의 Bounding Box를 선택한다.
    3. 선택된 Bounding Box와 IoU가 0.5 이상인 Box를 제거한다.
    4. 한 객체에 한 Bounding Box, 한 Cell만 남는다.

    [gaussian37 : 네이버 블로그](https://blog.naver.com/infoefficien/221229808532)

### 2.4. Limitations of YOLO.
- 한 Grid Cell은 한 객체만 검출하기 때문에, 한 Grid Cell에 2개 이상의 객체가 있다면 검출이 어려운 공간적 제약이 있다.
- 학습하지 못한 종횡비(Aspect Ratio)에 약하다.
- 큰 Bounding Box와 작은 Bounding Box에 같은 가중치를 두기 때문에 부정확한 Localization이 발생할 수 있다.

## 3. Comparison to Other Detection Systems.

1. Deformable Parts Models(DPM)
2. R-CNN

## 4. Experiments.

### 4.1. Comparison to Other Real-Time Systems.

![YOLO 4](https://user-images.githubusercontent.com/66259854/130081319-c60339c8-dbb4-43de-ba4d-1680f722337f.png)

1. Real Time Detectors에서 Fast YOLO, YOLO가 mAP, FPS 면에서 가장 성능이 좋다.
2. Less Than Real-Time에서 VGG-16으로 훈련시킨 YOLO는 mAP가 적당한 수준이지만 FPS 처리가 좋다.

### 4.2. VOC 2007 Error Analysis.

![YOLO 5](https://user-images.githubusercontent.com/66259854/130081322-2b87a203-79a0-4c1c-9cc8-9dc0cd9b57d6.png)

1. Object Detection이 잘 되었는지, 어떤 Error인지 확인한다.

    Diagnosing Error in Object Detectors 논문에 소개된 방법을 사용하였다.

    1. Correct : Class가 정확하고, IOU > 0.5
    2. Localization : Class가 정확하고, 0.1 < IOU < 0.5
    3. Similar : Class가 비슷하고, IOU > 0.1
    4. Other : Class가 틀렸지만, IOU > 0.1
    5. Background : 모든 Object에 대해 IOU < 0.1

2. YOLO는 Localization Error가 19.0%로 가장 크다.

    SSE를 사용하여 Localization / Classification Loss의 가중치가 같기 때문이다.

3. Fast R-CNN은 Background Error가 13.6%로 가장 크다.

    이미지 전체를 사용하지 않기 때문에 주변 정보를 처리할 수 없기 때문이다.

### 4.3. Combining Fast R-CNN and YOLO.

![YOLO 6](https://user-images.githubusercontent.com/66259854/130081324-01473e0d-b6f0-4c74-879a-5cd687a6ad82.png)

YOLO의 Background Error가 현저히 낮기 때문에, YOLO와 Fast R-CNN을 Ensemble 한다.

속도가 조금 느려졌지만, mAP는 Fast R-CNN 단독 기준으로 3.2% 올랐다.

### 4.4. VOC 2012 Results.

![YOLO 7](https://user-images.githubusercontent.com/66259854/130081327-e2a4f16c-fbcd-4f8f-abd7-52ed6587acad.png)

PASCAL VOC 2007이 아닌 2012 데이터셋에서 YOLO와 다른 Model을 돌린 결과이다.

### 4.5. Generalizability: Person Detection in Artwork.

![YOLO 8](https://user-images.githubusercontent.com/66259854/130081334-093c6e59-ebae-462b-8d87-5329b2721090.png)

훈련은 실제 이미지로, 테스트는 피카소 데이터셋과 일반 예술 작품을 사용했다.

YOLO가 53.3%로 가장 높은 정확도를 나타냈다.

## 5. Real-Time Detection In The Wild.

![YOLO 9](https://user-images.githubusercontent.com/66259854/130081337-d86c794a-035e-4a9d-94af-c3694443acc3.png)

## 6. Conclusion.

1. 장점.
    - Multi-task → One Regression.
    - One Stage Method.
    - End-to-End.
    - Real Time Detection.
    - Low Background Error.
    - Generalizability.
    - Fast.

1. 단점.
    - High Localization Error.
    - Low Accuracy.
    - Spatial Constraints.
    - Strange Aspect Ratio.
    - Inaccurate Localization.

## Link.

[논문 리뷰 - YOLO(You Only Look Once) 톺아보기](https://bkshin.tistory.com/entry/%EB%85%BC%EB%AC%B8-%EB%A6%AC%EB%B7%B0-YOLOYou-Only-Look-Once?category=1066362)
