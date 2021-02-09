# 이미지는 수정 예정.

# CS231n LEC11.
## Stanford University CS231n, Spring 2017.
**Detection and Segmentation.**

## Recall from last time.

![CS231n%20LEC11%203214911eacc34b3da98a52d1278dd5b9/f72252a4-e436-4c43-8346-3b3b77c91141.pdf-0011.png](CS231n%20LEC11%203214911eacc34b3da98a52d1278dd5b9/f72252a4-e436-4c43-8346-3b3b77c91141.pdf-0011.png)

![CS231n%20LEC11%203214911eacc34b3da98a52d1278dd5b9/f72252a4-e436-4c43-8346-3b3b77c91141.pdf-0012.png](CS231n%20LEC11%203214911eacc34b3da98a52d1278dd5b9/f72252a4-e436-4c43-8346-3b3b77c91141.pdf-0012.png)

![CS231n%20LEC11%203214911eacc34b3da98a52d1278dd5b9/f72252a4-e436-4c43-8346-3b3b77c91141.pdf-0013.png](CS231n%20LEC11%203214911eacc34b3da98a52d1278dd5b9/f72252a4-e436-4c43-8346-3b3b77c91141.pdf-0013.png)

![CS231n%20LEC11%203214911eacc34b3da98a52d1278dd5b9/f72252a4-e436-4c43-8346-3b3b77c91141.pdf-0014.png](CS231n%20LEC11%203214911eacc34b3da98a52d1278dd5b9/f72252a4-e436-4c43-8346-3b3b77c91141.pdf-0014.png)

## Today.

![CS231n%20LEC11%203214911eacc34b3da98a52d1278dd5b9/f72252a4-e436-4c43-8346-3b3b77c91141.pdf-0015.png](CS231n%20LEC11%203214911eacc34b3da98a52d1278dd5b9/f72252a4-e436-4c43-8346-3b3b77c91141.pdf-0015.png)

![CS231n%20LEC11%203214911eacc34b3da98a52d1278dd5b9/f72252a4-e436-4c43-8346-3b3b77c91141.pdf-0017.png](CS231n%20LEC11%203214911eacc34b3da98a52d1278dd5b9/f72252a4-e436-4c43-8346-3b3b77c91141.pdf-0017.png)

LEC11에서는 Computer Vision Tasks를 다룬다.

## Content.

## Semantic Segmentation.

![CS231n%20LEC11%203214911eacc34b3da98a52d1278dd5b9/f72252a4-e436-4c43-8346-3b3b77c91141.pdf-0019.png](CS231n%20LEC11%203214911eacc34b3da98a52d1278dd5b9/f72252a4-e436-4c43-8346-3b3b77c91141.pdf-0019.png)

입력: Image.

출력: Image의 모든 Pixel에 대해 카테고리 결정.

모든 픽셀에 대한 Cross-Entropy를 계산하면 Network 전체를 End-to-end로 학습할 수 있다.

Semantic Segmentation은 개별 객체 구분이 불가능하다. → Instance Segmentation에서 해결.

![CS231n%20LEC11%203214911eacc34b3da98a52d1278dd5b9/f72252a4-e436-4c43-8346-3b3b77c91141.pdf-0021.png](CS231n%20LEC11%203214911eacc34b3da98a52d1278dd5b9/f72252a4-e436-4c43-8346-3b3b77c91141.pdf-0021.png)

첫 번째로, Sliding Window가 있다.

입력 이미지를 작은 영역으로 쪼개고, 그 영역으로 CNN을 통해 Classification 한다.

비용이 과도하게 발생하고, Feature가 공유되지 않기 때문에 좋은 방법은 아니다.

![CS231n%20LEC11%203214911eacc34b3da98a52d1278dd5b9/f72252a4-e436-4c43-8346-3b3b77c91141.pdf-0023.png](CS231n%20LEC11%203214911eacc34b3da98a52d1278dd5b9/f72252a4-e436-4c43-8346-3b3b77c91141.pdf-0023.png)

두 번째로, Fully Convolutional을 이용한다.

Output Tensor: C X H X W.

출력에서 C는 카테고리 수이고, 모든 Pixel에 대해 Classification Scores를 매긴 결과이다.

![CS231n%20LEC11%203214911eacc34b3da98a52d1278dd5b9/f72252a4-e436-4c43-8346-3b3b77c91141.pdf-0044.png](CS231n%20LEC11%203214911eacc34b3da98a52d1278dd5b9/f72252a4-e436-4c43-8346-3b3b77c91141.pdf-0044.png)

위의 Conv Network는 Computer Vision Tasks에서는 비용이 너무 크기 때문에 적합하지 않다.

따라서 Max Pooling, Stride Convolution을 사용하여 Downsampling, Upsampling 한다.

Image Classification(FC-layer)과 다르게 출력의 크기가 입력과 같아진다. → 계산 효율이 좋다.

### Question.

Q: "Training Data를 어떻게 만드는지?"

입력 이미지의 모든 픽셀에 대해 Labeling이 필요하다.

일반적으로 비용이 상당히 크다.

Q: "손실 함수는 어떻게 디자인하는지?"

모든 픽셀을 Classification 하는 상황이다.

출력의 모든 픽셀과 Ground Truth 사이에 Cross Entropy를 계산한다.

이 값을 더하거나 평균을 내는 식으로 Loss를 계산한다.

### In-network Upsampling.

Downsampling은 Average Pooling, Max Pooling 등 이미 알고 있는 내용이다.

그렇다면 Upsampling은 어떻게 할까?

![CS231n%20LEC11%203214911eacc34b3da98a52d1278dd5b9/f72252a4-e436-4c43-8346-3b3b77c91141.pdf-0026.png](CS231n%20LEC11%203214911eacc34b3da98a52d1278dd5b9/f72252a4-e436-4c43-8346-3b3b77c91141.pdf-0026.png)

1. Nearest Neighbor.

    똑같은 값의 이웃을 만들어 Unpooling 한다.

2. Bad of Nails.

    주변에 0을 추가하여 Unpooling 한다.

    0으로 평평할 때 Non-zero Region만 바늘처럼 값이 튀기 때문에 이런 이름으로 불린다.

![CS231n%20LEC11%203214911eacc34b3da98a52d1278dd5b9/f72252a4-e436-4c43-8346-3b3b77c91141.pdf-0027.png](CS231n%20LEC11%203214911eacc34b3da98a52d1278dd5b9/f72252a4-e436-4c43-8346-3b3b77c91141.pdf-0027.png)

Max Unpooling은 Max Pooling과 연관된다.

Bed of Nails와 같이 0을 추가하지만, Max Pooling의 위치를 기억하여 Unpooling에 적용한다.

#### Question.

Q: "왜 Max Unpooling이 좋은 아이디어이고 어떤 점에서 중요한지?"

Max Pooling을 하게 되면 공간 정보를 잃어버리는데, 이 공간 정보를 조금이나마 유지할 수 있다.

### Learning Upsampling: Transpose Convolution.
1. Normal Convolution. Stride 2, Pad 1.

    ![CS231n%20LEC11%203214911eacc34b3da98a52d1278dd5b9/f72252a4-e436-4c43-8346-3b3b77c91141.pdf-0033.png](CS231n%20LEC11%203214911eacc34b3da98a52d1278dd5b9/f72252a4-e436-4c43-8346-3b3b77c91141.pdf-0033.png)

    Convolution 연산에서 Stride가 1이 아니라면, Downsampling 효과가 있다.

2. Transpose Convolution. Stride 2, Pad 1.

    ![CS231n%20LEC11%203214911eacc34b3da98a52d1278dd5b9/f72252a4-e436-4c43-8346-3b3b77c91141.pdf-0038.png](CS231n%20LEC11%203214911eacc34b3da98a52d1278dd5b9/f72252a4-e436-4c43-8346-3b3b77c91141.pdf-0038.png)

    ![CS231n%20LEC11%203214911eacc34b3da98a52d1278dd5b9/f72252a4-e436-4c43-8346-3b3b77c91141.pdf-0039.png](CS231n%20LEC11%203214911eacc34b3da98a52d1278dd5b9/f72252a4-e436-4c43-8346-3b3b77c91141.pdf-0039.png)

    Transpose Convolution은 연산 과정을 반대로하여 Upsampling 효과를 얻는다.

    1. 1개의 Scalar Pixel에 Filter Vector를 곱해 출력에 넣는다.
    2. 입력 값 Scalar Pixel은 가중치 역할을 한다.
    3. Stride로 움직여 Image Size를 키운다.
    4. 겹치는 부분은 모두 더한다.

    ![CS231n%20LEC11%203214911eacc34b3da98a52d1278dd5b9/f72252a4-e436-4c43-8346-3b3b77c91141.pdf-0041.png](CS231n%20LEC11%203214911eacc34b3da98a52d1278dd5b9/f72252a4-e436-4c43-8346-3b3b77c91141.pdf-0041.png)

    ![CS231n%20LEC11%203214911eacc34b3da98a52d1278dd5b9/f72252a4-e436-4c43-8346-3b3b77c91141.pdf-0043.png](CS231n%20LEC11%203214911eacc34b3da98a52d1278dd5b9/f72252a4-e436-4c43-8346-3b3b77c91141.pdf-0043.png)

    Convolution을 Matrix Multiplication으로 표현할 수 있다.

    Filter의 움직임 또한 표현이 가능하다.

    1. Stride 1 → 일반 Conv나 Transpose Conv나 큰 차이가 없다.
    2. Stride > 1 → Transpose Convolution은 차원을 확장하는 모습이 보이고, 근본적으로 다른 연산이 된다.

    #### Question.

    Q: "왜 평균을 내지 않고 더하는지?"

    Transpose Convolution의 수식 때문이다.

    Receptive field의 크기에 따라 Magnitudes이 달라지기 때문에 Sum은 문제가 될 수 있다.

    예시처럼 3X3 Stride 2를 사용하면 Checkerboard Artifacts이 발생하곤 한다.

    Q: "Stride 1/2 Convolution이라는 용어는 어떻게 생겨났는지?"

    강의 진행자의 논문에서 나온 용어이다.

    Input과 Output의 크기 비율이 1 : 2이기 때문이다.

## Classification + Localization.

![CS231n%20LEC11%203214911eacc34b3da98a52d1278dd5b9/f72252a4-e436-4c43-8346-3b3b77c91141.pdf-0048.png](CS231n%20LEC11%203214911eacc34b3da98a52d1278dd5b9/f72252a4-e436-4c43-8346-3b3b77c91141.pdf-0048.png)

![CS231n%20LEC11%203214911eacc34b3da98a52d1278dd5b9/f72252a4-e436-4c43-8346-3b3b77c91141.pdf-0049.png](CS231n%20LEC11%203214911eacc34b3da98a52d1278dd5b9/f72252a4-e436-4c43-8346-3b3b77c91141.pdf-0049.png)

기존 Image Classification과 구조가 비슷하다.

Object Classification과 Object Boundary를 동시에 진행한다.

1. Object Classification.

    Class Score를 출력으로 한다.

    Softmax Loss를 구한다.

2. Object Boundary.

    Bounding Box Coordinates를 출력으로 한다.

    Ground Truth Bbox와 예측한 Bbox 사이의 Loss를 측정한다.

    L2 Loss로 가장 쉽게 구할 수 있다.

Softmax Loss와 L2 Loss를 더하여 Multitask Loss를 구하고, Hyperparameter를 통해 두 Loss를 조절한다.

Fully Supervised Setting이기 때문에, 카테고리 Label와 Bounding Box GT를 미리 가지고 있어야 한다.

### Question.

Q: "왜 Classification과 Bbox Regression을 동시에 학습시키는 게 좋으며, 가령 오분류에 대해 Bbox가 있으면 어떻게 하는가?"

일반적으로 두 Loss의 학습은 문제가 없으며, 많은 사람들이 사용한다.

오분류는 Bbox를 하나만 예측하지 않고, 카테고리마다 하나를 예측한다.

Q: "두 개의 Loss 단위가 달라서 Gradient 계산에 문제가 생기진 않는지?"

두 Loss의 가중치를 조절하는 Hyperparameter를 사용한다.

이 Hyperparameter는 손실함수의 값 자체를 바꾼다.

지금까지의 Hyperparameter와는 다르고, 조절하는 것이 어렵다.

따라서 Loss 값으로 비교하지 않고 다른 성능 지표를 도입하여 조절한다.

Q: "앞 쪽의 큰 네트워크를 고정하고, 각 FC-layer만 학습시키는 방법은 어떤지?"

Fine Tune은 이런 문제에서 시도해볼 만한 방법이다.

실제로 많이 하는 트릭으로, 네트워크를 Freeze하고 두 FC-layer만 학습시킨다.

그리고 학습이 끝나면 다시 합치고 Fine Tune 한다.

[[Deep Learning] pre-training 과 fine-tuning (파인튜닝)](https://eehoeskrap.tistory.com/186)

---

![CS231n%20LEC11%203214911eacc34b3da98a52d1278dd5b9/f72252a4-e436-4c43-8346-3b3b77c91141.pdf-0050.png](CS231n%20LEC11%203214911eacc34b3da98a52d1278dd5b9/f72252a4-e436-4c43-8346-3b3b77c91141.pdf-0050.png)

![CS231n%20LEC11%203214911eacc34b3da98a52d1278dd5b9/f72252a4-e436-4c43-8346-3b3b77c91141.pdf-0052.png](CS231n%20LEC11%203214911eacc34b3da98a52d1278dd5b9/f72252a4-e436-4c43-8346-3b3b77c91141.pdf-0052.png)

Bbox와 같이 이미지 내의 위치를 예측하는 것은 다른 문제에도 적용할 수 있다.

사람 이미지가 입력으로 들어가면, 출력으로 각 관절의 좌표 값, 사람의 포즈를 예측한다.

예시의 Data Sets은 14개의 관절의 위치로 사람의 포즈를 정의한다.

예시처럼 L2 Loss를 사용하기도 하고, 다양한 Regression Losses를 적용할 수 있다.

### Question.

Q: "Regression Losses가 무엇인지?"

Cross Entropy나 Softmax가 아닌 Losses를 의미한다.

L2, L1, Smooth L1 등이 이에 해당한다.

(+) Classification과 Regression의 차이는 결과가 Categorical인지 Cintinuous인지이다.

Categorical 한 Class Score의 경우, Cross Entropy, Softmax, SVM, Margin Loss를 사용할 수 있다.

Cintinuous 한 (위의 예시처럼) 관절의 위치라면, 출력이 연속적이므로 L2, L1, Smooth L1를 사용할 수 있다.

## Object Detection.

![CS231n%20LEC11%203214911eacc34b3da98a52d1278dd5b9/f72252a4-e436-4c43-8346-3b3b77c91141.pdf-0054.png](CS231n%20LEC11%203214911eacc34b3da98a52d1278dd5b9/f72252a4-e436-4c43-8346-3b3b77c91141.pdf-0054.png)

Ross Girshick의 자료.

PASCAL VOC는 2012년까지 정체되다가, Deep Learning의 등장으로 성능이 빠르게 증가하였다.

![CS231n%20LEC11%203214911eacc34b3da98a52d1278dd5b9/f72252a4-e436-4c43-8346-3b3b77c91141.pdf-0056.png](CS231n%20LEC11%203214911eacc34b3da98a52d1278dd5b9/f72252a4-e436-4c43-8346-3b3b77c91141.pdf-0056.png)

Object Detection은 이미지마다 객체의 수가 다르다.

1. 상단 이미지는 객체가 하나이므로 4개의 숫자만 예측하면 된다.
2. 중간 이미지는 객체가 셋이므로 12개의 숫자가 필요하다.
3. 더 많은 객체의 경우는 매우 복잡하다. 어떻게 할 수 있을까.

![CS231n%20LEC11%203214911eacc34b3da98a52d1278dd5b9/f72252a4-e436-4c43-8346-3b3b77c91141.pdf-0058.png](CS231n%20LEC11%203214911eacc34b3da98a52d1278dd5b9/f72252a4-e436-4c43-8346-3b3b77c91141.pdf-0058.png)

![CS231n%20LEC11%203214911eacc34b3da98a52d1278dd5b9/f72252a4-e436-4c43-8346-3b3b77c91141.pdf-0061.png](CS231n%20LEC11%203214911eacc34b3da98a52d1278dd5b9/f72252a4-e436-4c43-8346-3b3b77c91141.pdf-0061.png)

첫 번째로, 앞선 Semantic Segmentation과 비슷한 방법이다.

입력 이미지를 작은 영역으로 나누어 CNN에 넣는다.

추출할 영역의 Size를 정하기 어려워 좋은 방법이 아니다. → Brute Force.

![CS231n%20LEC11%203214911eacc34b3da98a52d1278dd5b9/f72252a4-e436-4c43-8346-3b3b77c91141.pdf-0062.png](CS231n%20LEC11%203214911eacc34b3da98a52d1278dd5b9/f72252a4-e436-4c43-8346-3b3b77c91141.pdf-0062.png)

두 번째로, 전통적인 신호처리 기법인 Region Proposal Network를 사용한다.

객체가 있을 법한 Region Proposals을 만든다.

이미지 내에 뭉텅진(Blobby) 곳을 찾는 방식이다.

그리고 이 Region Proposals을 CNN의 입력으로 한다.

Selective Search 방식을 사용하면 2000개 가량의 Region Proposals을 만든다.

Selective Search는 Noise가 심하고, 실제 객체가 아닌 경우가 많지만 Recall이 높다.

### R-CNN.

위의 모든 아이디어들이 R-CNN 논문에 등장한다.

1. Region Proposal Network = Region of Interest(ROI) 수행.

    ![CS231n%20LEC11%203214911eacc34b3da98a52d1278dd5b9/f72252a4-e436-4c43-8346-3b3b77c91141.pdf-0064.png](CS231n%20LEC11%203214911eacc34b3da98a52d1278dd5b9/f72252a4-e436-4c43-8346-3b3b77c91141.pdf-0064.png)

2. 추출한 ROI의 사이즈를 고정된 크기로 바꾼다.

    ![CS231n%20LEC11%203214911eacc34b3da98a52d1278dd5b9/f72252a4-e436-4c43-8346-3b3b77c91141.pdf-0065.png](CS231n%20LEC11%203214911eacc34b3da98a52d1278dd5b9/f72252a4-e436-4c43-8346-3b3b77c91141.pdf-0065.png)

3. CNN을 거치고, SVM을 사용한다.

    ![CS231n%20LEC11%203214911eacc34b3da98a52d1278dd5b9/f72252a4-e436-4c43-8346-3b3b77c91141.pdf-0067.png](CS231n%20LEC11%203214911eacc34b3da98a52d1278dd5b9/f72252a4-e436-4c43-8346-3b3b77c91141.pdf-0067.png)

4. Region Proposals을 보정하기 위한 Regression 과정을 거친다.

    ![CS231n%20LEC11%203214911eacc34b3da98a52d1278dd5b9/f72252a4-e436-4c43-8346-3b3b77c91141.pdf-0068.png](CS231n%20LEC11%203214911eacc34b3da98a52d1278dd5b9/f72252a4-e436-4c43-8346-3b3b77c91141.pdf-0068.png)

    Bbox의 카테고리도 예측하고, Bbox를 보정하는 Offset 4개도 예측한다.

    이를 Multi-task Loss로 하여 학습한다.

5. R-CNN에는 문제점이 있다.

    ![CS231n%20LEC11%203214911eacc34b3da98a52d1278dd5b9/f72252a4-e436-4c43-8346-3b3b77c91141.pdf-0069.png](CS231n%20LEC11%203214911eacc34b3da98a52d1278dd5b9/f72252a4-e436-4c43-8346-3b3b77c91141.pdf-0069.png)

    1. R-CNN은 여전히 계산 비용이 높다.
    2. 학습되지 않은 Region Proposal은 문제의 소지가 있다.
    3. 학습 시간이 오래 걸린다.

    CNN에서 나온 Feature를 디스크에 덤핑한다.

    논문에 따르면 학습 시간에 81시간이 소요되었다고 한다.

#### Question.

Q: "꼭 ROI가 사각형이어야 하는가?"

사각형이 아니면 크기를 조절하기 까다롭다.

Instatnt Semantation의 경우에는 사각형이 아닌 경우도 있다.

Q: "Offsets이 항상 ROI의 안쪽으로만 작용할 수 있는지?"

Offsets이 항상 안쪽으로만 작용해선 안 된다.

필요한 경우, 예측한 Offsets이 Bbox의 외부로 향하기도 한다.

Q: "필요한 데이터가 무엇인지?"

R-CNN도 Fully Supervised이다.

학습 데이터에는 모든 객체에 대한 Bbox가 필요하다.

### Fast R-CNN.

![CS231n%20LEC11%203214911eacc34b3da98a52d1278dd5b9/f72252a4-e436-4c43-8346-3b3b77c91141.pdf-0073.png](CS231n%20LEC11%203214911eacc34b3da98a52d1278dd5b9/f72252a4-e436-4c43-8346-3b3b77c91141.pdf-0073.png)

Fast R-CNN에서는 전체 이미지에 CNN을 수행한다.

전체 이미지에 대한 고해상도 Feature Map을 얻을 수 있다.

이미지에서 ROI를 뜯어내지 않고, CNN Feature Map에 ROI를 Projection 하고 Feature Map에서 뜯어낸다.

그러므로 CNN Feature를 여러 ROI가 공유할 수 있다.

![CS231n%20LEC11%203214911eacc34b3da98a52d1278dd5b9/f72252a4-e436-4c43-8346-3b3b77c91141.pdf-0075.png](CS231n%20LEC11%203214911eacc34b3da98a52d1278dd5b9/f72252a4-e436-4c43-8346-3b3b77c91141.pdf-0075.png)

이후의 FC-layer는 고정된 크기를 입력받기 때문에, ROI의 크기를 조정해야 한다.

학습이 가능하도록, 미분가능한 방법으로 ROI Pooling Layer을 사용한다.

![CS231n%20LEC11%203214911eacc34b3da98a52d1278dd5b9/f72252a4-e436-4c43-8346-3b3b77c91141.pdf-0077.png](CS231n%20LEC11%203214911eacc34b3da98a52d1278dd5b9/f72252a4-e436-4c43-8346-3b3b77c91141.pdf-0077.png)

FC-layer 이후, Classification Score와 Linear Regression Offset을 계산한다.

두 Loss를 더하여 Multi-task Loss로 Backprop를 진행한다.

![CS231n%20LEC11%203214911eacc34b3da98a52d1278dd5b9/f72252a4-e436-4c43-8346-3b3b77c91141.pdf-0078.png](CS231n%20LEC11%203214911eacc34b3da98a52d1278dd5b9/f72252a4-e436-4c43-8346-3b3b77c91141.pdf-0078.png)

[Region of interest pooling explained](https://deepsense.ai/region-of-interest-pooling-explained/)

ROI Pooling은 Max Pooling과 비슷하다고 한다.

ROI Pooling에 대해서는 더 조사하여 추가 예정.

![CS231n%20LEC11%203214911eacc34b3da98a52d1278dd5b9/f72252a4-e436-4c43-8346-3b3b77c91141.pdf-0080.png](CS231n%20LEC11%203214911eacc34b3da98a52d1278dd5b9/f72252a4-e436-4c43-8346-3b3b77c91141.pdf-0080.png)

속도를 보면, Fast R-CNN이 중복 계산이 없어 시간이 매우 빠르다.

그러나 Test Time에서 아직 시간이 2초 가량 남아있는데, Region Proposal을 구하는 Selective Search가 CPU를 사용하는 알고리즘이기 때문이다.

### Faster R-CNN.

![CS231n%20LEC11%203214911eacc34b3da98a52d1278dd5b9/f72252a4-e436-4c43-8346-3b3b77c91141.pdf-0081.png](CS231n%20LEC11%203214911eacc34b3da98a52d1278dd5b9/f72252a4-e436-4c43-8346-3b3b77c91141.pdf-0081.png)

Region Proposal을 직접 만들고, 별도의 RPN이 존재한다.

RPN의 두 Loss와, ROI의 두 Loss까지 4개의 Loss가 존재한다.

BPN으로 Regression 문제를 풀고, ROI 단위로 Classification 한다.

![CS231n%20LEC11%203214911eacc34b3da98a52d1278dd5b9/Untitled.png](CS231n%20LEC11%203214911eacc34b3da98a52d1278dd5b9/Untitled.png)

![CS231n%20LEC11%203214911eacc34b3da98a52d1278dd5b9/f72252a4-e436-4c43-8346-3b3b77c91141.pdf-0082.png](CS231n%20LEC11%203214911eacc34b3da98a52d1278dd5b9/f72252a4-e436-4c43-8346-3b3b77c91141.pdf-0082.png)

Selective Search 부분을 FC-layer로 변경하여 남은 2초 가량의 시간도 줄일 수 있다.

### YOLO & SSD.

![CS231n%20LEC11%203214911eacc34b3da98a52d1278dd5b9/f72252a4-e436-4c43-8346-3b3b77c91141.pdf-0084.png](CS231n%20LEC11%203214911eacc34b3da98a52d1278dd5b9/f72252a4-e436-4c43-8346-3b3b77c91141.pdf-0084.png)

Region Proposals을 활용하지 않는 방법도 있다.

YOLO(You Only Look Once), SSD(Single Shot MultiBox Detector)이다.

각 Task를 따로 계산하는 대신, 하나의 Regression Problem으로 Object Dectection을 수행한다.

한 번의 Feed Forward로 가능하여 Single Shot Mthods이다.

Region Proposals을 활용하는 방법보다 정확도가 낮은 대신, 속도가 빠르다.

1. 이미지를 7X7 Grid로 나누면 각 Grid Cell 내부에는 Base Bbox가 존재한다.

2. 예시의 경우, 3가지 Base Bbox가 있는데, 두 직사각형, 정사각형이다. 

    실제로는 더 많은 개수를 사용한다.

3. Bbox의 Offset을 예측한다.

    실제 위치가 되려면 얼마나 옮겨야 하는지를 의미한다.

4. 각 Bbox에 대해 Classification Scores를 계산한다.

    Bbox 안에 카테고리에 속하는 객체가 존재할 가능성을 의미한다.

5. 입력 이미지가 들어오면, 7X7 Grid마다 (5B + C)개의 Tensor를 갖는다.

    B는 Base Bbox Offset 4개, Confidence Score 1개로 구성된다.

    C는 C개 카테고리에 대한 Classification Score이다.

![CS231n%20LEC11%203214911eacc34b3da98a52d1278dd5b9/f72252a4-e436-4c43-8346-3b3b77c91141.pdf-0085.png](CS231n%20LEC11%203214911eacc34b3da98a52d1278dd5b9/f72252a4-e436-4c43-8346-3b3b77c91141.pdf-0085.png)

VGG, ResNet과 같은 다양한 Base Networks를 적용할 수 있다.

또한, 다양한 Architecture를 선택할 수 있다.

많은 Hyperparameter도 조절할 수 있고, CVPR의 다양한 변수들을 이용한 통제실험 논문을 참고할 수 있다.

### Dense Captioning.

![CS231n%20LEC11%203214911eacc34b3da98a52d1278dd5b9/f72252a4-e436-4c43-8346-3b3b77c91141.pdf-0086.png](CS231n%20LEC11%203214911eacc34b3da98a52d1278dd5b9/f72252a4-e436-4c43-8346-3b3b77c91141.pdf-0086.png)

![CS231n%20LEC11%203214911eacc34b3da98a52d1278dd5b9/f72252a4-e436-4c43-8346-3b3b77c91141.pdf-0087.png](CS231n%20LEC11%203214911eacc34b3da98a52d1278dd5b9/f72252a4-e436-4c43-8346-3b3b77c91141.pdf-0087.png)

![CS231n%20LEC11%203214911eacc34b3da98a52d1278dd5b9/f72252a4-e436-4c43-8346-3b3b77c91141.pdf-0088.png](CS231n%20LEC11%203214911eacc34b3da98a52d1278dd5b9/f72252a4-e436-4c43-8346-3b3b77c91141.pdf-0088.png)

강의 진행자와 Andrej가 Object Detection과 Image Captioning을 조합한 논문이다.

각 Region에 대해 카테고리를 예측하는 것이 아닌, 각 Region의 Caption을 예측한다.

따라서 각 Region에 Caption이 있는 Data Set이 필요하다.

End-to-end로 학습하여 모두 동시에 예측할 수 있다.

Faster R-CNN와 비슷하게 Region Proposal Stage가 있고, 예측한 Bbox 단위로 추가 처리를 한다.

Caption을 예측해야 하므로 RNN Language Model를 도입했다.

## Insatnce Segmentation.

![CS231n%20LEC11%203214911eacc34b3da98a52d1278dd5b9/f72252a4-e436-4c43-8346-3b3b77c91141.pdf-0089.png](CS231n%20LEC11%203214911eacc34b3da98a52d1278dd5b9/f72252a4-e436-4c43-8346-3b3b77c91141.pdf-0089.png)

Instance Segmentation은 Object Detection과 Semantic Segmentation을 합친 방식이다.

![CS231n%20LEC11%203214911eacc34b3da98a52d1278dd5b9/f72252a4-e436-4c43-8346-3b3b77c91141.pdf-0090.png](CS231n%20LEC11%203214911eacc34b3da98a52d1278dd5b9/f72252a4-e436-4c43-8346-3b3b77c91141.pdf-0090.png)

SOTA Mask R-CNN은 Faster R-CNN과 유사하다.

1. 입력 이미지가 CNN과 RPN을 거친다.
2. Feature Map에서 ROI를 뜯어낸다.
3. Feature Map에서 ROI Pooling을 수행하면 두 갈래로 나뉜다.
4. 위 갈래에서는 Region Proposal이 속하는 카테고리를 계산하고, Bbox Regression도 수행한다.
5. 아래 갈래에서는 Classification, Bbox Regression 하지 않고, Bbox마다 Segmentation Mask를 예측한다.
6. 즉, ROI 영역에서 Semantic Segmentation을 수행한다.

![CS231n%20LEC11%203214911eacc34b3da98a52d1278dd5b9/f72252a4-e436-4c43-8346-3b3b77c91141.pdf-0091.png](CS231n%20LEC11%203214911eacc34b3da98a52d1278dd5b9/f72252a4-e436-4c43-8346-3b3b77c91141.pdf-0091.png)

Instance Segmentation의 결과!

![CS231n%20LEC11%203214911eacc34b3da98a52d1278dd5b9/f72252a4-e436-4c43-8346-3b3b77c91141.pdf-0092.png](CS231n%20LEC11%203214911eacc34b3da98a52d1278dd5b9/f72252a4-e436-4c43-8346-3b3b77c91141.pdf-0092.png)

![CS231n%20LEC11%203214911eacc34b3da98a52d1278dd5b9/f72252a4-e436-4c43-8346-3b3b77c91141.pdf-0093.png](CS231n%20LEC11%203214911eacc34b3da98a52d1278dd5b9/f72252a4-e436-4c43-8346-3b3b77c91141.pdf-0093.png)

또한 Classification + Localization처럼 Pose Estimation도 가능하다.

Loss와 Layer 하나만 더 추가하여 구현할 수 있다.

## Recap & Next Time.

![CS231n%20LEC11%203214911eacc34b3da98a52d1278dd5b9/f72252a4-e436-4c43-8346-3b3b77c91141.pdf-0094.png](CS231n%20LEC11%203214911eacc34b3da98a52d1278dd5b9/f72252a4-e436-4c43-8346-3b3b77c91141.pdf-0094.png)

![CS231n%20LEC11%203214911eacc34b3da98a52d1278dd5b9/f72252a4-e436-4c43-8346-3b3b77c91141.pdf-0095.png](CS231n%20LEC11%203214911eacc34b3da98a52d1278dd5b9/f72252a4-e436-4c43-8346-3b3b77c91141.pdf-0095.png)

## 링크.

[Lecture 11 | Recurrent Neural Networks |](https://www.youtube.com/watch?v=nDPWywWRIRo&list=PL3FW7Lu3i5JvHM8ljYj-zLfQRF3EO8sYv&index=11)

[CS231n: Convolutional Neural Networks for Visual Recognition](http://cs231n.stanford.edu/2017/)

[CS231n 2017 Lecture11 PDF](http://cs231n.stanford.edu/slides/2017/cs231n_2017_lecture11.pdf)

[soline013/CS231N_17_KOR_SUB](https://github.com/soline013/CS231N_17_KOR_SUB/blob/master/kor/Lecture%2011%20%20%20Detection%20and%20Segmentation.ko.srt)

[Visual Image Media_HJK : 네이버 블로그](https://blog.naver.com/dr_moms/221631504020)