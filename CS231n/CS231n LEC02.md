# CS231n LEC02.
## Stanford University, Spring 2017.
**Image Classification.**

## Assignment.
K-Nearest Neighbor.

Linear Classifiers: SVM, Softmax.

Two-layer Neural Network.

Image Features.

## Image Classification.
![image](https://user-images.githubusercontent.com/66259854/98202482-72a54300-1f75-11eb-959e-3151ecd1dc6b.png)

입력 이미지를 받고 시스템에서 미리 정한 카테고리 집합 중에 어떤 카테고리인지 고르는 것.

* * *

![image](https://user-images.githubusercontent.com/66259854/98202495-7769f700-1f75-11eb-95c3-f654649584ca.png)

기계 입장에서는, 픽셀 모양의 숫자 집합으로 각 픽셀은 세 개의 숫자로 표현한다.

e.g. 800 x 600 x 3. 이걸 Semantic Gap(의미론적 차이)이라 함.

Challenging Problem: Viewpoint Variation, Illumination, Deformation, Occlusion, Background Clutter, Intracalss Variation 등의 작은 변화에도 강해야 한다. 

## How Can We Do Image Classification?
   1. 이미지에서 Edges 계산하기.
      
      ![image](https://user-images.githubusercontent.com/66259854/98202504-7b961480-1f75-11eb-866a-33255bbe95ed.png)
      
      많은 Corners와 Edges를 분류하여 명시적인 규칙의 집합을 만든다.
      
      → 작은 변화에 강하지 않다. 즉, 잘 작동하지 않는다. 
      
   2. Data Driven Approach.
    
      ![image](https://user-images.githubusercontent.com/66259854/98202515-82bd2280-1f75-11eb-8170-fb29589a2efd.png)

      인터넷으로 많은 데이터를 수집하고, Machine Learning Classifier 학습.
      
      그 학습 모델로 새로운 이미지를 테스트하기.
      
        1. Train 함수 | (Input: Image, Label) → (Output: Model)
        2. Predict 함수 | (Input: Model) → (Output: Accuracy)

## Nearest Neighbor. (NN)
![image](https://user-images.githubusercontent.com/66259854/98202512-7fc23200-1f75-11eb-9f56-25228a6c0286.png)

Train에서 모든 학습 데이터를 기억, Predict에서 새로운 이미지가 들어오면 학습 데이터와 비교.

## Example Dataset: CIFAR 10.
![image](https://user-images.githubusercontent.com/66259854/98202520-8650a980-1f75-11eb-8458-2113adc6aae2.png)

NN을 사용하여 분류한 오른쪽 그림.

제일 왼쪽 열은 Test Image이고, 오른쪽으로 갈수록 Test Image와 유사하다.

그렇다면 이미지 쌍이 있을 때, 어떤 비교 함수를 사용하여 비교할 것인가? → L1 Distance.

## L1 Distance. (Manhattan Distance)
![image](https://user-images.githubusercontent.com/66259854/98202525-8a7cc700-1f75-11eb-94e5-1ee675b3b11b.png)

  1. 이미지를 Pixel-wise로 비교한다.
  2. Test/Training 이미지에서 서로 같은 자리의 픽셀을 빼고 절대값을 취한다.
  3. 모든 픽셀을 더한다.

## L1 Distance로 NN을 구현한 Python 코드.
![image](https://user-images.githubusercontent.com/66259854/98202536-9072a800-1f75-11eb-930a-b2a6af97883f.png)

N개의 Training Set이 있을 때, 걸리는 시간?
  
  1. Train: O(1) → 데이터를 기억하는 시간, 상수 시간.
  2. Test: O(N) → N개를 전부 비교하는 시간, 느리다.

Train Time < Test Time으로, Test는 Low Power Device 등의 환경에서 빠른 성능을 요구.

## Decision Regions. (NN's Application)
![image](https://user-images.githubusercontent.com/66259854/98202542-94062f00-1f75-11eb-891e-abc8214e37b6.png)

Point: Training Data.

Point Color: Class Label(Category)

2차원 평면의 모든 좌표와 가장 가까운 학습 데이터 게산, 해당하는 Class로 칠함.

문제점.
  1. The Nearest Neighbor만을 보기 때문에, 가운데 노란색 영역이 생겼다.
  2. 초록색 영역 또한 파란색 영역을 침범하고 있다.
  3. Noise, Spurious라 부름.

## K-NN.
![image](https://user-images.githubusercontent.com/66259854/98202556-98324c80-1f75-11eb-9ca3-7a4568d0e345.png)

  1. Distance Metric을 이용하여, 가까운 이웃을 K만큼 찾는다.
  2. 이웃끼리 투표하여 가장 많은 표를 받은 Label로 예측한다.
  3. 투표 방법 중 득표수만 고려하는 게 쉽고 정확하다.
  4. K > 1일 때, 더 부드럽고 좋은 결과가 나온다.

Q: "Labelling 되지 않은 흰색 지역은 어떻게 처리하는가?"

K-NN이 대다수를 정할 수 없는 지역으로, 추론하거나 임의로 정할 수 있다.

Image 문제에서 K-NN은 좋은 방법이 아니다.

이미지를 고차원에 존재하는 하나의 점 / 이미지의 픽셀을 하나의 고차원으로 생각할 수 있는데, 이런 관점을 계속 오가는 것이 좋다.

## *추가 예정.*

## 링크.
[Lecture 2 | Image Classification |](https://www.youtube.com/watch?v=OoUX-nOEjG0&feature=emb_title)

[CS231n: Convolutional Neural Networks for Visual Recognition](http://cs231n.stanford.edu/2017/)

[CS231n 2017 Lecture2 PDF](http://cs231n.stanford.edu/slides/2017/cs231n_2017_lecture2.pdf)

[soline013/CS231N_17_KOR_SUB](https://github.com/soline013/CS231N_17_KOR_SUB/blob/master/kor/Lecture%202%20%20%20Image%20Classification.ko.srt)
