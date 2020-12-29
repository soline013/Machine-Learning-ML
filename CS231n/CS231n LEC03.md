# CS231n LEC03.
## Stanford University, Spring 2017.
**Loss Functions and Optimization.**

## Recall from Last Time.
![image](https://user-images.githubusercontent.com/66259854/99666333-395cef00-2aae-11eb-8b48-74a5cbae10cd.png)
![image](https://user-images.githubusercontent.com/66259854/99666349-3eba3980-2aae-11eb-8788-96c65bbf76c3.png)
![image](https://user-images.githubusercontent.com/66259854/99666358-411c9380-2aae-11eb-815a-c4cd66b5dd6c.png)

## Start LEC03, Linear Classifier.
![image](https://user-images.githubusercontent.com/66259854/99666371-4548b100-2aae-11eb-93ca-fb1a2978f761.png)

지난 시간, 임의의 행렬 W를 가지고 예측한 10 Class Scores.

  1. 고양이 사진 - Cat 카테고리가 2.9점이나, 점수가 더 높은 다른 카테고리가 존재한다.
  2. 자동차 사진 - Automobile 카테고리가 6.04점으로 제일 높다. 잘 예측한 것.
  3. 개구리 사진 - Frog 카테고리가 -4.34이다.

가장 나은 W를 찾기 위한 방법 → Loss Function.

* * *

![image](https://user-images.githubusercontent.com/66259854/99666381-48dc3800-2aae-11eb-94a1-f27ff815cd35.png)

X=Input, Y=Label, Target (In CIFAR10, 1-10 or 0-9)

$L = \frac{1}{N} \sum_i L_i(f(x_i, W), y_i)$

## Multi-class SVM Loss.
![image](https://user-images.githubusercontent.com/66259854/99666384-4bd72880-2aae-11eb-86f8-63637aa336b9.png)

  1. True카테고리를 제외한 나머지 카테고리의 Y합을 구한다.
  2. True카테고리 스코어($S_{Y_i}$)와 Not True카테고리 스코어($S_j$)를 비교하여,   
     True > Not True이고, 차이가 Safety Margin 이상이라면 Loss는 0이다. (여기서 Margin=1)
     
  3. Not True 카테고리의 모든 값의 합이 한 이미지의 Loss이다.
  4. 전체 Training Set에서 Loss의 평균을 구한다.

Max(0, value) 형식으로 Loss Function을 만들고, 이를 Hinge(경첩)이라 부름.

X축은 True카테고리 스코어, Y축은 Loss.

Case Space Notation 대신 Zero One Notation을 사용.

### Example.
![image](https://user-images.githubusercontent.com/66259854/99666387-4e398280-2aae-11eb-8e51-386d964b561a.png)
![image](https://user-images.githubusercontent.com/66259854/99666390-50034600-2aae-11eb-97cb-09e20ff38273.png)
![image](https://user-images.githubusercontent.com/66259854/99666398-51cd0980-2aae-11eb-9219-c9091c5d3453.png)
![image](https://user-images.githubusercontent.com/66259854/99666404-542f6380-2aae-11eb-84e9-7d1e5830456f.png)

### Question.

Q: “$S, S_{Y_i}$는 무엇을 의미하는가?"   
$S_1, S_2, ..., S_n$: 카테고리 별 예측된 스코어 값.   
$S_{Y_i}$: i번째 True카테고리 스코어, $Y_i$는 True카테고리.

Q: “Safety Margin은 어떻게 정하는가?”   
Loss Function의 스코어가 아닌, 여러 스코어의 상대적 차이에 관심이 있다.   
행렬 W에 의해 상쇄되는 값이므로 크게 상관이 없다.

Q: “Car 스코어를 조금 바꾼다면 Loss는 바뀌는가?”   
True인 Car 스코어가 여전히 높아서 바뀌지 않고 0일 것이다.

Q: "SVM Loss의 최대와 최소는?"   
최소는 0이고, 최대는 무한대이다.

Q: “모든 스코어가 0에 가깝고, 값이 비슷하다면 Loss는 어떻게 되는가?”   
Class Number(C) – 1   
순회 횟수가 C – 1이기 때문이다.   
(+Using Debugging, 처음 학습에서 Loss = C -1이 아니라면 Bug가 있다고 생각할 수 있다.   
처음 학습에서 행렬 W는 임의의 작은 수로 초기화, 스코어 또한 임의의 일정한 값을 갖기 때문.)

Q: “True Class도 더하면 어떻게 되는가?”   
Loss + 1   
Loss가 최소인 0이 되도록 하는 것 때문에 관습적으로 True Class는 넣지 않는다.

Q: “Loss에서 전체 합이 아닌 평균을 쓰면 어떻게 되는가?”   
상관없다. 클래스의 수는 정해져 있고, 스코어 값은 신경 쓸 요소가 아니다.

Q: “Loss Function을 제곱으로 바꾼다면?” $L=\sum_{j≠y_i}max(0, S_j-S_{Y_i}+1)^2$   
결과가 달라진다. 비선형적 방식으로 Loss 계산이 달라진다.   
작은 것은 작아지고, 큰 것은 더 커진다.

## Multi-class SVM Loss Code.
![image](https://user-images.githubusercontent.com/66259854/99666415-572a5400-2aae-11eb-92ad-5e7a19c75cd5.png)

margins[y]: max로 나온 결과에서 True Class만 0으로 만든다.

Vectorized 하므로 전체를 순회할 필요가 없다.

전체 합을 구할 때, 제외할 부분만 0이 되는 것.

그리고 많은 W 중 Loss = 0인 W를 선택해선 안 된다. → Overfitting.

### Question.
Q: “Loss = 0인 W는 하나만 있을까?”   
W는 변하기 때문에 다른 W도 존재한다.   
W와 2W가 있다면, 여전히 Loss = 0이다.


## Regularization.
![image](https://user-images.githubusercontent.com/66259854/99666427-5a254480-2aae-11eb-90a3-bf7329504de1.png)

![image](https://user-images.githubusercontent.com/66259854/99666431-5bef0800-2aae-11eb-9fdc-e746d52cf021.png)

파란색 점에 Fit 한다고 할 때, 파란색 곡선 형태는 Test Data의 성능을 고려하지 않는다.

→ 따라서 초록색 직선이 이상적이고, Regularization(정규화)를 거친다.

Regularization: Regularization Term(Penalty)을 추가하여 보다 단순한 W를 선택하게 한다.

$\lambda$: Hyper Parameter, Regularization Strength로 Trade-off 설정.

### Question.
Q: “$Wx+\lambda R$이 어떻게 곡선을 직선으로 바꾸는가?”   
쉽게 말하면, 저차 다항식을 선호하도록 만든다.   
더 복잡해지지 않도록 하는 것 / Soft Penalty를 추가하는 것 → 여전히 복잡해질 수 있다.

## L2 Regularization.
![image](https://user-images.githubusercontent.com/66259854/99666442-5e516200-2aae-11eb-9be9-5e63487ec406.png)

가중치 행렬 W에 대한 Euclidean Norm, Squared Norm. ($\frac{1}{2}*Squared Norm$을 사용하기도 함.)

(+Euclidean Norm: 두 점 사이의 거리를 계산하고, 이를 통해 유클리드 공간을 정의, 그 거리에 대응하는 Norm(크기)값이다.)

  1. Dot Product를 진행하면 $w^T_1x=w^T_2x=1$로 같으므로, $w_1$, $w_2$는 같다.
  2. 이때 L2는 Norm이 더 작은 $w_2$를 선호한다.
  3. L2는 W 중 어떤 게 변동이 심한지에 따라 복잡도를 정의한다. (퍼져 있으면 덜 복잡하다.)   
     L2는 x의 모든 요소가 영향을 끼치는 쪽으로 선호.
  4. L1은 W의 0의 개수에 따라 복잡도를 정의하므로, $w_1$을 선호한다. (0이 많으면 덜 복잡하다.)

## Softmax. (Multinomial Logistic Regression)
![image](https://user-images.githubusercontent.com/66259854/99666455-614c5280-2aae-11eb-9b12-36778e0f53da.png)

![image](https://user-images.githubusercontent.com/66259854/99666462-63161600-2aae-11eb-93ea-5ec3a5eca9ea.png)

Multi-class SVM은 스코어 자체는 의미가 없이, True Class가 더 높은 스코어면 되었다.

하지만 Softmax는 스코어에 의미를 부여한다.

  1. 스코어를 $e^s$로 바꾸어 양수로 만든다.
  2. 이 수들의 합을 분모로 하여 정규화를 한다.
  3. 0~1 사이의 확률이 나온다.

Log는 단조 증가 함수로 Log의 값을 최대로 하는 게 더 편하므로 Log를 사용한다.

또한, Loss Function으로써 나쁜 정도를 측정하므로 (-)를 붙인다.

### Question.
Q: “Softmax Loss의 최대와 최소는?”   
최소는 0이고, 최대는 무한대이다.   
유한 정밀도가 존재하므로 이론적인 수치에 도달할 수는 없다.

Q: “S의 값이 모두 0에 가까운 작은 수라면 Loss는 어떻게 되는가?”   
$-log(\frac{1}{c})=log(c)$ (Using Debugging)

## SVM Loss VS Softmax Loss.
![image](https://user-images.githubusercontent.com/66259854/99666481-67daca00-2aae-11eb-8d4c-68f2fa4b185d.png)
  1. SVM
      - True카테고리 스코어($S_{Y_i}$)와 Not True카테고리 스코어($S_{j}$)의 Margin을 고려.
      - True카테고리 스코어($S_{Y_i}$)가 바뀌어도 Loss는 바뀌지 않음.
      - 성능 개선을 신경 쓰지 않음. (Safety Margin만 넘기면 되므로, 스코어 자체는 의미 없다.)
  2. Softmax
      - 확률을 구하여 -log(True Class)를 고려.
      - True카테고리 스코어($S_{Y_i}$)가 바뀌면 Loss는 바뀜.
      - 성능을 개선하고자 함. (스코어 자체가 의미를 가지므로.)

## Recap.
![image](https://user-images.githubusercontent.com/66259854/99666492-6ad5ba80-2aae-11eb-9653-9654153bb3a2.png)

## Optimization(최적화).
  1. 임의 탐색(Random Search)
  
     ![image](https://user-images.githubusercontent.com/66259854/99666501-6c9f7e00-2aae-11eb-9893-d9114639c2b7.png)
     ![image](https://user-images.githubusercontent.com/66259854/99666507-6e694180-2aae-11eb-9fa8-3b299c114971.png)

     임의의 W를 매우 많이 모으고 Loss를 계산하는 것.

  2. Follow the Slope.
  
     ![image](https://user-images.githubusercontent.com/66259854/99666514-70cb9b80-2aae-11eb-8ede-1ce5d9175e7a.png)
     
     Numerical Gradient.
     
     유한 차분법(Finite Difference Methods) → Using Debugging.
     
     ![image](https://user-images.githubusercontent.com/66259854/99666519-72955f00-2aae-11eb-9d43-319cfb668371.png)
     
     Analytic Gradient.
     
     Gradient를 나타내는 식을 찾아 미분으로 한 번에 계산하는 게 좋다.

## Gradient Descent Code.
![image](https://user-images.githubusercontent.com/66259854/99666529-74f7b900-2aae-11eb-8eb3-3f0e5b25a895.png)

W를 초기화하고, Gradient는 함수가 증가하는 방향이므로 (-)를 붙여 감소하는 방향으로 바꾼다.

Step Size: Hyper Parameter, Learning Rate.

## Stochastic Gradient Descent(SGD).
![image](https://user-images.githubusercontent.com/66259854/99666539-775a1300-2aae-11eb-9394-79f7e788bf82.png)

Minibatch라는 작은 집합으로 나누어 학습한다.

보통 2의 승수를 사용함.

Monte Carlo Method의 실제 값 추정과 유사하다.

## Image Features.
![image](https://user-images.githubusercontent.com/66259854/99666553-79bc6d00-2aae-11eb-8a95-ecc85eec6e9c.png)

실제 Raw 이미지 픽셀을 입력 받는 것은 좋지 않다. Because Multi-modality.

Features를 추출하고, Concat하여 하나의 특징 벡터를 만든다.

* * *

![image](https://user-images.githubusercontent.com/66259854/99666558-7c1ec700-2aae-11eb-930f-7b8b39ec14db.png)

이런 Data는 Linear Classifier이 불가능한데, 극좌표계로 특징을 변환하면 가능하다.

* * *

![image](https://user-images.githubusercontent.com/66259854/99666569-7de88a80-2aae-11eb-89e6-296ed91011a9.png)

  1. 이미지에서 모든 픽셀의 Hue 값을 뽑아 각 Bucket에 넣는다.
  2. Bucket을 확인하면 이미지의 전체적인 색을 확인할 수 있다.

* * *

![image](https://user-images.githubusercontent.com/66259854/99666575-7fb24e00-2aae-11eb-878c-d9af4a3d74b2.png)

Hubel & Wiesel의 연구를 떠올릴 수 있다.

  1. Local Orientation Edges를 측정한다.
  2. 특정 규모의 픽셀로 나누고, 지배적인 Edge의 방향을 양자화 하여 Bucket에 넣는다.
  3. Edge Orientations에 대한 히스토그램을 계산한다.

* * *

![image](https://user-images.githubusercontent.com/66259854/99666578-8214a800-2aae-11eb-88b2-eb176e23d573.png)

Visual Words라는 용어를 정의해야 했음.

  1. 많은 이미지를 임의로 조각 내고, K-means 같은 알고리즘으로 군집화.
  2. Visual Words는 색과 Edges를 파악할 수 있다.
  3. Visual Words의 집합인 Codebook을 만든다.
  4. 어떤 이미지에서 Visual Words의 발생 빈도로 이미지를 인코딩한다.

## 링크.
[Lecture 3 | Loss Functions and Optimization |](https://www.youtube.com/watch?v=h7iBpEHGVNc&feature=emb_title)

[CS231n: Convolutional Neural Networks for Visual Recognition](http://cs231n.stanford.edu/2017/)

[CS231n 2017 Lecture3 PDF](http://cs231n.stanford.edu/slides/2017/cs231n_2017_lecture3.pdf)

[soline013/CS231N_17_KOR_SUB](https://github.com/soline013/CS231N_17_KOR_SUB/blob/master/kor/Lecture%203%20%20%20Loss%20Functions%20and%20Optimization.ko.srt)
