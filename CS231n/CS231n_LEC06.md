# CS231n LEC06.
## Stanford University CS231n, Spring 2017.
**Training Neural Networks.**

## Recall from last time.
![image](https://user-images.githubusercontent.com/66259854/101941803-d8978100-3c2b-11eb-96f4-522edaeea7bb.png)
![image](https://user-images.githubusercontent.com/66259854/101941902-febd2100-3c2b-11eb-9e28-100d96e79375.png)
![image](https://user-images.githubusercontent.com/66259854/101941922-07adf280-3c2c-11eb-92ec-b6e8a6a8a308.png)
![image](https://user-images.githubusercontent.com/66259854/101941938-0f6d9700-3c2c-11eb-819a-8cfb19329cc4.png)
![image](https://user-images.githubusercontent.com/66259854/101941963-1694a500-3c2c-11eb-8286-ce83187ca0f0.png)
![image](https://user-images.githubusercontent.com/66259854/101941980-1eece000-3c2c-11eb-8b57-9d950388823f.png)
![image](https://user-images.githubusercontent.com/66259854/101941998-257b5780-3c2c-11eb-8d5c-a748fc4d480a.png)

## Overview.

1. One Time Setup.

    Activation Functions, Preprocessing, Weight Initialization, (Regularization, Gradient Checking.)

2. Training Dynamics.

    Babysitting the Learning Process, Hyperparameter Optimization, (Parameter Updates.)

3. Evaluation.

    (Model Ensembles.)

## One Time Setup.

## Activation Functions.

![image](https://user-images.githubusercontent.com/66259854/101942876-940ce500-3c2d-11eb-8ce7-dec71272c606.png)

![image](https://user-images.githubusercontent.com/66259854/101942892-996a2f80-3c2d-11eb-9b68-6dd79b9d206d.png)

입력된 데이터와 가중치를 곱하고 활성함수, 비선형 연산을 거친다.

다양한 Activation Function이 있다.

### Sigmoid.

![image](https://user-images.githubusercontent.com/66259854/101943184-041b6b00-3c2e-11eb-91d7-da036de77c7a.png)

입력 값을 0 ~ 1 사이의 값으로 만든다.

입력이 크면 1에 가깝고, 입력이 작으면 0에 가깝다.

0 근처의 Rigime은 선형과 비슷하다.

Sigmoid에는 세 가지 문제점이 있다.

1. Saturation neurons kill the gradients.
   
   ![image](https://user-images.githubusercontent.com/66259854/101943197-0978b580-3c2e-11eb-89b0-492cf7a45d6a.png)
   
   Back Prop에서 Gradient는 $\frac{dL}{dx} = \frac{dL}{d \sigma}  \ \frac{d \sigma}{dx}$이다.
   
   1. x의 값이 -10이라면, Gradient는 0이 계속 전달된다.
   2. x의 값이 0이라면, Gradient는 의미 있는 값이 계속 전달된다.
   3. x의 값이 10이라면, Gradient는 1이 계속 전달된다.
   
   즉, Sigmoid 함수의 수평 부분에서 Gradient는 의미 없는 값이다.

2. Sigmoid outputs are not zero-centered.
   
   ![image](https://user-images.githubusercontent.com/66259854/101943211-0ed60000-3c2e-11eb-8e3a-29db09ba8195.png)
   
   x가 항상 양수일 때, x는 어떤 가중치랑 곱해지고 활성함수를 통과한다.
   
   W에 대한 Gradient 값은 전부 양수거나 음수이다.
   
   $$\frac{dL}{df} \times \frac{df}{dw} = \frac{dL}{df} \times x$$
   
   따라서 Parameter를 업데이트 하여도 W는 다 같이 증가하거나 감소하여 지그재그 모양이 된다.
   
   Gradient 업데이트를 여러 번 수행해야 한다.

3. exp() is a bit compute expensive.
   
   그렇게 큰 문제는 아니다.
   
   오히려 내적의 계산 비용이 더 크다.

---

### tanh.

![image](https://user-images.githubusercontent.com/66259854/101944151-9a9c5c00-3c2f-11eb-9bc6-730d34d66d22.png)   

tanh는 입력 값을 -1 ~ 1 사이로 만든다.

1. 장점.
   
   Zero-centered로, "Sigmoid outputs are not zero-centered." 문제가 해결된다.
   
2. 단점.
   
   그러나 "Saturation neurons kill the gradients."는 여전하다.

---

### ReLU.

![CS231n%20LEC06%203fb2279d00804036b36489fa2286428a/Untitled%2013.png](CS231n%20LEC06%203fb2279d00804036b36489fa2286428a/Untitled%2013.png)

입력이 음수면 Element-wise 연산으로 값이 0이다.

1. 장점.
   1. Sigmoid, tanh와 다르게 양의 수에서 Saturation(포화)가 없다.
   2. max 연산으로 속도가 빠르다.

2. 단점.
   1. Zero-centered가 아니다.
   2. 음의 수에서 Saturation이 발생한다.
      
      ![CS231n%20LEC06%203fb2279d00804036b36489fa2286428a/Untitled%2014.png](CS231n%20LEC06%203fb2279d00804036b36489fa2286428a/Untitled%2014.png)
      
      ReLU는 Gradient의 절반을 죽이는데, 이를 Dead ReLU라고 한다.
      
      ![CS231n%20LEC06%203fb2279d00804036b36489fa2286428a/Untitled%2015.png](CS231n%20LEC06%203fb2279d00804036b36489fa2286428a/Untitled%2015.png)
      
      Data Cloud = Training Data.
      
      ReLU에서 평면의 절반만 Activate 된다.
      
      실제 네트워크의 10~20%는 Dead ReLU로 나온다.
      
      ReLU의 초기화에서 Positive Biases 값을 추가한다. (그래도 대부분 Zero-bias로 초기화.)
      
      Data Cloud와 ReLU가 멀리 있는 경우에도 Dead ReLU가 발생하는데 2가지 이유가 있다. 
      
      1. Wrong Initialize.
         
         가중치가 초평면을 이루는데, 초평면 자체가 멀리서 생길 수 있다.

      2. High High Learning Rate.
         
         가중치가 커져 데이터의 Manifold를 벗어남.

### Leaky ReLU. PReLU.

![CS231n%20LEC06%203fb2279d00804036b36489fa2286428a/Untitled%2016.png](CS231n%20LEC06%203fb2279d00804036b36489fa2286428a/Untitled%2016.png)

음수 구간에 기울기를 줘서, Saturation을 방지한다.

        PReLU는 기울기을 $\alpha$로 두고 Back Prop로 Parameter를 찾아간다.

    - ELU.

        ![CS231n%20LEC06%203fb2279d00804036b36489fa2286428a/Untitled%2017.png](CS231n%20LEC06%203fb2279d00804036b36489fa2286428a/Untitled%2017.png)

        Zero-mean에 보다 가까운 출력값이 나온다.

        음수 구간에서 Saturation이 발생하지만 Noise에 강하다.

        ReLU(Saturation)와 Leaky ReLU(Zero-mean)의 중간 정도..?

    - Maxout.

        ![CS231n%20LEC06%203fb2279d00804036b36489fa2286428a/Untitled%2018.png](CS231n%20LEC06%203fb2279d00804036b36489fa2286428a/Untitled%2018.png)

        입력 형식이 정해져 있지 않다.

        $w^T_1x+b_1, w^T_2x+b_2$ 처럼 받고 두 개 중 최댓값을 취한다.

        뉴런 당 Parameter가 배가 되는 단점이 있다.

    ![CS231n%20LEC06%203fb2279d00804036b36489fa2286428a/Untitled%2019.png](CS231n%20LEC06%203fb2279d00804036b36489fa2286428a/Untitled%2019.png)

- Data Preprocessing.

    ![CS231n%20LEC06%203fb2279d00804036b36489fa2286428a/Untitled%2020.png](CS231n%20LEC06%203fb2279d00804036b36489fa2286428a/Untitled%2020.png)

    Image는 스케일이 꽤 맞춰져 있어, Normalization이 거의 필요 없다.

    Sigmoid에는 Zero-mean이 필요하지만, Preprocessing을 해도 처음 Layer에서만 작동하고 이후에는 문제가 반복된다.

    1. Zero-mean으로 만든다.

        Zero-mean에서 평균은 Training Data로 계산한다.

        Test Data에서도 똑같은 처리가 필요함.

    2.  이후에 Normalize 한다.

        Normalization은 보통 표준편차로 한다.

        데이터가 모두 Positive, 0, Negative일 때 Suboptimal 하므로,

        Normalization으로 모든 차원이 동일한 범위에 있도록 한다.

    ![CS231n%20LEC06%203fb2279d00804036b36489fa2286428a/Untitled%2021.png](CS231n%20LEC06%203fb2279d00804036b36489fa2286428a/Untitled%2021.png)

    PCA, Whitening 같은 과정도 존재.

    ![CS231n%20LEC06%203fb2279d00804036b36489fa2286428a/Untitled%2022.png](CS231n%20LEC06%203fb2279d00804036b36489fa2286428a/Untitled%2022.png)

    Channel은 RGB로 32, 32, 3 같은 경우 3이 RGB이다.

    VGG에서는 RGB 각각의 평균을 구한다.

- Weight Initialization.

    Q: "Parameter를 모두 0으로 만들면 어떻게 될까?"

    뉴런이 모두 똑같다. Parameter도 같은 값으로 업데이트 된다.

    1. 임의의 작은 값으로 초기화.

        ![CS231n%20LEC06%203fb2279d00804036b36489fa2286428a/Untitled%2023.png](CS231n%20LEC06%203fb2279d00804036b36489fa2286428a/Untitled%2023.png)

        작은 값을 위해 0.01을 나눠 표준편차를 0.01로 만든다.

        ![CS231n%20LEC06%203fb2279d00804036b36489fa2286428a/Untitled%2024.png](CS231n%20LEC06%203fb2279d00804036b36489fa2286428a/Untitled%2024.png)

        ![CS231n%20LEC06%203fb2279d00804036b36489fa2286428a/Untitled%2025.png](CS231n%20LEC06%203fb2279d00804036b36489fa2286428a/Untitled%2025.png)

        tanh이므로 첫 번째 레이어 평균은 항상 0 근처.

        표준편차는 점점 0에 수렴하는데, W가 너무 작은 값이기 때문.

        Thus, All Activations Become Zero!

        1. 파란선: 레이어에 따른 평균.
        2. 빨간선: 표준편차.
        3. 하단 그래프: Standard Gaussian.

        X가 엄청 작은 값이므로 Gradient도 작아질 것이고, 업데이트가 잘 일어나지 않음.

    2. 가중치를 더 큰 값으로 초기화.

        ![CS231n%20LEC06%203fb2279d00804036b36489fa2286428a/Untitled%2026.png](CS231n%20LEC06%203fb2279d00804036b36489fa2286428a/Untitled%2026.png)

        출력이 -1 또는 1이 나오고, Saturation 되어 Gradient는 0이 된다.

    3. Xavier Initialization.

        ![CS231n%20LEC06%203fb2279d00804036b36489fa2286428a/Untitled%2027.png](CS231n%20LEC06%203fb2279d00804036b36489fa2286428a/Untitled%2027.png)

        선형 함수에서만 적용 가능하다.

        W는 정규분포로 얻은 값에 입력 데이터의 제곱근으로 나누어 구한다.

        입력과 출력의 분산을 맞춰준다.

        1. 입력이 작으면, 더 작은 값으로 나누어 W가 커진다.
        2. 입력이 크면, 더 큰 값으로 나누어 W가 작아진다.

        ![CS231n%20LEC06%203fb2279d00804036b36489fa2286428a/Untitled%2028.png](CS231n%20LEC06%203fb2279d00804036b36489fa2286428a/Untitled%2028.png)

        ReLU에서는 효과가 적은데, 출력의 절반을 죽이기 때문.

        ![CS231n%20LEC06%203fb2279d00804036b36489fa2286428a/Untitled%2029.png](CS231n%20LEC06%203fb2279d00804036b36489fa2286428a/Untitled%2029.png)

    ![CS231n%20LEC06%203fb2279d00804036b36489fa2286428a/Untitled%2030.png](CS231n%20LEC06%203fb2279d00804036b36489fa2286428a/Untitled%2030.png)

- Batch Normalization.

    ![CS231n%20LEC06%203fb2279d00804036b36489fa2286428a/Untitled%2031.png](CS231n%20LEC06%203fb2279d00804036b36489fa2286428a/Untitled%2031.png)

    Batch Normalization은 Gradient Vanishing을 해결하기 위해 만들어졌다.

    Weight Initializtion 보다도 Input의 평균과 분산을 구하여 정규분포로 만드는 것.

    ![CS231n%20LEC06%203fb2279d00804036b36489fa2286428a/Untitled%2032.png](CS231n%20LEC06%203fb2279d00804036b36489fa2286428a/Untitled%2032.png)

    Input을 Batch 단위로 나누어 처리한다.

    Batch당 N개의 학습 데이터가 있고, D차원이다.

    각 차원별로 평균과 분산을 구하고, 하나의 Batch에서 Normalize 한다.

    ![CS231n%20LEC06%203fb2279d00804036b36489fa2286428a/Untitled%2033.png](CS231n%20LEC06%203fb2279d00804036b36489fa2286428a/Untitled%2033.png)

    BN은 Activation Function 전에, FC나 Conv 이후에 넣는다.

    이때, Conv에서는 차원마다 진행하지 않고 같은 채널에 있는 요소로 진행한다.

    ![CS231n%20LEC06%203fb2279d00804036b36489fa2286428a/Untitled%2034.png](CS231n%20LEC06%203fb2279d00804036b36489fa2286428a/Untitled%2034.png)

    Normalization 연산으로 Saturation이 전혀 없는 것보다는, Saturation의 정도를 조절하는 것이 좋다.

    따라서 BN에 Scaling 연산을 추가한다.

     $\gamma$는 표준편차로 Scaling 효과를, $\beta$는 평균으로 Shift 효과를 주어 Normalization 이전 상태로 만들 수 있다. ($\gamma$, $\beta$는 학습 가능한 Parameter.)

    ![CS231n%20LEC06%203fb2279d00804036b36489fa2286428a/Untitled%2035.png](CS231n%20LEC06%203fb2279d00804036b36489fa2286428a/Untitled%2035.png)

    ![CS231n%20LEC06%203fb2279d00804036b36489fa2286428a/Untitled%2036.png](CS231n%20LEC06%203fb2279d00804036b36489fa2286428a/Untitled%2036.png)

    BN은 Regularization의 역할을 할 수 있다.

    Layer의 출력은 Batch 안에 있는 모든 데이터에게 영향을 받기 때문에 Regularization Effect가 발생한다.

### Training Dynamics.

- Babysitting the Learning Process.

    ![CS231n%20LEC06%203fb2279d00804036b36489fa2286428a/Untitled%2037.png](CS231n%20LEC06%203fb2279d00804036b36489fa2286428a/Untitled%2037.png)

    Image에서는 Zero-mean만 주로 사용.

    ![CS231n%20LEC06%203fb2279d00804036b36489fa2286428a/Untitled%2038.png](CS231n%20LEC06%203fb2279d00804036b36489fa2286428a/Untitled%2038.png)

    ![CS231n%20LEC06%203fb2279d00804036b36489fa2286428a/Untitled%2039.png](CS231n%20LEC06%203fb2279d00804036b36489fa2286428a/Untitled%2039.png)

    ![CS231n%20LEC06%203fb2279d00804036b36489fa2286428a/Untitled%2040.png](CS231n%20LEC06%203fb2279d00804036b36489fa2286428a/Untitled%2040.png)

    ![CS231n%20LEC06%203fb2279d00804036b36489fa2286428a/Untitled%2041.png](CS231n%20LEC06%203fb2279d00804036b36489fa2286428a/Untitled%2041.png)

    ![CS231n%20LEC06%203fb2279d00804036b36489fa2286428a/Untitled%2042.png](CS231n%20LEC06%203fb2279d00804036b36489fa2286428a/Untitled%2042.png)

    ![CS231n%20LEC06%203fb2279d00804036b36489fa2286428a/Untitled%2043.png](CS231n%20LEC06%203fb2279d00804036b36489fa2286428a/Untitled%2043.png)

    ![CS231n%20LEC06%203fb2279d00804036b36489fa2286428a/Untitled%2044.png](CS231n%20LEC06%203fb2279d00804036b36489fa2286428a/Untitled%2044.png)

    ![CS231n%20LEC06%203fb2279d00804036b36489fa2286428a/Untitled%2045.png](CS231n%20LEC06%203fb2279d00804036b36489fa2286428a/Untitled%2045.png)

- Hyperparameter Optimization.

    ![CS231n%20LEC06%203fb2279d00804036b36489fa2286428a/Untitled%2046.png](CS231n%20LEC06%203fb2279d00804036b36489fa2286428a/Untitled%2046.png)

    Cross-validation은 Training Set으로 학습하고 Validation Set으로 평가한다.

    1. Coarse(First)는 넓은 범위에서 값을 골라, 적은 Epoch로도 잘 작동하는지 확인.
    2. Fine(Second)는 보다 좁은 범위에서 학습을 지켜보면서 Parameter의 최적값을 찾는다.

    1. Run Coarse Search.

        ![CS231n%20LEC06%203fb2279d00804036b36489fa2286428a/Untitled%2047.png](CS231n%20LEC06%203fb2279d00804036b36489fa2286428a/Untitled%2047.png)

        Coarse를 하면서 빨간색 네모와 같이 Fine을 시작할 범위를 찾는다.

        Hyper Parameter 최적화에는 Log를 쓰는 것이 좋음.

    2. Now Run Finer Search.

        ![CS231n%20LEC06%203fb2279d00804036b36489fa2286428a/Untitled%2048.png](CS231n%20LEC06%203fb2279d00804036b36489fa2286428a/Untitled%2048.png)

        약 53%의 정확도를 갖는다.

        범위가 변하였고 Learning Rate의 최적값이 바뀐 범위의 경계 부분에 집중되어 있다.

        이런 경우 범위를 조금만 더 수정하여도 더 좋은 범위가 있을 수 있고, 최적의 값이 범위 중간에 오는 것이 좋다.

    3. Grid Search.

        ![CS231n%20LEC06%203fb2279d00804036b36489fa2286428a/Untitled%2049.png](CS231n%20LEC06%203fb2279d00804036b36489fa2286428a/Untitled%2049.png)

        고정된 값과 간격으로 샘플링하는 방법이다.

        찾을 수 없는 구간이 생길 수 있다는 단점이 있다.

    ![CS231n%20LEC06%203fb2279d00804036b36489fa2286428a/Untitled%2050.png](CS231n%20LEC06%203fb2279d00804036b36489fa2286428a/Untitled%2050.png)

    ![CS231n%20LEC06%203fb2279d00804036b36489fa2286428a/Untitled%2051.png](CS231n%20LEC06%203fb2279d00804036b36489fa2286428a/Untitled%2051.png)

    ![CS231n%20LEC06%203fb2279d00804036b36489fa2286428a/Untitled%2052.png](CS231n%20LEC06%203fb2279d00804036b36489fa2286428a/Untitled%2052.png)

    Loss Curve를 확인함에 있어서 Learning Rate가 중요하다.

    1. Loss가 발산하는 경우, LR가 너무 높다.
    2. Loss에 정체기가 생기는 경우, LR가 높다.
    3. Loss가 Linear한 경우, LR가 너무 낮다.
    4. Loss가 지속적으로 잘 내려가는 경우가 좋다.

    ![CS231n%20LEC06%203fb2279d00804036b36489fa2286428a/Untitled%2053.png](CS231n%20LEC06%203fb2279d00804036b36489fa2286428a/Untitled%2053.png)

    Loss가 가파르게 내려가는 경우, 초기화에 문제가 있을 수 있다.

    Graident의 Back Prop가 초기에 안 되다가 학습이 진행되면서 회복된다.

    ![CS231n%20LEC06%203fb2279d00804036b36489fa2286428a/Untitled%2054.png](CS231n%20LEC06%203fb2279d00804036b36489fa2286428a/Untitled%2054.png)

    Training Acc와 Validation Acc가 큰 차이를 보이면, Overfitting의 가능성이 높다.

    Regularization Strength를 높인다.

    ![CS231n%20LEC06%203fb2279d00804036b36489fa2286428a/Untitled%2055.png](CS231n%20LEC06%203fb2279d00804036b36489fa2286428a/Untitled%2055.png)

    Weight Updates / Weight Magnitudes의 비율을 봤을 때 0.001 정도가 좋다.

### Summary.

![CS231n%20LEC06%203fb2279d00804036b36489fa2286428a/Untitled%2056.png](CS231n%20LEC06%203fb2279d00804036b36489fa2286428a/Untitled%2056.png)

### 링크.

[https://www.youtube.com/watch?v=wEoyxE0GP2M&t=2821s](https://www.youtube.com/watch?v=wEoyxE0GP2M&t=2821s)

[CS231n: Convolutional Neural Networks for Visual Recognition](http://cs231n.stanford.edu/2017/)

[](http://cs231n.stanford.edu/slides/2017/cs231n_2017_lecture6.pdf)

[soline013/CS231N_17_KOR_SUB](https://github.com/soline013/CS231N_17_KOR_SUB/blob/master/kor/Lecture%206%20%20%20Training%20Neural%20Networks%20I.ko.srt)
