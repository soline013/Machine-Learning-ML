---

## Abstract.

- EfficientNet은 PMLR 학회에서 2019년에 발표된 논문이다.
- CNN에서 정확도를 높이기 위해 사용한 방법: Fixed Resource Budget
    1. Depth: 모델의 깊이, Layer 수를 늘린다.
    2. Width: 너비, Filter(or Channel) 수를 늘린다.
    3. Resolution: 입력 이미지의 해상도(크기), Input Image의 크기를 키운다.
    
- 논문에서 제시하는 CNN의 정확도를 높이기 위한 방법: Compound Coefficient & EfficientNet
    1. Depth, Width, Resolution의 균형이 더 좋은 성능을 발휘한다.
    2. Compound Coefficient를 통해 Depth, Width, Resolution를 Compound Scaling 한다.
    3. NAS(Neural Architecture Search)를 사용한 새로운 Baseline Network, EfficientNet을 제안한다.
    4. EfficientNet은 더 정확하고 효율적(적은 파라미터 수)이다.
    5. EfficientNet-B7은 ImageNet에서 84.3% Top-1 Acc를 달성했다. 기존 ConvNet에 비해 8.4배 작고, 6.1배 빠르다. CIFAR-100(91.7%), Flowers(98.8%), 3 Other Transfer Learning Datasets → SOTA
    
    ![스크린샷 2021-10-09 22.55.56.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/8455e113-4c99-42b7-ab0d-1ad34c9ee2c8/스크린샷_2021-10-09_22.55.56.png)
    

## 1. Introduction.

1. CNN에서 정확도를 높이기 위해 사용한 방법
    
    ![스크린샷 2021-10-09 22.56.37.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/3fbcc6c7-684c-47e4-8e9c-cf45fb49efb1/스크린샷_2021-10-09_22.56.37.png)
    
    1. (a)를 Baseline으로 잡았을 때, (b), (c), (d)는 모두 하나의 CNN의 정확도를 높이기 위한 방법을 사용한 것이다.
    2. (b): 너비, Filter(or Channel) 수를 늘린다. Wide Residual Network, Channel을 2배씩 늘렸다.
    3. (c): 모델의 깊이, Layer 수를 늘린다. ResNet-18부터 ResNet-200까지 Layer를 늘렸다.
    4. (d): 입력 이미지의 해상도(크기), Input Image의 크기를 키운다. GPipe, Resolution을 2배씩 늘렸다.
    5. (e): 번외로, (b), (c), (d)의 방법을 모두 사용하였다.
    
2. Compound Scaling Method
    1. Depth, Width, Resolution의 균형은 각각을 일정한 비율로 증가시키면 된다.
    2. 고정된 계수 세트를 사용하는 Compound Scaling을 제안한다.
    3. Depth($\alpha^N$), Width($\beta^N$), Image Size($\gamma^N$)
    4. $\alpha, \beta, \gamma$는 Constant Coeffcients로, 기존 Small Model에서 Grid Search를 통해 결정된다.
    
3. EfficientNet
    1. 기존 MobileNet과 ResNet에 Compound Scaling Method가 잘 작동한다는 것을 실험을 통해 알 수 있었다.
    2. Compound Scaling Method의 효과는 Baseline Network에 따라 크게 달라진다.
    

## 2. Related Work.

1. ConvNet Accuracy.
2. ConvNet Efficiency.
3. Model Scaling.

## 3. Compound Model Scaling.

- 3.1. Problem Formulation.
    1. i번 Conv Layer는 $Y_i = F_i (X_i)$로 표현할 수 있다.
        - $F_i$: Layer Operator, Layer 연산을 의미.
        - $X_i, Y_i$: Input, Output, Tensor Shape $<H_i, W_i, C_i>$
        
    2. 위의 Conv Layer 표현을 이용하여 CNN을 표현할 수 있다.
        
        $$N = \bigodot_{i=1...s} F^{L_i}_{i}(X_{<H_i, W_i, C_i>})$$
        
        - $F^{L_i}_{i}$: $F_i$가 $L_i$만큼 반복.
        - Conv Layer를 거치면 Spatial Dimension(H, W)는 작아지고 Channel Dimension이 증가한다.
        
    3. Optimization Problem.
        - 일반적인 CNN 디자인은 가장 좋은 성능의 $F_i$를 찾는 것이다.
        - 하지만 Model Scaling은 $H_i, W_i, C_i, L_i$를 확장하고 조절한다.
        - $F_i$를 고정하여 문제를 단순화하였으나, 각 Layer에서 다른 $H_i, W_i, C_i, L_i$를 찾아야 하는 넓은 Design Space가 남는다.
        - Design Space를 줄이기 위해, 모든 Layer의 비율을 일정하게 조정하도록 제안한다.
        - Resource(GPU 성능)가 제한된 환경에서 Model 정확도를 최대화 하는 것이 목표이다.
        - 정확도를 높이면서, 연산량은 최대한 줄인다.
        
        $$\underset{d,w,r}{\text{max}} \quad Accuracy(N(d, w, r)) \\ s.t. \quad N(d,w,r) = \bigodot_{i=1...s} \hat{F}^{d \cdot \hat{L}_i}_i (X_{<r \cdot \hat{H}_i, \ r \cdot \hat{W}_i, \ w \cdot \hat{C}_i>}) \\ \text{Memory}(N) \le \text{target\_memory} \\ \text{FLOPS}(N) \le \text{target\_flops}$$
        
        - $w, d, r$: Scaling을 위한 계수.
        - $H_i, W_i, C_i, L_i, F_i$: Baseline Network에서 미리 정의한 파라미터들.
        
- 3.2. Scaling Dimensions.
    1. Difficulty of Optimization Problem
        1. $w, d, r$의 최적 값이 서로 의존적이고, 리소스 제약 조건 아래에서 값이 변화한다.
        2. 따라서 기존 방법으로는 이중 하나만 사용하여 Scaling 하는 경우가 많다.
        
    
    ![스크린샷 2021-10-09 22.56.48.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/8d812f89-9fee-4e88-a1e4-0d69371f6728/스크린샷_2021-10-09_22.56.48.png)
    
    1. Depth: 모델의 깊이, Layer 수를 늘린다.
        1. Intuition: Deep Network는 Richer, Complex한 Feature를 포착할 수 있다.
        2. 하지만 깊어질수록 Gradient Vanishing이 발생하여 학습에 어려움이 생긴다.
        3. Gradient Vanishing 해결을 위해 Skip Connections과 Batch Norm를 사용한 ResNet이 등장했지만, 101과 1000은 정확도가 비슷하며 여전히 문제가 존재한다.
        4. 가운데 그래프는 Depth만을 증가시킨 성능으로, d=6.0에서 증가폭이 거의 사라진다.
        
    2. Width: 너비, Filter(or Channel) 수를 늘린다.
        1. Intuition: Wider Network는 Fine-grained Feature를 포착할 수 있고, Deep Network에 비해 학습이 쉽다.
        2. 하지만 Depth가 작은 Network는 Richer, Complex한 Feature를 포착하기 어렵다.
        3. 좌측 그래프는 Width만을 증가시킨 성능으로, w=3.8에서 증가폭이 감소한다.
        
    3. Resolution: 입력 이미지의 해상도(크기), Input Image의 크기를 키운다.
        1. Intuition: Higher Resolution은 Fine-grained Patterns을 포착할 수 있다.
        2. 초기 해상도는 224X224 였으나, 논문 당시 600X600에서 학습이 가능했다.
        3. 우측 그래프는 Resolution만을 증가시킨 성능으로, 가장 증가폭의 감소가 더디다.
    
    1. Observation 1.
        
        > Scaling up any dimension of network width, depth, or resolution improves accuracy, but the accuracy gain diminishes for bigger models.
        > 
    
- 3.3. Compound Scaling.
    1. Figure 4.
        
        ![스크린샷 2021-10-09 22.57.12.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/33146d69-748c-45e0-8230-16706a1757d4/스크린샷_2021-10-09_22.57.12.png)
        
        1. 3.2.를 통해 Different Scaling Dimension은 독립적이지 않다는 걸 알 수 있다.
        2. 따라서 모든 차원을 Scaling 하여 균형을 맞춰야 한다.
        3. 위 그래프는 Depth와 Resolution, 두 차원을 Scaling한 결과이다.
        4. Width를 고정하고 Depth와 Resolution 각각 다른 비율로 증가시킨 경우, 가장 좋은 성능을 보인다.
        
    2. Observation 2.
        
        > In order to pursue better accuracy and efficiency, it is critical to balance all dimensions of network width, depth, and resolution during ConvNet scaling.
        > 
        
    3. Compound Scaling Method.
        
        $$\text{depth}: d = \alpha^{\phi} \\ \text{width}: w = \beta^{\phi} \\ \text{resolution}: r = \gamma^{\phi} \\ s.t. \ \alpha \cdot \beta^2 \cdot \gamma^2 \approx 2 \\ \alpha \ge 1, \beta \ge 1, \gamma \ge 1$$
        
        1. Compound Coeffcient $\phi$를 사용하여 w, d, r을 Scaling 한다.
        2. $\alpha, \beta, \gamma$는 Small Grid Search로 결정되는 상수이다.
        3. Compound Coeffcient $\phi$는 리소스에 따라 사용자가 조절할 수 있다.
        4. Depth를 2배 하면 FLOPS는 2배 증가하지만, Width나 Resolution을 2배하면 FLOPS는 4배 증가하기 때문에, $\alpha \cdot \beta^2 \cdot \gamma^2 \approx 2$로 설정하였다.
        5. 이러한 Total FLOPS 설정은 어떤 $\phi$를 사용해도 $2^\phi$ 만큼 증가하도록 한다.
        

## 4. EfficientNet Architecture.

![스크린샷 2021-10-09 22.57.49.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/4352bc3d-ee84-44d3-8b59-0fb0a8bb8720/스크린샷_2021-10-09_22.57.49.png)

- Model Scaling으로 위에서 $F_i$를 고정하였기 때문에, 좋은 Baseline Network가 중요하다.
- Compound Scaling을 기존 CNN에도 적용하지만, 새로운 Mobile-size Baseline을 설계하였다.
- EfficientNet-B0은 가장 Base한 Network로 위와 같은 구조를 가진다.
- ResNet의 Bottlenect은 Channel을 마지막에서 4배 증가시키지만, Inverted Bottleneck은 중간에 증가시키고 마지막에 감소시킨다.

*MBConv: Mobile Inverted Bottleneck.

---

<aside>
👣 STEP 1: we first fix $\phi$ = 1, assuming twice more resources available, and do a small grid search of $\alpha$, $\beta$, $\gamma$ based on Equation 2 and 3. In particular, we find the best values for EfficientNet-B0 are $\alpha$ = 1.2, $\beta$ = 1.1, $\gamma$ = 1.15, under constraint of $\alpha \cdot \beta^2 \cdot \gamma^2 \approx 2$.

</aside>

<aside>
👣 STEP 2: we then fix $\alpha$, $\beta$, $\gamma$ as constants and scale up
baseline network with different $\phi$ using Equation 3, to
obtain EfficientNet-B1 to B7 (Details in Table 2)

</aside>

1. Compound Coefficient $\phi$를 1로 고정하고 $\alpha$, $\beta$, $\gamma$를 구한다.
2. $\alpha$, $\beta$, $\gamma$를 고정하고 서로 다른 $\phi$로 Scaling 하여, EfficientNet-B1 to B7을 생성한다.

---

- 번외 이미지.
    
    ![다운로드.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/e8ccf575-3563-4607-9270-28e11359f0ba/다운로드.png)
    
    ![다운로드 (1).png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/60cf515a-0bf3-48f9-bdfc-20e46e4bbfff/다운로드_(1).png)
    

## 5. Experiments.

### 5.1. Scaling Up MobileNets and ResNets.

![스크린샷 2021-10-09 22.58.30.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/f11b45cf-d46c-443f-b1f2-a94afe28596f/스크린샷_2021-10-09_22.58.30.png)

- MoblieNets과 ResNets에 Compound Scaling을 적용하였다.
- 기존 FLOPS와 크게 달라지지 않았지만, 정확도가 높아졌다.
- Baseline Network의 중요성을 볼 수 있다.

### 5.2. ImageNet Results for EfficientNet.

![스크린샷 2021-10-09 22.58.10.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/ce583922-a572-463d-a7f0-13d780989880/스크린샷_2021-10-09_22.58.10.png)

- 파라미터 수(FLOPS) 별로 모델을 정렬하였다.
- EfficientNet-B7은 기존 SOTA인 GPipe와 동일한 성능을 보여주지만, 파라미터수는 약 8.4배 적다.

---

![스크린샷 2021-10-09 22.59.02.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/2edaec06-6b57-44d7-adc6-74947acbc035/스크린샷_2021-10-09_22.59.02.png)

- Abstract의 Figure 1.과 유사한 그래프이다.
- Figure 1.은 Number of Parameters로, Figure 5.는 FLOPS로 비교하였다.
- EfficientNet이 높은 정확도와 적은 연산량을 보여준다.

---

![스크린샷 2021-10-09 22.58.44.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/979092ec-efd4-4b42-9999-20b90896e668/스크린샷_2021-10-09_22.58.44.png)

- Batch Size 1, Single Core ...으로 비교한 지연 시간 결과이다.
- EfficientNet-B7이 낮은 지연 시간을 보여준다.

### 5.3. Transfer Learning Results for EfficientNet.

![스크린샷 2021-10-09 22.59.19.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/f0514c34-bc89-48cf-8a8a-70dfacc311ec/스크린샷_2021-10-09_22.59.19.png)

- Transfer Learning의 성능 비교 결과이다.
- 적은 파라미터 수로 비슷하거나 더 나은 정확도를 나타낸다.

---

![스크린샷 2021-10-09 22.59.43.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/3c533f38-8497-4108-8ede-4c5f02d98d4b/스크린샷_2021-10-09_22.59.43.png)

- Transfer Learning의 성능을 그래프로 비교한 것이다.
- 빨간 선이 EfficientNet의 정확도이고, 다른 Model은 각자 다른 기호를 가지고 있다.

---

![스크린샷 2021-10-09 23.00.18.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/7a16bbc5-7463-4318-8031-5e65cd4f19d3/스크린샷_2021-10-09_23.00.18.png)

- Transfer Learning에 사용한 Dataset이다.

## 6. Discussion.

![스크린샷 2021-10-09 23.00.40.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/79699a6d-c427-4c59-bd2b-41ecd68f75c1/스크린샷_2021-10-09_23.00.40.png)

![스크린샷 2021-10-09 23.00.53.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/f8079248-bbf3-43ed-810c-640abb89033e/스크린샷_2021-10-09_23.00.53.png)

- Baseline인 EfficientNet-B0에 대해 각각 다른 Scaling을 적용한 결과이다.

---

![스크린샷 2021-10-09 23.00.01.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/d75b265c-e137-4b4f-b7d4-e14be1a250e7/스크린샷_2021-10-09_23.00.01.png)

- CAM은 이미지에서 어떤 부분이 활성화 되었는지 확인할 수 있다.
- 빨간색에 가까울수록 해당하는 Class가 강하게 반응하고, 파란색에 가까울수록 약하게 반응한다.

## 7. Conclusion.

- Compound Scaling.
- EfficientNet.

## Acknowledgements.

> We thank Ruoming Pang, Vijay Vasudevan, Alok Aggarwal, Barret Zoph, Hongkun Yu, Jonathon Shlens, Raphael Gon- tijo Lopes, Yifeng Lu, Daiyi Peng, Xiaodan Song, Samy Bengio, Jeff Dean, and the Google Brain team for their help.
> 

## Appendix.

![스크린샷 2021-10-09 23.51.34.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/6b926f29-d1a5-4578-9503-ff8b5d7793d4/스크린샷_2021-10-09_23.51.34.png)

## Link.

[EfficientNet : Rethinking Model Scaling for Convolutional Neural Networks 논문 리뷰](https://ropiens.tistory.com/110)