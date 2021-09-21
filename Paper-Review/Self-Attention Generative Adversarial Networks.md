# Self-Attention Generative Adversarial Networks.

[Self-Attention Generative Adversarial Networks](https://arxiv.org/pdf/1805.08318v2.pdf)

## Abstract.

- Self Attention Generative Adversarial Network(SAGAN)은 Attention-driven, Long-Range Dependency Modeling을 이용한다.
- 전통적인 Convolutional GANs는 저해상도 Feature Maps에서 Local Points 함수만을 사용하여 고해상도의 디테일을 만든다.
- SAGAN에서는 모든 Feature Locations로부터 힌트를 얻어 디테일을 만든다.
- Discriminator는 이미지에서 멀리있는 디테일들이 서로 일치하는지 확인할 수 있다.
- Generator Conditioning이 GAN의 성능에 영향을 미친다는 것이 최근 연구로 나타나, 해당 논문에서는 Spectral Normalization을 Generator에 적용했다.
- ImageNet Dataset에서 Best Published Inception Score가 36.8에서 52.52로 늘었고, Frechet Inception Distance가 27.62에서 18.65로 줄었다.
- Attention의 시각화는 Generator가 고정된 Local Region보다 Object와 일치하는 정보를 사용한다.

## 1. Introduction.

1. Image Synthesis는 많은 Open Problem이 있지만 Deep Convolutional Network 기반의 GAN이 등장하면서 많이 발전하였다.
2. 그러나 Convolutional GAN로 만든 샘플을 조사한 결과, Convolutional GAN은 ImageNet과 같은 Multi-class Dataset을 훈련할 때, 몇몇 이미지 Class에서 모델링에 어려움을 겪었다.
3. 예를 들어, SOTA ImageNet GAN은 몇몇 구조적 제한(기하학적 특징보다 텍스쳐로 구분되는 바다, 하늘, 풍경 Class)을 가진 이미지 합성에는 좋지만, 몇 개 Class에서 일정하게 발생하는 기하학적이나 구조적 패턴을 잡아내는 데는 그리 좋지 않다.(개는 현실과 같은 털 질감을 갖지만, 명확하게 구분된 발이 없는 경우이다.)
4. GAN은 다른 이미지 영역 사이에서 Dependency를 모델링할 때, Convolution에 크게 의존한다.
5. Convolution은 Local Receptive Field를 갖기 때문에, Long Range Dependency는 Convolution Layer를 몇 개 거쳐야 처리된다.
6. 따라서 Long Term Dependency 학습을 여러 이유로 방해한다.
    1. 작은 모델은 Dependency를 대표할 수 없다.
    2. Optimzation은 Dependency를 잘 표현하는, 여러 Layer를 조정하는 Parameter를 발견하는 데 어려움을 겪을 수 있다.
    3. Parameterization은 새로운 입력에 적용할 때, 통계적으로 취약하고 실패하기 쉽다.
    4. Kernel Size를 늘리면 Representational Capacity가 증가할 수 있지만, Local Conv를 사용하여 얻는 효율성이 떨어진다.

### 번외. Local Receptive Field, Long Range Dependency(Long Term Dependency)

![SAGAN 0](https://user-images.githubusercontent.com/66259854/124282235-944ade80-db85-11eb-9606-cfd36bd1b78f.png)

[Understanding the receptive field of deep convolutional networks | AI Summer](https://theaisummer.com/receptive-field/)

1. Local Receptive Field는 국부 수용영역으로, "the size of the region in the input that produces the feature"이다. 즉, Layer 1에서 3X3, 5X5의 영역 등을 의미한다.
2. 그렇다면 Long Range Dependency는 하나의 Layer가 아닌, Layer 1 ~ Layer 3에 걸쳐 일어날 것이다. 초록색 영역과 노란색 영역이 Layer 3에서 처리될 것이다.

---

<img width="802" alt="SAGAN1" src="https://user-images.githubusercontent.com/66259854/124282242-9614a200-db85-11eb-9b99-5421ad44e521.png">

Self-attention은 Long Range Dependency와 Compuational Cost 사이에서 Balance가 좋다.

Self-attention을 GAN에 도입한다.

## 2. Related Work.

1. GAN
2. Attention

## 3. Self-Attention Generative Adversarial Networks.

대부분의 GAN은 Convolution Layer를 사용한다.

Conv Layer는 Local Neighborhood에 대한 정보를 제공하므로, Long-range Dependency를 모델링하는 것은 비효율적이다.

따라서 Non-local Model을 적용한다.

---

<img width="802" alt="SAGAN4" src="https://user-images.githubusercontent.com/66259854/124282246-9745cf00-db85-11eb-8869-7f8364da9ed9.png">

Image Feature $x \in \mathbb{R}^{C \times N}$은 Attention 계산을 위해 $f(x) = W_f x, \ g(x) = W_g x$ 로 변환된다.

1. C: 채널의 수
2. N: 이전 Hidden Layer에서 Feature Locations의 수

---

$$\beta_{j,i} = \frac{\text{exp}(s_{ij})} {\sum^N_{i=1} {\text{exp}(s_{ij})}}, \ \text{where} \ s_{ij} = f(x_i)^T g(x_j),$$

$\beta_{j, i}$는 jth Region을 합성할 때, ith Location에 얼마나 집중하는지 나타낸다.

$f(x_i)$는 Transpose 되고, 두 Feature Space는 행렬곱 이후 Softmax를 지나 Attention Map으로 들어간다.

---

$$o_j = v(\sum^N_{i=1} \beta_{j, i} h(x_i)), \ h(x_i) = W_hx_i, \ v(x_i) = W_v x_i.$$

$o = (o_1, \, o_2, \, ..., \, o_j, \, ..., \, o_N) \in \mathbb{R}^{C \times N}$: Output of Attention Layer

$h(x_i)$와 $\beta_{j, i}$가 행렬곱되고, $v(x_i)$를 통해 Output $o_j$가 도출된다.

---

- $W_g \in \mathbb{R}^{\bar{C} \times C}, \ W_f \in \mathbb{R}^{\bar{C} \times C}, \ W_h \in \mathbb{R}^{\bar{C} \times C}, \ W_v \in \mathbb{R}^{\bar{C} \times C}$는 모두 학습된 Weight 행렬로, 1X1 Convolution Layer이다.
- ImageNet에서 채널의 수 $\bar{C}$를 $C/k, \ (k = 1,2,4,8)$로 줄여도 큰 성능 저하가 없었기 때문에, 메모리 효율화를 위해 $k=8$로 정하였다.
- 최종 Output은 $y_i = \gamma o_i + x_i$로, $o_i$에 Scale Parameter $\gamma$를 곱하고, Input Feature Map $x_i$를 더하였다.
- $\gamma$는 학습 가능한 값으로 0으로 초기화된다.
- Network는 처음에 Local Neighborhoods에서 힌트를 얻지만, $\gamma$가 학습되면서 점점 Non-local에 초점을 맞춘다.

---

![image](https://user-images.githubusercontent.com/66259854/133993840-b6e88334-7ecc-4e0f-b2e8-8dfa5cc3276b.png)

SAGAN은 Attention이 Generator와 Discriminator 모두에 적용된다.

Generator와 Discriminator는 Adversarial Loss의 Hinge Version을 최소화함으로써, 번갈아가며 학습한다.

### 번외. Adversarial Loss.

![image](https://user-images.githubusercontent.com/66259854/134179050-48f898db-a283-4653-b767-4fbb6f4d4010.png)

$x \sim P_{data(x)}$: 실제 데이터의 분포

$z \sim p_z(z)$: 분포가정(e.g. 정규분포)에서 Latent Code의 분포

GAN의 판별자 D는 Real or Fake를 판단하기 때문에, Binary Cross Entropy를 사용한다. Real일 때 y=1, Fake일 때 y=0이다.

$BCE = - \frac{1}{n} \sum^n_{i=1} (y_i \, \text{log}(p_i) +(1-y_i) \text{log}(1-p_i))$를 사용한 Loss이다.

## 4. Techniques to Stabilize the Training of GANs.

1. Spectral Normalization For Both Generator and Discriminator

    각 층에서 Spectral Norm을 제한하여, Discirminator의 Lipschitz Constant를 억제한다.

    Spectral Normalization은 추가적인 Hyper Parameter 튜닝이 필요하지 않다.

    Generator에도 Spectral Norm을 적용한다.

    ### 번외. Spectral Norm, Lipschitz Constant.

    [Spectral Normalization for Generative Adversarial Networks(SN-GAN) - (2)](https://hichoe95.tistory.com/61)

2. Imbalanced Learning Rate For Generator and Discriminator Updates

    정규화된 Discriminator는 학습이 느린데, Generator Update Step당 보통 Multiple Discriminator Update Step이 필요하다.

    "GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium" 논문은 TTUR(Two Time-scale Update Rule For Training GAN)을 소개하고, 이 논문이 TTUR을 사용한다.

    TTUR을 사용하여 정규화된 Discriminator에서 학습을 조금 빠르게 한다.

    ### 번외. TTUR.

    [TTUR Paper](https://arxiv.org/pdf/1706.08500.pdf)

## 5. Experiments.

LSVRC2012(ImageNet) Dataset을 사용한다.

TTUR을 평가하고, Self-attention에 대해 평가하고, SAGAN을 다른 SOTA 모델과 비교한다.

4개의 GPU를 사용하여 2주의 시간이 소요되었다.

1. Evaluation Metrics

    Inception Score(IS)와 Frechet Inception Distance(FID)가 정량적인 평가에 사용되었다.

    - 번외. IS, FID.

        [GAN 평가지표(IS:Inception Score/FID:Frechet Inception Distance)](https://m.blog.naver.com/chrhdhkd/222013835684)

        [GAN의 평가와 편향](https://velog.io/@tobigs-gm1/evaluationandbias)

2. 모든 SAGAN은 128 X 128 크기의 이미지를 만들어낸다.

    Sperctral Norm과 Conditional Batch Norm을 Generator와 Discriminator에 사용한다.

    Optimizer로 Adam을 사용하고, $\beta_1 = 0 \ and \ \beta_2 = 0.9$이다.

    LR은 Gererator는 0.0001, Discriminator는 0.0004이다.

3. SN, TTUR

    <img width="802" alt="SAGAN2" src="https://user-images.githubusercontent.com/66259854/124282253-990f9280-db85-11eb-94bc-aebefcb4459d.png">

    <img width="802" alt="SAGAN3" src="https://user-images.githubusercontent.com/66259854/124282257-99a82900-db85-11eb-8a79-d0ad7f6eecf5.png">


4. Self-attention

    <img width="802" alt="SAGAN5" src="https://user-images.githubusercontent.com/66259854/124282274-9d3bb000-db85-11eb-897e-d9c436601ecf.png">

    <img width="802" alt="SAGAN6" src="https://user-images.githubusercontent.com/66259854/124282276-9dd44680-db85-11eb-9ff5-992a881fa13b.png">


5. SOTA

    <img width="802" alt="SAGAN7" src="https://user-images.githubusercontent.com/66259854/124282280-9e6cdd00-db85-11eb-946d-612baa116017.png">

    <img width="802" alt="SAGAN8" src="https://user-images.githubusercontent.com/66259854/124282283-9f057380-db85-11eb-9828-7c7e57395aab.png">

## 6. Conclusion.

1. Long Range Dependency

    Self-attentiion을 GAN에 적용한다.

2. Introduction에서 언급한 Multi-class Dataset을 훈련할 때 얻는 어려움.

    Spectral Normalization을 Generator와 Discriminator에 사용한다.

    TTUR로 정규화된 Discriminator의 학습 속도를 높인다.

## Link.

[[논문 리뷰] Self-Attention Generative Adversarial Networks](https://simonezz.tistory.com/77)
