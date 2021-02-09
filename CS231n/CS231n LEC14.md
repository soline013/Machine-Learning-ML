# 이미지는 수정 예정.

# CS231n LEC14.
## Stanford University CS231n, Spring 2017.
**Deep Reinforcement Learning.**

## Recall from last time.

![CS231n%20LEC14%20d66c599e581d40669654e69af238e10c/473b6022-aa43-484c-95b7-92301872df6c.pdf-0005.png](CS231n%20LEC14%20d66c599e581d40669654e69af238e10c/473b6022-aa43-484c-95b7-92301872df6c.pdf-0005.png)

![CS231n%20LEC14%20d66c599e581d40669654e69af238e10c/473b6022-aa43-484c-95b7-92301872df6c.pdf-0006.png](CS231n%20LEC14%20d66c599e581d40669654e69af238e10c/473b6022-aa43-484c-95b7-92301872df6c.pdf-0006.png)

## Overview.

![CS231n%20LEC14%20d66c599e581d40669654e69af238e10c/473b6022-aa43-484c-95b7-92301872df6c.pdf-0008.png](CS231n%20LEC14%20d66c599e581d40669654e69af238e10c/473b6022-aa43-484c-95b7-92301872df6c.pdf-0008.png)

## Reinforcement Learning.

![CS231n%20LEC14%20d66c599e581d40669654e69af238e10c/473b6022-aa43-484c-95b7-92301872df6c.pdf-0007.png](CS231n%20LEC14%20d66c599e581d40669654e69af238e10c/473b6022-aa43-484c-95b7-92301872df6c.pdf-0007.png)

![CS231n%20LEC14%20d66c599e581d40669654e69af238e10c/473b6022-aa43-484c-95b7-92301872df6c.pdf-0013.png](CS231n%20LEC14%20d66c599e581d40669654e69af238e10c/473b6022-aa43-484c-95b7-92301872df6c.pdf-0013.png)

강화 학습은 Agent의 보상을 최대화할 수 있는 행동을 학습한다.

1. Agent는 환경이 제공한 상태 $s_t$에서 행동 $a_t$를 취한다.
2. 행동에 따른 보상 $r_t$, 다음 상태 $s_{t+1}$를 제공한다.
3. Agent가 종료 상태(Terminal State)가 될 때까지 반복한다.

## Example.

![CS231n%20LEC14%20d66c599e581d40669654e69af238e10c/473b6022-aa43-484c-95b7-92301872df6c.pdf-0014.png](CS231n%20LEC14%20d66c599e581d40669654e69af238e10c/473b6022-aa43-484c-95b7-92301872df6c.pdf-0014.png)

![CS231n%20LEC14%20d66c599e581d40669654e69af238e10c/473b6022-aa43-484c-95b7-92301872df6c.pdf-0015.png](CS231n%20LEC14%20d66c599e581d40669654e69af238e10c/473b6022-aa43-484c-95b7-92301872df6c.pdf-0015.png)

![CS231n%20LEC14%20d66c599e581d40669654e69af238e10c/473b6022-aa43-484c-95b7-92301872df6c.pdf-0016.png](CS231n%20LEC14%20d66c599e581d40669654e69af238e10c/473b6022-aa43-484c-95b7-92301872df6c.pdf-0016.png)

![CS231n%20LEC14%20d66c599e581d40669654e69af238e10c/473b6022-aa43-484c-95b7-92301872df6c.pdf-0017.png](CS231n%20LEC14%20d66c599e581d40669654e69af238e10c/473b6022-aa43-484c-95b7-92301872df6c.pdf-0017.png)

모두 다른 Objective, State, Action, Reward가 있다.

## Mathematically Formalize.

## Markov Decision Process.

![CS231n%20LEC14%20d66c599e581d40669654e69af238e10c/473b6022-aa43-484c-95b7-92301872df6c.pdf-0019.png](CS231n%20LEC14%20d66c599e581d40669654e69af238e10c/473b6022-aa43-484c-95b7-92301872df6c.pdf-0019.png)

MDP로 강화 학습을 수식화할 수 있다.

Markov Property는 현재 상태만으로 모든 상태를 나타낸다.

1. $S$: 가능한 상태들의 집합.
2. $A$: 가능한 행동들의 집합.
3. $R$: (State, Action)에서 받는 보상의 분포.
4. $\mathbb{P}$: (State, Action)에서 다음 상태의 분포, 전이확률(Transition Probability).
5. $\gamma$: Discount Factor, 미래의 가치에 대한 현재의 가치, 0~1의 값이다. [DQN](https://www.notion.so/DQN-80f2c4dace9643bc8c241418a3b5ecfa)

![CS231n%20LEC14%20d66c599e581d40669654e69af238e10c/473b6022-aa43-484c-95b7-92301872df6c.pdf-0020.png](CS231n%20LEC14%20d66c599e581d40669654e69af238e10c/473b6022-aa43-484c-95b7-92301872df6c.pdf-0020.png)

1. 환경은 초기 상태 분포 $P(s_0)$에서 상태 $s_0$를 Sampling 한다.
2. Time Step $t=0$부터 종료 상태까지 다음을 반복한다.
    1. Agent가 행동 $a_t$를 선택한다.
    2. 환경은 $R$에서 보상 $r_t$를 Sampling 한다.
    3. 환경은 $\mathbb{P}$에서 다음 상태 $s_{t+1}$를 Sampling 한다.
    4. Agent가 보상과 다음 상태를 받는다.

정책 $\pi$는 각 상태에서 Agent가 어떤 행동을 할지 명시한다.

최적의 정책 $\pi^*$은 Cumulative Discounted Reward를 최대화하는 것이다.

보상에는 미래의 보상도 포함되며 Discount Factor의 영향을 받는다.

![CS231n%20LEC14%20d66c599e581d40669654e69af238e10c/473b6022-aa43-484c-95b7-92301872df6c.pdf-0021.png](CS231n%20LEC14%20d66c599e581d40669654e69af238e10c/473b6022-aa43-484c-95b7-92301872df6c.pdf-0021.png)

![CS231n%20LEC14%20d66c599e581d40669654e69af238e10c/473b6022-aa43-484c-95b7-92301872df6c.pdf-0022.png](CS231n%20LEC14%20d66c599e581d40669654e69af238e10c/473b6022-aa43-484c-95b7-92301872df6c.pdf-0022.png)

Grid World에서 격자는 상태이고, 움직임은 행동이다.

한 번 움직일 때마다 음의 보상, 가령 r = -1을 얻는다.

목표는 회색 종료 상태에 최소한의 행동으로 도달하는 것이다.

1. Random Policy는 무작위로 방향을 결정한다.
2. Optimal Policy는 종료 상태에 도달하는 적절한 방향을 결정한다.

![CS231n%20LEC14%20d66c599e581d40669654e69af238e10c/473b6022-aa43-484c-95b7-92301872df6c.pdf-0024.png](CS231n%20LEC14%20d66c599e581d40669654e69af238e10c/473b6022-aa43-484c-95b7-92301872df6c.pdf-0024.png)

최적의 $\pi^*$는 보상의 합을 최대화시킨다.

MDP에서 발생하는 무작위성을 다루기 위해, 보상의 합에 대한 기댓값을 최대화하는 것이다.

![CS231n%20LEC14%20d66c599e581d40669654e69af238e10c/473b6022-aa43-484c-95b7-92301872df6c.pdf-0027.png](CS231n%20LEC14%20d66c599e581d40669654e69af238e10c/473b6022-aa43-484c-95b7-92301872df6c.pdf-0027.png)

우리가 얻을 수 있는 상태, 행동, 보상들은 하나의 경로가 된다.

1. Value Funcion: 상태 $s$와 정책 $\pi$가 주어졌을 때, 누적 보상의 기댓값.
2. Q-value Funcion: 상태 $s$와 행동 $a$가 주어졌을 때, 누적 보상의 기댓값.

![CS231n%20LEC14%20d66c599e581d40669654e69af238e10c/473b6022-aa43-484c-95b7-92301872df6c.pdf-0030.png](CS231n%20LEC14%20d66c599e581d40669654e69af238e10c/473b6022-aa43-484c-95b7-92301872df6c.pdf-0030.png)

최적의 Q-value Funcion $Q^*$는 (s, a) 쌍에서 얻을 수 있는 누적 보상의 기댓값을 최대화시킨다.

$Q^*$는 Bellman 방정식을 만족한다.

![CS231n%20LEC14%20d66c599e581d40669654e69af238e10c/473b6022-aa43-484c-95b7-92301872df6c.pdf-0034.png](CS231n%20LEC14%20d66c599e581d40669654e69af238e10c/473b6022-aa43-484c-95b7-92301872df6c.pdf-0034.png)

Bellman 방정식으로 각 Step마다 $Q^*$를 조금씩 최적화하여 최선의 정책을 구할 수 있다.

1. 문제점

    Scalable하지 않으므로 모든 (s, a)마다 Q를 계산해야 한다.

    Atari 게임의 경우 스크린의 모든 Pixel이 상태가 되고, 상태 공간이 매우 커서 계산이 불가능하다.

2. 해결법

    함수 Q를 Neural Network 등의 방법으로 근사시킨다.

## Q-learning.

![CS231n%20LEC14%20d66c599e581d40669654e69af238e10c/473b6022-aa43-484c-95b7-92301872df6c.pdf-0037.png](CS231n%20LEC14%20d66c599e581d40669654e69af238e10c/473b6022-aa43-484c-95b7-92301872df6c.pdf-0037.png)

![CS231n%20LEC14%20d66c599e581d40669654e69af238e10c/Untitled.png](CS231n%20LEC14%20d66c599e581d40669654e69af238e10c/Untitled.png)

NN을 이용해 Q(s, a)를 근사시키는 것을 Deep Q-learning이라고 한다.

함수 Parameter $\theta$는 NN의 가중치이다.

1. Bellman Equation을 만족하도록 Q-function을 학습하여 Error를 최소로 하는 것이 목표이다.
2. Loss Function: Q(s, a)와 목적 함수 $y_i$의 거리를 측정한다.
3. Forward pass: Loss Function을 계산한다.
4. Backward pass: Loss를 바탕으로 Parameter를 Update 한다.
5. 계속 Update 하여 Q-function이 목적에 가까워지도록 한다.

![CS231n%20LEC14%20d66c599e581d40669654e69af238e10c/Untitled%201.png](CS231n%20LEC14%20d66c599e581d40669654e69af238e10c/Untitled%201.png)

Atari 게임의 경우, 게임 내 Pixel 정보를 이용한다.

1. 입력은 상태 s로, 게임의 Pixel이 들어온다.

    4 프레임 정도 누적하여 사용한다.

2. RGB → Grayscale, Down Sampling, Cropping 등 전처리를 한다.

    예시는 84 X 84 X 4 형태이다.

3. 8X8, 4X4 Conv와 256 FC Layer를 거친다.

4. 출력은 상태 s에서 행동 a의 Q-value이다.

    행동이 4개라면 출력도 4차원이다.

5. Single Forward Pass로 모든 함수에 대한 Q-value를 계산할 수 있다.

![CS231n%20LEC14%20d66c599e581d40669654e69af238e10c/Untitled%202.png](CS231n%20LEC14%20d66c599e581d40669654e69af238e10c/Untitled%202.png)

1. Q-network는 시간적으로 연속인 샘플들을 이용하는 것이 좋지 않다.

    모든 샘플들이 Correlation(상관 관계)를 가지고 있기 때문이다.

    e.g. Atari처럼 게임을 진행하는 경우.

2. Q-network의 Parameter를 보면 우리가 정책을 결정하고 있기 때문에 다음 샘플에 대해서도 결정하게 된다.

    이는 학습에 좋지 않은 영향을 미친다.

    e.g. 왼쪽이 보상을 최대화하는 길이라면, 다음 샘플도 왼쪽으로 편향될 수 있다.

3. 해결책으로 Experience Replay를 사용한다.
    1. Replay Memory에는 (상태, 행동, 보상, 다음 상태)로 구성된 전이 테이블이 있다.
    2. 전이 테이블을 계속 Update하고, 임의의 Mini-batch를 사용하여 Q-network를 학습한다.
    3. 즉, 연속적인 샘플 대신 전이 테이블에서 임의로 Sampling 하여 사용한다.
    4. 각 전이가 가중치 Update에 여러 번 기여할 수 있다는 장점이 있다. → 하나의 샘플이 계속 뽑힐 수 있다. → 데이터 효율 증가.

![CS231n%20LEC14%20d66c599e581d40669654e69af238e10c/473b6022-aa43-484c-95b7-92301872df6c.pdf-0054.png](CS231n%20LEC14%20d66c599e581d40669654e69af238e10c/473b6022-aa43-484c-95b7-92301872df6c.pdf-0054.png)

![CS231n%20LEC14%20d66c599e581d40669654e69af238e10c/473b6022-aa43-484c-95b7-92301872df6c.pdf-0055.png](CS231n%20LEC14%20d66c599e581d40669654e69af238e10c/473b6022-aa43-484c-95b7-92301872df6c.pdf-0055.png)

![CS231n%20LEC14%20d66c599e581d40669654e69af238e10c/473b6022-aa43-484c-95b7-92301872df6c.pdf-0056.png](CS231n%20LEC14%20d66c599e581d40669654e69af238e10c/473b6022-aa43-484c-95b7-92301872df6c.pdf-0056.png)

![CS231n%20LEC14%20d66c599e581d40669654e69af238e10c/473b6022-aa43-484c-95b7-92301872df6c.pdf-0057.png](CS231n%20LEC14%20d66c599e581d40669654e69af238e10c/473b6022-aa43-484c-95b7-92301872df6c.pdf-0057.png)

![CS231n%20LEC14%20d66c599e581d40669654e69af238e10c/473b6022-aa43-484c-95b7-92301872df6c.pdf-0058.png](CS231n%20LEC14%20d66c599e581d40669654e69af238e10c/473b6022-aa43-484c-95b7-92301872df6c.pdf-0058.png)

![CS231n%20LEC14%20d66c599e581d40669654e69af238e10c/473b6022-aa43-484c-95b7-92301872df6c.pdf-0059.png](CS231n%20LEC14%20d66c599e581d40669654e69af238e10c/473b6022-aa43-484c-95b7-92301872df6c.pdf-0059.png)

![CS231n%20LEC14%20d66c599e581d40669654e69af238e10c/473b6022-aa43-484c-95b7-92301872df6c.pdf-0060.png](CS231n%20LEC14%20d66c599e581d40669654e69af238e10c/473b6022-aa43-484c-95b7-92301872df6c.pdf-0060.png)

![CS231n%20LEC14%20d66c599e581d40669654e69af238e10c/473b6022-aa43-484c-95b7-92301872df6c.pdf-0061.png](CS231n%20LEC14%20d66c599e581d40669654e69af238e10c/473b6022-aa43-484c-95b7-92301872df6c.pdf-0061.png)

알고리즘의 전체 과정!

## Policy Gradients.

![CS231n%20LEC14%20d66c599e581d40669654e69af238e10c/473b6022-aa43-484c-95b7-92301872df6c.pdf-0064.png](CS231n%20LEC14%20d66c599e581d40669654e69af238e10c/473b6022-aa43-484c-95b7-92301872df6c.pdf-0064.png)

Q-learning은 모든 (State, Action)을 학습해야 하므로 Function이 너무 복잡하다.

그렇다면 정책 자체를 학습해보자.

![CS231n%20LEC14%20d66c599e581d40669654e69af238e10c/473b6022-aa43-484c-95b7-92301872df6c.pdf-0067.png](CS231n%20LEC14%20d66c599e581d40669654e69af238e10c/473b6022-aa43-484c-95b7-92301872df6c.pdf-0067.png)

정책들은 가중치 $\theta$에 의해 매개변수가 된다.

$J(\theta)$는 미래의 보상들을 누적 합의 기댓값으로 나타낸다.

최적의 정책 $\theta^*$은 $\underset{\theta}{argmax} \, J(\theta)$로 표현한다.

![CS231n%20LEC14%20d66c599e581d40669654e69af238e10c/473b6022-aa43-484c-95b7-92301872df6c.pdf-0068.png](CS231n%20LEC14%20d66c599e581d40669654e69af238e10c/473b6022-aa43-484c-95b7-92301872df6c.pdf-0068.png)

경로에 대한 미래의 보상을 기댓값으로 나타내기 위해 경로를 Sampling 한다.

경로는 정책 $\pi_{\theta}$를 따라 결정되는데, 각 경로에 대해 보상을 계산할 수 있다.

따라서 여기서 기댓값은 정책으로부터 Sampling 한 경로들의 기댓값이 되는 것이다.

![CS231n%20LEC14%20d66c599e581d40669654e69af238e10c/Untitled%203.png](CS231n%20LEC14%20d66c599e581d40669654e69af238e10c/Untitled%203.png)

$\theta^*$은 Gradient Ascent로 찾을 수 있다.

$J(\theta)$을 미분해도 계산할 수 없는 식이 나오는데, p가 $\theta$에 종속된 상황에서 기댓값 안에 $p(\tau ; \theta)$ Gradient를 계산하기 어렵다.

따라서 1을 곱하는 것처럼 $\frac{p(\tau;\theta)}{p(\tau;\theta)}$를 곱하여 식을 정리할 수 있다.

기댓값에 대한 Gradient가 Gradient에 대한 기댓값으로 바뀐다.

![CS231n%20LEC14%20d66c599e581d40669654e69af238e10c/Untitled%204.png](CS231n%20LEC14%20d66c599e581d40669654e69af238e10c/Untitled%204.png)

$p(\tau)$는 어떤 경로에 대한 확률이다.

이는 주어진 (State, Action)에서 모든 상태에 대한 "전이확률"과 "정책에게 얻은 행동에 대한 확률"의 곱이다.

$log(p;\tau)$는 곱의 형태가 합의 형태로 바뀌고, $\theta$에 대해 미분을 하여도 전이확률은 영향을 미치지 않는다.

따라서 Gradient 계산에서 전이확률은 필요하지 않다.

![CS231n%20LEC14%20d66c599e581d40669654e69af238e10c/Untitled%205.png](CS231n%20LEC14%20d66c599e581d40669654e69af238e10c/Untitled%205.png)

1. 특정 경로로 얻은 보상이 크다면, 그 행동에 대한 확률을 높인다.
2. 특정 경로로 얻은 보상이 작다면, 그 행동에 대한 확률을 줄인다.
3. 경로가 좋으면 경로에 포함된 모든 행동들이 좋은 판정을 받지만, 기댓값에 의해 Averages Out 된다.
4. Averages Out을 통해 Unbiased Extimator를 얻을 수 있다.

1. 장점.

    1. Gradient만 잘 계산하면 Loss Function이 작다.
    2. 정책 Parameter $\theta$에 대한 Local Optimum을 구할 수 있다.

2. 단점.

    1. Averages Out 되기 때문에 분산이 높다.

![CS231n%20LEC14%20d66c599e581d40669654e69af238e10c/Untitled%206.png](CS231n%20LEC14%20d66c599e581d40669654e69af238e10c/Untitled%206.png)

![CS231n%20LEC14%20d66c599e581d40669654e69af238e10c/Untitled%207.png](CS231n%20LEC14%20d66c599e581d40669654e69af238e10c/Untitled%207.png)

분산을 줄이는 방법을 살펴보자.

1. 현재 상태부터 받을 미래의 보상만 고려하여, 어떤 행동을 취할지 확률을 키우는 방법이다.

    해당 경로에서 얻는 전체 보상 대신, 현재 Time Step부터 얻을 수 있는 보상을 고려한다.

2. 미래의 보상에 대해 Discount Factor(할인율, 감가율)를 적용하는 방법이다.

    나중에 수행하는 행동에 대해 가중치를 낮춘다.

3. Baseline이라는 방법으로, 현재 상태에서 우리가 원하는 보상의 크기에 대한 함수이다.

    확률을 키우거나 줄이는 보상 수식이 "미래의 보상들의 합 - 기준 값(Baseline)"으로 바뀐다.

![CS231n%20LEC14%20d66c599e581d40669654e69af238e10c/473b6022-aa43-484c-95b7-92301872df6c.pdf-0086.png](CS231n%20LEC14%20d66c599e581d40669654e69af238e10c/473b6022-aa43-484c-95b7-92301872df6c.pdf-0086.png)

단순한 Baseline은 지금까지의 보상에 대해 Moving Average를 취하는 것이다.

즉, 지금까지의 모든 경로에 대해 보상에 대한 평균을 구한다.

이런 Variance Reduction 방법은 Vanilla REINFORCE이다.

Discount Factor를 적용하고, 단순한 Baseline을 추가한다.

![CS231n%20LEC14%20d66c599e581d40669654e69af238e10c/Untitled%208.png](CS231n%20LEC14%20d66c599e581d40669654e69af238e10c/Untitled%208.png)

더 좋은 Baseline은 현재 상태에서의 Q-function과 Value function의 차이를 통해 나타낼 수 있다.

![CS231n%20LEC14%20d66c599e581d40669654e69af238e10c/Untitled%209.png](CS231n%20LEC14%20d66c599e581d40669654e69af238e10c/Untitled%209.png)

이는 Policy Gradient와 Q-learning을 통해 학습할 수 있다.

$A(s, a) = Q(s, a) - V(s)$, Advantage Function(보상 함수)으로 나타낸다.

행동이 예상보다 얼마나 좋은지 나타낸다.

이 알고리즘을 Actor-critic Algorithm이라고 한다.

1. Actor: Policy로, 어떤 행동을 취할지 결정하는 함수.
2. Critic: Q-function으로, 행동이 얼마나 좋았고, 어떻게 조절할지 알려주는 함수.

    기존의 Q-learning과 다르게 정책이 만들어낸 (State, Action) 쌍에 대해서만 학습한다.

![CS231n%20LEC14%20d66c599e581d40669654e69af238e10c/Untitled%2010.png](CS231n%20LEC14%20d66c599e581d40669654e69af238e10c/Untitled%2010.png)

1. Policy Parameter $\theta$, Critic parameter $\phi$를 초기화한다.
2. 매 학습마다 현재의 정책을 기반으로 M개의 경로를 Sampling 한다.
3. 각 경로마다 보상 함수를 계산하고, 이를 이용해 Gradient Estimator를 계산하여 누적시킨다. 즉, $\theta$의 학습이다.
4. $\phi$를 학습시키기 위해 가치 함수를 학습해야 하는데, 보상 함수를 최소화 시키는 것과 같다.

![CS231n%20LEC14%20d66c599e581d40669654e69af238e10c/Untitled%2011.png](CS231n%20LEC14%20d66c599e581d40669654e69af238e10c/Untitled%2011.png)

RAM은 REINFORCE의 예시로 Hard Attention과 연관이 있다.

Glimpses(짧은 시간에 일부만 보는 것)를 통해 Image Classification 문제를 해결한다.

1. 저해상도로 먼저 보는 인간의 지각 능력에서 영감을 받았다.
2. 일부만 보는 것으로 계산 자원을 절약할 수 있다.
3. 필요없는 부분이 무시되므로 Classification 성능을 높인다.

강화 학습 수식으로 Image Classification을 나타낸다.

1. State: 지금까지 관찰한 Glimpses.
2. Action: 다음에 볼 이미지 부분을 선택.
3. Reward: 이미지 분류에 성공하면 1, 아니라면 0.

강화 학습이 필요한 이유는 이미지에서 Glimpses를 뽑아내는 것은 미분이 불가능하기 때문이다.

어떻게 Glimpses를 얻을지 정책을 통해 학습한다.

![CS231n%20LEC14%20d66c599e581d40669654e69af238e10c/473b6022-aa43-484c-95b7-92301872df6c.pdf-0095.png](CS231n%20LEC14%20d66c599e581d40669654e69af238e10c/473b6022-aa43-484c-95b7-92301872df6c.pdf-0095.png)

![CS231n%20LEC14%20d66c599e581d40669654e69af238e10c/Untitled%2012.png](CS231n%20LEC14%20d66c599e581d40669654e69af238e10c/Untitled%2012.png)

![CS231n%20LEC14%20d66c599e581d40669654e69af238e10c/473b6022-aa43-484c-95b7-92301872df6c.pdf-0100.png](CS231n%20LEC14%20d66c599e581d40669654e69af238e10c/473b6022-aa43-484c-95b7-92301872df6c.pdf-0100.png)

1. 누적된 Gilmpses가 모델에 들어가고, 상태를 모델링하기 위해 RNN을 사용한다.
2. Policy Parameters를 이용하여 다음 행동을 선택한다.
3. 출력은 x-y 좌표로, 실제 값은 행동에 대한 분포, Gaussian을 따르고 출력은 분포의 평균 값이다.

![CS231n%20LEC14%20d66c599e581d40669654e69af238e10c/473b6022-aa43-484c-95b7-92301872df6c.pdf-0101.png](CS231n%20LEC14%20d66c599e581d40669654e69af238e10c/473b6022-aa43-484c-95b7-92301872df6c.pdf-0101.png)

## Summary.

![CS231n%20LEC14%20d66c599e581d40669654e69af238e10c/473b6022-aa43-484c-95b7-92301872df6c.pdf-0102.png](CS231n%20LEC14%20d66c599e581d40669654e69af238e10c/473b6022-aa43-484c-95b7-92301872df6c.pdf-0102.png)

## 링크.

[Lecture 14 | Deep Reinforcement Learning |](https://www.youtube.com/watch?v=lvoHnicueoE&list=PL3FW7Lu3i5JvHM8ljYj-zLfQRF3EO8sYv&index=14)

[CS231n: Convolutional Neural Networks for Visual Recognition](http://cs231n.stanford.edu/2017/)

[CS231n 2017 Lecture14 PDF](http://cs231n.stanford.edu/slides/2017/cs231n_2017_lecture14.pdf)

[soline013/CS231N_17_KOR_SUB](https://github.com/soline013/CS231N_17_KOR_SUB/blob/master/kor/Lecture%2011%20%20%20Detection%20and%20Segmentation.ko.srt)