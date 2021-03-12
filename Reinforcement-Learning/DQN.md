# DQN

DQN(Deep Q Learning).

## Return.

$$R_{t_o} = \sum^{\infin}_{t=t_0}  \gamma^{t-t_o} r_t$$

$R_{t_0}$ = Return.

$\gamma^{t-t_0}$ = Discounted Constant. 0~1. Ensures the sum converges.

불확실한 미래의 보상이 덜 중요해지고, 가까운 미래의 보상이 더 중요하다.

[강화 학습(RL)_ MDP](https://brunch.co.kr/@minkh/3)

## Function Q.

$$Q^* : State \times Action → \R$$

상태에서 행동이 있다면, 반환 R을 최대화하는 정책을 쉽게 구축할 수 있다.

→ $\pi^* (s) = \underset{a}{argmax} Q^* (s,a)$

---

$$Q^{\pi}(s, a) = r + \gamma Q^{\pi} (s', \pi(s'))$$

함수 Q는 Bellman 방정식을 준수한다.

세상의 모든 것을 알지 못하기에 $Q^*$에는 도달할 수 없지만, 최대한 $Q^*$를 닮도록 한다.

## Our Goals.

Discounted Cumulative Reward를 최대화하는 Policy를 학습하는 것이다.

## Temporal Difference Error.

$$\delta = Q(s, a) - (r + \gamma \ \underset{a}{max} Q(s', a))$$

Equality의 두 측면 사이의 차이이다.

## Huber Loss.

$$\mathcal{L} = \frac{1}{|B|} \underset{(s, a, s', x) \in B}{\sum \mathcal{L}(\delta)}$$

$$where \ \ \mathcal{L}(\delta) = \begin{cases} \frac{1}{2} \delta^2 & for|\delta| \le 1, \\ |\delta| - \frac{1}{2} & otherwise.
\end{cases}$$

Huber Loss는 오류가 작으면 평균 제곱 오차(Mean Squared Error)와 같이 동작하고, 오류가 크면 평균 절대 오차(Mean Absolute Error)와 유사하다.

재현 메모리에서 Sampling한 전환 배치 B에서 계산한다.

## Model.

현재와 이전의 차이를 취하는 CNN이다.

입력: s.

출력: Q(s, left), Q(s, right).

## Total Data Flow.

![Untitled](https://user-images.githubusercontent.com/66259854/110951345-4e74cb00-8388-11eb-841e-f331e7c237fb.png)

> 행동은 무작위 또는 정책에 따라 선택되어, gym 환경에서 다음 단계 샘플을 가져옵니다. 결과를 재현 메모리에 저장하고 모든 반복에서 최적화 단계를 실행합니다. 최적화는 재현 메모리에서 무작위 배치를 선택하여 새 정책을 학습합니다. “이전” target_net은 최적화에서 기대 Q 값을 계산하는 데에도 사용되고, 최신 상태를 유지하기 위해 가끔 업데이트됩니다.

## 링크.

[강화 학습 (DQN) 튜토리얼 - PyTorch Tutorials 1.6.0 documentation](https://tutorials.pytorch.kr/intermediate/reinforcement_q_learning.html)