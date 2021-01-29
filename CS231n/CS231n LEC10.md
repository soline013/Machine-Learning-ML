# CS231n LEC10.
## Stanford University CS231n, Spring 2017.
**Recurrent Neural Networks.**

## Recall from last time.

![CS231n%20LEC10%20eb9e2b5eba9b421c8ec6ef272fd20e0e/Untitled.png](CS231n%20LEC10%20eb9e2b5eba9b421c8ec6ef272fd20e0e/Untitled.png)

![CS231n%20LEC10%20eb9e2b5eba9b421c8ec6ef272fd20e0e/cbbf2cf5-0515-420f-921d-20b653cee8db.pdf-0006.jpg](CS231n%20LEC10%20eb9e2b5eba9b421c8ec6ef272fd20e0e/cbbf2cf5-0515-420f-921d-20b653cee8db.pdf-0006.jpg)

![CS231n%20LEC10%20eb9e2b5eba9b421c8ec6ef272fd20e0e/cbbf2cf5-0515-420f-921d-20b653cee8db.pdf-0008.jpg](CS231n%20LEC10%20eb9e2b5eba9b421c8ec6ef272fd20e0e/cbbf2cf5-0515-420f-921d-20b653cee8db.pdf-0008.jpg)

![CS231n%20LEC10%20eb9e2b5eba9b421c8ec6ef272fd20e0e/Untitled%201.png](CS231n%20LEC10%20eb9e2b5eba9b421c8ec6ef272fd20e0e/Untitled%201.png)

![CS231n%20LEC10%20eb9e2b5eba9b421c8ec6ef272fd20e0e/cbbf2cf5-0515-420f-921d-20b653cee8db.pdf-0007.jpg](CS231n%20LEC10%20eb9e2b5eba9b421c8ec6ef272fd20e0e/cbbf2cf5-0515-420f-921d-20b653cee8db.pdf-0007.jpg)

![CS231n%20LEC10%20eb9e2b5eba9b421c8ec6ef272fd20e0e/cbbf2cf5-0515-420f-921d-20b653cee8db.pdf-0009.jpg](CS231n%20LEC10%20eb9e2b5eba9b421c8ec6ef272fd20e0e/cbbf2cf5-0515-420f-921d-20b653cee8db.pdf-0009.jpg)

## Recurrent Neural Networks: Process Sequences.

![CS231n%20LEC10%20eb9e2b5eba9b421c8ec6ef272fd20e0e/cbbf2cf5-0515-420f-921d-20b653cee8db.pdf-0011.jpg](CS231n%20LEC10%20eb9e2b5eba9b421c8ec6ef272fd20e0e/cbbf2cf5-0515-420f-921d-20b653cee8db.pdf-0011.jpg)

![CS231n%20LEC10%20eb9e2b5eba9b421c8ec6ef272fd20e0e/cbbf2cf5-0515-420f-921d-20b653cee8db.pdf-0012.jpg](CS231n%20LEC10%20eb9e2b5eba9b421c8ec6ef272fd20e0e/cbbf2cf5-0515-420f-921d-20b653cee8db.pdf-0012.jpg)

![CS231n%20LEC10%20eb9e2b5eba9b421c8ec6ef272fd20e0e/cbbf2cf5-0515-420f-921d-20b653cee8db.pdf-0013.jpg](CS231n%20LEC10%20eb9e2b5eba9b421c8ec6ef272fd20e0e/cbbf2cf5-0515-420f-921d-20b653cee8db.pdf-0013.jpg)

![CS231n%20LEC10%20eb9e2b5eba9b421c8ec6ef272fd20e0e/cbbf2cf5-0515-420f-921d-20b653cee8db.pdf-0014.jpg](CS231n%20LEC10%20eb9e2b5eba9b421c8ec6ef272fd20e0e/cbbf2cf5-0515-420f-921d-20b653cee8db.pdf-0014.jpg)

![CS231n%20LEC10%20eb9e2b5eba9b421c8ec6ef272fd20e0e/cbbf2cf5-0515-420f-921d-20b653cee8db.pdf-0015.jpg](CS231n%20LEC10%20eb9e2b5eba9b421c8ec6ef272fd20e0e/cbbf2cf5-0515-420f-921d-20b653cee8db.pdf-0015.jpg)

가변 입력과 가변 출력으로 다양한 선택지를 제공한다.

## Sequential Processing of Non-Sequence Data.

![CS231n%20LEC10%20eb9e2b5eba9b421c8ec6ef272fd20e0e/cbbf2cf5-0515-420f-921d-20b653cee8db.pdf-0016.jpg](CS231n%20LEC10%20eb9e2b5eba9b421c8ec6ef272fd20e0e/cbbf2cf5-0515-420f-921d-20b653cee8db.pdf-0016.jpg)

![CS231n%20LEC10%20eb9e2b5eba9b421c8ec6ef272fd20e0e/cbbf2cf5-0515-420f-921d-20b653cee8db.pdf-0017.jpg](CS231n%20LEC10%20eb9e2b5eba9b421c8ec6ef272fd20e0e/cbbf2cf5-0515-420f-921d-20b653cee8db.pdf-0017.jpg)

입출력은 고정된 길이지만, 가변 과정인 경우에도 사용할 수 있다.

## RNN.

![CS231n%20LEC10%20eb9e2b5eba9b421c8ec6ef272fd20e0e/cbbf2cf5-0515-420f-921d-20b653cee8db.pdf-0019.jpg](CS231n%20LEC10%20eb9e2b5eba9b421c8ec6ef272fd20e0e/cbbf2cf5-0515-420f-921d-20b653cee8db.pdf-0019.jpg)

1. RNN이 입력을 받는다.
2. Hidden State를 업데이트한다.
3. 출력 값을 내보낸다.

![CS231n%20LEC10%20eb9e2b5eba9b421c8ec6ef272fd20e0e/cbbf2cf5-0515-420f-921d-20b653cee8db.pdf-0020.jpg](CS231n%20LEC10%20eb9e2b5eba9b421c8ec6ef272fd20e0e/cbbf2cf5-0515-420f-921d-20b653cee8db.pdf-0020.jpg)

RNN Block은 재귀적인 관계를 함수 f로 연산할 수 있다.

1. $h_{t-1}$: 이전 상태의 Hidden State.
2. $x_t$: 현재 상태의 입력.
3. $h_t$: 다음 상태의 Hidden State.

![CS231n%20LEC10%20eb9e2b5eba9b421c8ec6ef272fd20e0e/cbbf2cf5-0515-420f-921d-20b653cee8db.pdf-0021.jpg](CS231n%20LEC10%20eb9e2b5eba9b421c8ec6ef272fd20e0e/cbbf2cf5-0515-420f-921d-20b653cee8db.pdf-0021.jpg)

RNN에서 Y를 가지려면 $h_t$를 입력받는 FC-layer가 필요하다.

함수 f와 Parameter W는 매 스텝 동일하다.

![CS231n%20LEC10%20eb9e2b5eba9b421c8ec6ef272fd20e0e/cbbf2cf5-0515-420f-921d-20b653cee8db.pdf-0022.jpg](CS231n%20LEC10%20eb9e2b5eba9b421c8ec6ef272fd20e0e/cbbf2cf5-0515-420f-921d-20b653cee8db.pdf-0022.jpg)

1. 가중치 행렬 $W_{xh}$와 입력 $x_t$
2. 가중치 행렬 $W_{hh}$와 이전 Hidden State $h_{t-1}$
3. Non-linearity 구현을 위한 tanh → `LSTM에서 부가 설명.`
4. 가중치 행렬 $W_{hy}$와 Hidden State $h_t$

## RNN Computational Graph.

- Vanilla.

    ![CS231n%20LEC10%20eb9e2b5eba9b421c8ec6ef272fd20e0e/cbbf2cf5-0515-420f-921d-20b653cee8db.pdf-0023.jpg](CS231n%20LEC10%20eb9e2b5eba9b421c8ec6ef272fd20e0e/cbbf2cf5-0515-420f-921d-20b653cee8db.pdf-0023.jpg)

    ![CS231n%20LEC10%20eb9e2b5eba9b421c8ec6ef272fd20e0e/cbbf2cf5-0515-420f-921d-20b653cee8db.pdf-0024.jpg](CS231n%20LEC10%20eb9e2b5eba9b421c8ec6ef272fd20e0e/cbbf2cf5-0515-420f-921d-20b653cee8db.pdf-0024.jpg)

    ![CS231n%20LEC10%20eb9e2b5eba9b421c8ec6ef272fd20e0e/cbbf2cf5-0515-420f-921d-20b653cee8db.pdf-0025.jpg](CS231n%20LEC10%20eb9e2b5eba9b421c8ec6ef272fd20e0e/cbbf2cf5-0515-420f-921d-20b653cee8db.pdf-0025.jpg)

    ![CS231n%20LEC10%20eb9e2b5eba9b421c8ec6ef272fd20e0e/cbbf2cf5-0515-420f-921d-20b653cee8db.pdf-0026.jpg](CS231n%20LEC10%20eb9e2b5eba9b421c8ec6ef272fd20e0e/cbbf2cf5-0515-420f-921d-20b653cee8db.pdf-0026.jpg)

    가중치 행렬 W는 항상 동일하다.

- Many to Many.

    ![CS231n%20LEC10%20eb9e2b5eba9b421c8ec6ef272fd20e0e/cbbf2cf5-0515-420f-921d-20b653cee8db.pdf-0027.jpg](CS231n%20LEC10%20eb9e2b5eba9b421c8ec6ef272fd20e0e/cbbf2cf5-0515-420f-921d-20b653cee8db.pdf-0027.jpg)

    ![CS231n%20LEC10%20eb9e2b5eba9b421c8ec6ef272fd20e0e/cbbf2cf5-0515-420f-921d-20b653cee8db.pdf-0028.jpg](CS231n%20LEC10%20eb9e2b5eba9b421c8ec6ef272fd20e0e/cbbf2cf5-0515-420f-921d-20b653cee8db.pdf-0028.jpg)

    ![CS231n%20LEC10%20eb9e2b5eba9b421c8ec6ef272fd20e0e/cbbf2cf5-0515-420f-921d-20b653cee8db.pdf-0029.jpg](CS231n%20LEC10%20eb9e2b5eba9b421c8ec6ef272fd20e0e/cbbf2cf5-0515-420f-921d-20b653cee8db.pdf-0029.jpg)

    Back Prop에서는 $\frac{dL}{dW}$를 구해야 한다.

    1. RNN에서 Backward를 위한 W의 Gradient를 구하려면 각 스텝의 Local Gradient를 계산하고, 모두 더한다.
    2. 각 스텝의 개별 Loss를 구하면, RNN의 Loss는 개별 Loss의 합이다.

- Many to One.

    ![CS231n%20LEC10%20eb9e2b5eba9b421c8ec6ef272fd20e0e/cbbf2cf5-0515-420f-921d-20b653cee8db.pdf-0030.jpg](CS231n%20LEC10%20eb9e2b5eba9b421c8ec6ef272fd20e0e/cbbf2cf5-0515-420f-921d-20b653cee8db.pdf-0030.jpg)

- One to Many.

    ![CS231n%20LEC10%20eb9e2b5eba9b421c8ec6ef272fd20e0e/cbbf2cf5-0515-420f-921d-20b653cee8db.pdf-0031.jpg](CS231n%20LEC10%20eb9e2b5eba9b421c8ec6ef272fd20e0e/cbbf2cf5-0515-420f-921d-20b653cee8db.pdf-0031.jpg)

    고정 입력은 모델의 Initial Hidden State를 초기화하는 용도.

    $h_0$는 대부분 0으로 초기화하는데, 이때도 0으로 초기화할까?

- seq2seq(Sequence to Sequence).

    ![CS231n%20LEC10%20eb9e2b5eba9b421c8ec6ef272fd20e0e/cbbf2cf5-0515-420f-921d-20b653cee8db.pdf-0032.jpg](CS231n%20LEC10%20eb9e2b5eba9b421c8ec6ef272fd20e0e/cbbf2cf5-0515-420f-921d-20b653cee8db.pdf-0032.jpg)

    ![CS231n%20LEC10%20eb9e2b5eba9b421c8ec6ef272fd20e0e/cbbf2cf5-0515-420f-921d-20b653cee8db.pdf-0033.jpg](CS231n%20LEC10%20eb9e2b5eba9b421c8ec6ef272fd20e0e/cbbf2cf5-0515-420f-921d-20b653cee8db.pdf-0033.jpg)

    1. Encoder.

        Many to One, 가변 입력.

        가변 입력은 하나의 벡터로 변환된다.

    2. Decoder.

        One to Many, 가변 출력.

        하나의 벡터를 입력받아 가변 출력이 이루어진다.

    Transfomer는 Encoder와 Decoder를 사용한다.

    BERT는 Transfomer의 Encoder 부분을 사용하고,

    GPT는 Decoder 부분을 사용한다.

    [[모두를 위한 기계번역] NMT를 이해해보자 01 - 인코더 디코더 구조](https://m.blog.naver.com/PostView.nhn?blogId=bcj1210&logNo=221581930356&isFromSearchAddView=true)

## Example: Character Level Language Model.

![CS231n%20LEC10%20eb9e2b5eba9b421c8ec6ef272fd20e0e/cbbf2cf5-0515-420f-921d-20b653cee8db.pdf-0034.jpg](CS231n%20LEC10%20eb9e2b5eba9b421c8ec6ef272fd20e0e/cbbf2cf5-0515-420f-921d-20b653cee8db.pdf-0034.jpg)

![CS231n%20LEC10%20eb9e2b5eba9b421c8ec6ef272fd20e0e/cbbf2cf5-0515-420f-921d-20b653cee8db.pdf-0035.jpg](CS231n%20LEC10%20eb9e2b5eba9b421c8ec6ef272fd20e0e/cbbf2cf5-0515-420f-921d-20b653cee8db.pdf-0035.jpg)

![CS231n%20LEC10%20eb9e2b5eba9b421c8ec6ef272fd20e0e/cbbf2cf5-0515-420f-921d-20b653cee8db.pdf-0036.jpg](CS231n%20LEC10%20eb9e2b5eba9b421c8ec6ef272fd20e0e/cbbf2cf5-0515-420f-921d-20b653cee8db.pdf-0036.jpg)

## Example: Character Level Language Model Sampling.

![CS231n%20LEC10%20eb9e2b5eba9b421c8ec6ef272fd20e0e/cbbf2cf5-0515-420f-921d-20b653cee8db.pdf-0037.jpg](CS231n%20LEC10%20eb9e2b5eba9b421c8ec6ef272fd20e0e/cbbf2cf5-0515-420f-921d-20b653cee8db.pdf-0037.jpg)

![CS231n%20LEC10%20eb9e2b5eba9b421c8ec6ef272fd20e0e/cbbf2cf5-0515-420f-921d-20b653cee8db.pdf-0038.jpg](CS231n%20LEC10%20eb9e2b5eba9b421c8ec6ef272fd20e0e/cbbf2cf5-0515-420f-921d-20b653cee8db.pdf-0038.jpg)

![CS231n%20LEC10%20eb9e2b5eba9b421c8ec6ef272fd20e0e/cbbf2cf5-0515-420f-921d-20b653cee8db.pdf-0039.jpg](CS231n%20LEC10%20eb9e2b5eba9b421c8ec6ef272fd20e0e/cbbf2cf5-0515-420f-921d-20b653cee8db.pdf-0039.jpg)

![CS231n%20LEC10%20eb9e2b5eba9b421c8ec6ef272fd20e0e/cbbf2cf5-0515-420f-921d-20b653cee8db.pdf-0040.jpg](CS231n%20LEC10%20eb9e2b5eba9b421c8ec6ef272fd20e0e/cbbf2cf5-0515-420f-921d-20b653cee8db.pdf-0040.jpg)

모든 문자에 대한 Score를 Sampling에 이용한다.

Score를 확률분포로 표현하기 위해 Softmax를 사용한다.

'e'의 확률은 13%에 불과하지만, 'e'가 Sampling 되었다.

- Question.

    Q: "가장 높은 스코어를 선택하지 않고 확률분포에서 샘플링하는 이유?"

    Argmax Probability, Sampling 모두 사용할 수 있지만, Sampling을 사용하면 다양한 결과를 얻을 수 있다.

    위의 예제의 경우, Sampling을 사용했기에 올바른 결과를 얻을 수 있었다.

    Q: "Test Time에 One Hot 대신 Softmax를 입력으로 사용할 수 있는가?"

    첫 번째 문제는 입력이 Train에서의 입력과 달라지는 것이다.

    두 번째 문제는 실제로 Vocabularies가 매우 크다는 것이다.

    실제로는 One Hot Vector를 Sparse Vector로 처리한다.

    Sparse Vector Operation: 공간 절약을 위해 0이 아닌 값만 저장.

## Truncated Backpropagation.

![CS231n%20LEC10%20eb9e2b5eba9b421c8ec6ef272fd20e0e/cbbf2cf5-0515-420f-921d-20b653cee8db.pdf-0041.jpg](CS231n%20LEC10%20eb9e2b5eba9b421c8ec6ef272fd20e0e/cbbf2cf5-0515-420f-921d-20b653cee8db.pdf-0041.jpg)

Sequence가 긴 경우 학습이 느려지는 문제가 발생한다.

![CS231n%20LEC10%20eb9e2b5eba9b421c8ec6ef272fd20e0e/cbbf2cf5-0515-420f-921d-20b653cee8db.pdf-0042.jpg](CS231n%20LEC10%20eb9e2b5eba9b421c8ec6ef272fd20e0e/cbbf2cf5-0515-420f-921d-20b653cee8db.pdf-0042.jpg)

![CS231n%20LEC10%20eb9e2b5eba9b421c8ec6ef272fd20e0e/cbbf2cf5-0515-420f-921d-20b653cee8db.pdf-0043.jpg](CS231n%20LEC10%20eb9e2b5eba9b421c8ec6ef272fd20e0e/cbbf2cf5-0515-420f-921d-20b653cee8db.pdf-0043.jpg)

![CS231n%20LEC10%20eb9e2b5eba9b421c8ec6ef272fd20e0e/cbbf2cf5-0515-420f-921d-20b653cee8db.pdf-0044.jpg](CS231n%20LEC10%20eb9e2b5eba9b421c8ec6ef272fd20e0e/cbbf2cf5-0515-420f-921d-20b653cee8db.pdf-0044.jpg)

따라서 Truncated Backpropagation을 사용하는데, Sequence를 일정 길이로 나누고 Loss를 구한다.

SDG에서 사용하는 Mini Batch와 같은 방식이다.

- Question.

    Q: "RNN이 Markov Assumption(마르코프 가정)을 따르는가?"

    RNN은 이전 Hidden State를 계속해서 앞으로 가져가기 때문에 따르지 않는다.

    - 번외. Markov Assumption.

        상태가 연속적인 시간에 따라 이어질 때 어떤 시점의 상태는 그 시점 바로 이전의 상태에만 영향을 받는다는 가정.

        e.g. 오늘 날씨는 어제 날씨에만 영향을 받는다.

        [컴공의 공부노트 : 네이버 블로그](https://blog.naver.com/kkang9901/222029504981)

        [Markov Model](http://blog.daum.net/hazzling/15605818)

## min-char-rnn, Andrej.

![CS231n%20LEC10%20eb9e2b5eba9b421c8ec6ef272fd20e0e/cbbf2cf5-0515-420f-921d-20b653cee8db.pdf-0045.jpg](CS231n%20LEC10%20eb9e2b5eba9b421c8ec6ef272fd20e0e/cbbf2cf5-0515-420f-921d-20b653cee8db.pdf-0045.jpg)

Vocabulary를 만들고, Truncated Backpropagation을 수행하는 모델.

- Shakespeare RNN.

    ![CS231n%20LEC10%20eb9e2b5eba9b421c8ec6ef272fd20e0e/cbbf2cf5-0515-420f-921d-20b653cee8db.pdf-0046.jpg](CS231n%20LEC10%20eb9e2b5eba9b421c8ec6ef272fd20e0e/cbbf2cf5-0515-420f-921d-20b653cee8db.pdf-0046.jpg)

    ![CS231n%20LEC10%20eb9e2b5eba9b421c8ec6ef272fd20e0e/cbbf2cf5-0515-420f-921d-20b653cee8db.pdf-0047.jpg](CS231n%20LEC10%20eb9e2b5eba9b421c8ec6ef272fd20e0e/cbbf2cf5-0515-420f-921d-20b653cee8db.pdf-0047.jpg)

    ![CS231n%20LEC10%20eb9e2b5eba9b421c8ec6ef272fd20e0e/cbbf2cf5-0515-420f-921d-20b653cee8db.pdf-0048.jpg](CS231n%20LEC10%20eb9e2b5eba9b421c8ec6ef272fd20e0e/cbbf2cf5-0515-420f-921d-20b653cee8db.pdf-0048.jpg)

- Algebraic Topology.

    ![CS231n%20LEC10%20eb9e2b5eba9b421c8ec6ef272fd20e0e/cbbf2cf5-0515-420f-921d-20b653cee8db.pdf-0049.jpg](CS231n%20LEC10%20eb9e2b5eba9b421c8ec6ef272fd20e0e/cbbf2cf5-0515-420f-921d-20b653cee8db.pdf-0049.jpg)

    ![CS231n%20LEC10%20eb9e2b5eba9b421c8ec6ef272fd20e0e/cbbf2cf5-0515-420f-921d-20b653cee8db.pdf-0050.jpg](CS231n%20LEC10%20eb9e2b5eba9b421c8ec6ef272fd20e0e/cbbf2cf5-0515-420f-921d-20b653cee8db.pdf-0050.jpg)

    ![CS231n%20LEC10%20eb9e2b5eba9b421c8ec6ef272fd20e0e/cbbf2cf5-0515-420f-921d-20b653cee8db.pdf-0051.jpg](CS231n%20LEC10%20eb9e2b5eba9b421c8ec6ef272fd20e0e/cbbf2cf5-0515-420f-921d-20b653cee8db.pdf-0051.jpg)

- Linux Kernel.

    ![CS231n%20LEC10%20eb9e2b5eba9b421c8ec6ef272fd20e0e/cbbf2cf5-0515-420f-921d-20b653cee8db.pdf-0052.jpg](CS231n%20LEC10%20eb9e2b5eba9b421c8ec6ef272fd20e0e/cbbf2cf5-0515-420f-921d-20b653cee8db.pdf-0052.jpg)

    ![CS231n%20LEC10%20eb9e2b5eba9b421c8ec6ef272fd20e0e/cbbf2cf5-0515-420f-921d-20b653cee8db.pdf-0053.jpg](CS231n%20LEC10%20eb9e2b5eba9b421c8ec6ef272fd20e0e/cbbf2cf5-0515-420f-921d-20b653cee8db.pdf-0053.jpg)

    ![CS231n%20LEC10%20eb9e2b5eba9b421c8ec6ef272fd20e0e/cbbf2cf5-0515-420f-921d-20b653cee8db.pdf-0054.jpg](CS231n%20LEC10%20eb9e2b5eba9b421c8ec6ef272fd20e0e/cbbf2cf5-0515-420f-921d-20b653cee8db.pdf-0054.jpg)

    ![CS231n%20LEC10%20eb9e2b5eba9b421c8ec6ef272fd20e0e/cbbf2cf5-0515-420f-921d-20b653cee8db.pdf-0055.jpg](CS231n%20LEC10%20eb9e2b5eba9b421c8ec6ef272fd20e0e/cbbf2cf5-0515-420f-921d-20b653cee8db.pdf-0055.jpg)

## Searching for Interpretable Cells.

![CS231n%20LEC10%20eb9e2b5eba9b421c8ec6ef272fd20e0e/cbbf2cf5-0515-420f-921d-20b653cee8db.pdf-0056.jpg](CS231n%20LEC10%20eb9e2b5eba9b421c8ec6ef272fd20e0e/cbbf2cf5-0515-420f-921d-20b653cee8db.pdf-0056.jpg)

Hidden Layer의 Vector를 추출하면 해석 가능한 어떤 의미 있는 값이 나오지 않을까 추측하였다.

![CS231n%20LEC10%20eb9e2b5eba9b421c8ec6ef272fd20e0e/cbbf2cf5-0515-420f-921d-20b653cee8db.pdf-0057.jpg](CS231n%20LEC10%20eb9e2b5eba9b421c8ec6ef272fd20e0e/cbbf2cf5-0515-420f-921d-20b653cee8db.pdf-0057.jpg)

Hidden State 대부분은 의미 없는 값이 나온다.

Vector 하나를 뽑고, Sequence를 다시 Forward 한다.

각 색깔은 Sequence를 진행하는 동안 앞에서 뽑은 Vector이다.

1. 따옴표를 만나면 값이 켜져 빨간색이 되고, 따옴표가 끝나면 파란색이 된다.

    ![CS231n%20LEC10%20eb9e2b5eba9b421c8ec6ef272fd20e0e/cbbf2cf5-0515-420f-921d-20b653cee8db.pdf-0058.jpg](CS231n%20LEC10%20eb9e2b5eba9b421c8ec6ef272fd20e0e/cbbf2cf5-0515-420f-921d-20b653cee8db.pdf-0058.jpg)

2. 줄 바꿈을 위해 현재 줄의 단어 수를 세는 듯, 점점 빨간색으로 변하다가 다음 줄에서 파란색으로 초기화된다.

    ![CS231n%20LEC10%20eb9e2b5eba9b421c8ec6ef272fd20e0e/cbbf2cf5-0515-420f-921d-20b653cee8db.pdf-0059.jpg](CS231n%20LEC10%20eb9e2b5eba9b421c8ec6ef272fd20e0e/cbbf2cf5-0515-420f-921d-20b653cee8db.pdf-0059.jpg)

3. Linux 코드를 학습시킬 때 발견한 것으로, if문의  조건부에서 값이 켜져 빨간색이 된다.

    ![CS231n%20LEC10%20eb9e2b5eba9b421c8ec6ef272fd20e0e/cbbf2cf5-0515-420f-921d-20b653cee8db.pdf-0060.jpg](CS231n%20LEC10%20eb9e2b5eba9b421c8ec6ef272fd20e0e/cbbf2cf5-0515-420f-921d-20b653cee8db.pdf-0060.jpg)

4. Linux 코드 내에서 Quote나 Comment를 찾는다.

    ![CS231n%20LEC10%20eb9e2b5eba9b421c8ec6ef272fd20e0e/cbbf2cf5-0515-420f-921d-20b653cee8db.pdf-0061.jpg](CS231n%20LEC10%20eb9e2b5eba9b421c8ec6ef272fd20e0e/cbbf2cf5-0515-420f-921d-20b653cee8db.pdf-0061.jpg)

5. Linux 코드 내에서 들여쓰기에 따라 빨간색으로 변한다.

    ![CS231n%20LEC10%20eb9e2b5eba9b421c8ec6ef272fd20e0e/cbbf2cf5-0515-420f-921d-20b653cee8db.pdf-0062.jpg](CS231n%20LEC10%20eb9e2b5eba9b421c8ec6ef272fd20e0e/cbbf2cf5-0515-420f-921d-20b653cee8db.pdf-0062.jpg)

## Image Captioning By using RNN.

![CS231n%20LEC10%20eb9e2b5eba9b421c8ec6ef272fd20e0e/cbbf2cf5-0515-420f-921d-20b653cee8db.pdf-0063.jpg](CS231n%20LEC10%20eb9e2b5eba9b421c8ec6ef272fd20e0e/cbbf2cf5-0515-420f-921d-20b653cee8db.pdf-0063.jpg)

![CS231n%20LEC10%20eb9e2b5eba9b421c8ec6ef272fd20e0e/cbbf2cf5-0515-420f-921d-20b653cee8db.pdf-0064.jpg](CS231n%20LEC10%20eb9e2b5eba9b421c8ec6ef272fd20e0e/cbbf2cf5-0515-420f-921d-20b653cee8db.pdf-0064.jpg)

1. CNN은 요약된 이미지 정보 Vector를 출력하고, 이 Vector는 RNN의 h가 된다.
2. RNN은 Caption에 사용할 문자들을 만들어낸다.

---

1. Softmax를 사용하지 않고 4,096-dim Vector를 사용한다.

    ![CS231n%20LEC10%20eb9e2b5eba9b421c8ec6ef272fd20e0e/cbbf2cf5-0515-420f-921d-20b653cee8db.pdf-0067.jpg](CS231n%20LEC10%20eb9e2b5eba9b421c8ec6ef272fd20e0e/cbbf2cf5-0515-420f-921d-20b653cee8db.pdf-0067.jpg)

2. RNN의 입력은 "Hey, this is the strat of a sentence. Please start generating some text conditioned on this image information."이라는 START Token이다.

    ![CS231n%20LEC10%20eb9e2b5eba9b421c8ec6ef272fd20e0e/cbbf2cf5-0515-420f-921d-20b653cee8db.pdf-0068.jpg](CS231n%20LEC10%20eb9e2b5eba9b421c8ec6ef272fd20e0e/cbbf2cf5-0515-420f-921d-20b653cee8db.pdf-0068.jpg)

3. h는 기존 가중치 행렬에 이미지 정보를 나타내는 가중치 행렬을 더한다.

    ![CS231n%20LEC10%20eb9e2b5eba9b421c8ec6ef272fd20e0e/cbbf2cf5-0515-420f-921d-20b653cee8db.pdf-0069.jpg](CS231n%20LEC10%20eb9e2b5eba9b421c8ec6ef272fd20e0e/cbbf2cf5-0515-420f-921d-20b653cee8db.pdf-0069.jpg)

4. END Token이 Sampling되면 종료되고 Caption이 완성된다.

    ![CS231n%20LEC10%20eb9e2b5eba9b421c8ec6ef272fd20e0e/cbbf2cf5-0515-420f-921d-20b653cee8db.pdf-0070.jpg](CS231n%20LEC10%20eb9e2b5eba9b421c8ec6ef272fd20e0e/cbbf2cf5-0515-420f-921d-20b653cee8db.pdf-0070.jpg)

    ![CS231n%20LEC10%20eb9e2b5eba9b421c8ec6ef272fd20e0e/cbbf2cf5-0515-420f-921d-20b653cee8db.pdf-0071.jpg](CS231n%20LEC10%20eb9e2b5eba9b421c8ec6ef272fd20e0e/cbbf2cf5-0515-420f-921d-20b653cee8db.pdf-0071.jpg)

    ...

    ![CS231n%20LEC10%20eb9e2b5eba9b421c8ec6ef272fd20e0e/cbbf2cf5-0515-420f-921d-20b653cee8db.pdf-0074.jpg](CS231n%20LEC10%20eb9e2b5eba9b421c8ec6ef272fd20e0e/cbbf2cf5-0515-420f-921d-20b653cee8db.pdf-0074.jpg)

    Train에서 모든 Caption의 마지막에 END Token을 추가한다.

    그러면 Test에서 자동으로 마지막에 END Token을 샘플링한다.

5. Example.

    ![CS231n%20LEC10%20eb9e2b5eba9b421c8ec6ef272fd20e0e/cbbf2cf5-0515-420f-921d-20b653cee8db.pdf-0075.jpg](CS231n%20LEC10%20eb9e2b5eba9b421c8ec6ef272fd20e0e/cbbf2cf5-0515-420f-921d-20b653cee8db.pdf-0075.jpg)

    ![CS231n%20LEC10%20eb9e2b5eba9b421c8ec6ef272fd20e0e/cbbf2cf5-0515-420f-921d-20b653cee8db.pdf-0076.jpg](CS231n%20LEC10%20eb9e2b5eba9b421c8ec6ef272fd20e0e/cbbf2cf5-0515-420f-921d-20b653cee8db.pdf-0076.jpg)

## Attention.

![CS231n%20LEC10%20eb9e2b5eba9b421c8ec6ef272fd20e0e/cbbf2cf5-0515-420f-921d-20b653cee8db.pdf-0077.jpg](CS231n%20LEC10%20eb9e2b5eba9b421c8ec6ef272fd20e0e/cbbf2cf5-0515-420f-921d-20b653cee8db.pdf-0077.jpg)

1. CNN으로 하나의 벡터가 아닌 각 벡터가 공간 정보를 갖는 Grid of Vector, $L \times D$를 만든다.

    ![CS231n%20LEC10%20eb9e2b5eba9b421c8ec6ef272fd20e0e/cbbf2cf5-0515-420f-921d-20b653cee8db.pdf-0078.jpg](CS231n%20LEC10%20eb9e2b5eba9b421c8ec6ef272fd20e0e/cbbf2cf5-0515-420f-921d-20b653cee8db.pdf-0078.jpg)

2. $h_0$에서 이미지 위치에 대한 분포, $a1$을 계산한다.

    ![CS231n%20LEC10%20eb9e2b5eba9b421c8ec6ef272fd20e0e/cbbf2cf5-0515-420f-921d-20b653cee8db.pdf-0079.jpg](CS231n%20LEC10%20eb9e2b5eba9b421c8ec6ef272fd20e0e/cbbf2cf5-0515-420f-921d-20b653cee8db.pdf-0079.jpg)

3. Grid of Vector와 분포를 계산하여 이미지 Attention, $z1$을 만든다.

    ![CS231n%20LEC10%20eb9e2b5eba9b421c8ec6ef272fd20e0e/cbbf2cf5-0515-420f-921d-20b653cee8db.pdf-0080.jpg](CS231n%20LEC10%20eb9e2b5eba9b421c8ec6ef272fd20e0e/cbbf2cf5-0515-420f-921d-20b653cee8db.pdf-0080.jpg)

4. Attention $z1$과 First Word $y1$이 다음 Step의 입력으로 들어간다.

    ![CS231n%20LEC10%20eb9e2b5eba9b421c8ec6ef272fd20e0e/cbbf2cf5-0515-420f-921d-20b653cee8db.pdf-0081.jpg](CS231n%20LEC10%20eb9e2b5eba9b421c8ec6ef272fd20e0e/cbbf2cf5-0515-420f-921d-20b653cee8db.pdf-0081.jpg)

5. 이미지 위치에 대한 분포 $a2$와, 각 단어들의 분포 $d1$을 계산한다.

    ![CS231n%20LEC10%20eb9e2b5eba9b421c8ec6ef272fd20e0e/cbbf2cf5-0515-420f-921d-20b653cee8db.pdf-0082.jpg](CS231n%20LEC10%20eb9e2b5eba9b421c8ec6ef272fd20e0e/cbbf2cf5-0515-420f-921d-20b653cee8db.pdf-0082.jpg)

6. 또 다시 Grid of Vector와 분포를 계산하여 이미지 Attention, $z2$를 만들고 Step을 반복한다.

    ![CS231n%20LEC10%20eb9e2b5eba9b421c8ec6ef272fd20e0e/cbbf2cf5-0515-420f-921d-20b653cee8db.pdf-0083.jpg](CS231n%20LEC10%20eb9e2b5eba9b421c8ec6ef272fd20e0e/cbbf2cf5-0515-420f-921d-20b653cee8db.pdf-0083.jpg)

    ![CS231n%20LEC10%20eb9e2b5eba9b421c8ec6ef272fd20e0e/cbbf2cf5-0515-420f-921d-20b653cee8db.pdf-0084.jpg](CS231n%20LEC10%20eb9e2b5eba9b421c8ec6ef272fd20e0e/cbbf2cf5-0515-420f-921d-20b653cee8db.pdf-0084.jpg)

![CS231n%20LEC10%20eb9e2b5eba9b421c8ec6ef272fd20e0e/cbbf2cf5-0515-420f-921d-20b653cee8db.pdf-0085.jpg](CS231n%20LEC10%20eb9e2b5eba9b421c8ec6ef272fd20e0e/cbbf2cf5-0515-420f-921d-20b653cee8db.pdf-0085.jpg)

![CS231n%20LEC10%20eb9e2b5eba9b421c8ec6ef272fd20e0e/cbbf2cf5-0515-420f-921d-20b653cee8db.pdf-0086.jpg](CS231n%20LEC10%20eb9e2b5eba9b421c8ec6ef272fd20e0e/cbbf2cf5-0515-420f-921d-20b653cee8db.pdf-0086.jpg)

모델이 Caption을 만들기 위해 이미지의 Attention을 이동시킨다

다양한 위치에 Attention을 주는데, 의미 있는 부분에 Attention을 준다.

- 번외. Soft/Hard Attention.

    ["Soft & hard attention"](https://jhui.github.io/2017/03/15/Soft-and-hard-attention/)

    [Show, Attend and Tell : Image Captioning에서 Soft Attention, Hard Attention](https://ahjeong.tistory.com/8)

    1. Soft Attention.

        모든 특징(Grid of Vector)과 모든 이미지 위치 사이에 Weighted Combination을 취한다.

    2. Hard Attention.

        하나의 특징과 하나의 이미지 위치 사이에서만 Weighted Combination을 취한다.

- Question.

    Q: "CNN Output과 RNN Output을 어떻게 합치는지?, Encoded Image와 Encoded Question을 어떻게 조합하는지?"

    더 복잡한 조합도 있지만, concat으로 붙여서 FC-layer의 입력으로 만드는 것이 가장 흔한 방법이다.

## Visual Question Answering.

![CS231n%20LEC10%20eb9e2b5eba9b421c8ec6ef272fd20e0e/cbbf2cf5-0515-420f-921d-20b653cee8db.pdf-0087.jpg](CS231n%20LEC10%20eb9e2b5eba9b421c8ec6ef272fd20e0e/cbbf2cf5-0515-420f-921d-20b653cee8db.pdf-0087.jpg)

![CS231n%20LEC10%20eb9e2b5eba9b421c8ec6ef272fd20e0e/cbbf2cf5-0515-420f-921d-20b653cee8db.pdf-0088.jpg](CS231n%20LEC10%20eb9e2b5eba9b421c8ec6ef272fd20e0e/cbbf2cf5-0515-420f-921d-20b653cee8db.pdf-0088.jpg)

RNN + Attention은 Image Captioning 뿐만 아니라 VQA 같은 것을 가능하게 한다.

## Multi-layer RNN.

![CS231n%20LEC10%20eb9e2b5eba9b421c8ec6ef272fd20e0e/cbbf2cf5-0515-420f-921d-20b653cee8db.pdf-0089.jpg](CS231n%20LEC10%20eb9e2b5eba9b421c8ec6ef272fd20e0e/cbbf2cf5-0515-420f-921d-20b653cee8db.pdf-0089.jpg)

일반적으로 2, 3, 4 Layer RNN이 적합하다.

## Vanilla RNN Gradient Flow.

![CS231n%20LEC10%20eb9e2b5eba9b421c8ec6ef272fd20e0e/cbbf2cf5-0515-420f-921d-20b653cee8db.pdf-0090.jpg](CS231n%20LEC10%20eb9e2b5eba9b421c8ec6ef272fd20e0e/cbbf2cf5-0515-420f-921d-20b653cee8db.pdf-0090.jpg)

Forwardpropagation.

1. $h_{t-1}$과 $x_t$를 입력받아 Stack 한다.
2. 가중치 행렬 W와 Dot Product, Matmul 한다.
3. tanh을 적용하여 $h_t$를 만든다.

![CS231n%20LEC10%20eb9e2b5eba9b421c8ec6ef272fd20e0e/cbbf2cf5-0515-420f-921d-20b653cee8db.pdf-0091.jpg](CS231n%20LEC10%20eb9e2b5eba9b421c8ec6ef272fd20e0e/cbbf2cf5-0515-420f-921d-20b653cee8db.pdf-0091.jpg)

Backwardpropagation.

1. $h_t$에 대한 Loss의 미분값을 얻는다.
2. Loss에 대한 $h_{t-1}$의 미분값을 얻는다.
3. tanh Gate를 타고 Dot Product, Matmul Gate를 통과한다.

    행렬 곱 연산의 Backprop는 Transpose W를 곱하게 된다.

![CS231n%20LEC10%20eb9e2b5eba9b421c8ec6ef272fd20e0e/cbbf2cf5-0515-420f-921d-20b653cee8db.pdf-0094.jpg](CS231n%20LEC10%20eb9e2b5eba9b421c8ec6ef272fd20e0e/cbbf2cf5-0515-420f-921d-20b653cee8db.pdf-0094.jpg)

![CS231n%20LEC10%20eb9e2b5eba9b421c8ec6ef272fd20e0e/cbbf2cf5-0515-420f-921d-20b653cee8db.pdf-0095.jpg](CS231n%20LEC10%20eb9e2b5eba9b421c8ec6ef272fd20e0e/cbbf2cf5-0515-420f-921d-20b653cee8db.pdf-0095.jpg)

각 Cell을 통과할 때마다 Transpose W를 곱하기 때문에 깊은 Layer에서 문제가 발생한다.

1. Largest Singular Value > 1 → Exploding Gradients.

    Gradeint Clipping: Heuristics한 방법으로, L2 norm이 특정 임계값을 넘지 못하도록 조정한다. 그리 좋은 방법은 아니지만, RNN에서 많이 사용된다.

2. Largest Singular Value < 1 → Vanishing Gradients.

    RNN Architecture를 LSTM과 같이 바꿔야 한다.

## LSTM(Long Short Term Memory).

![CS231n%20LEC10%20eb9e2b5eba9b421c8ec6ef272fd20e0e/cbbf2cf5-0515-420f-921d-20b653cee8db.pdf-0096.jpg](CS231n%20LEC10%20eb9e2b5eba9b421c8ec6ef272fd20e0e/cbbf2cf5-0515-420f-921d-20b653cee8db.pdf-0096.jpg)

LSTM은 Fancier한 RNN이다.

Exploding & Vanishing Gradients 문제를 해결하고자 만들어졌다.

![CS231n%20LEC10%20eb9e2b5eba9b421c8ec6ef272fd20e0e/cbbf2cf5-0515-420f-921d-20b653cee8db.pdf-0097.jpg](CS231n%20LEC10%20eb9e2b5eba9b421c8ec6ef272fd20e0e/cbbf2cf5-0515-420f-921d-20b653cee8db.pdf-0097.jpg)

LSTM은 Cell 하나에 Hidden State가 두 개이다.

1. $h_t$(Hidden State): RNN과 같다.
2. $c_t$(Cell State): LSTM 내부에만 존재하고, 밖에 노출되지 않는다.

W에는 총 4개의 Gates가 있고, Gates 가중치 행렬을 합쳐 놓은 것.

1. Input Gate(i): 입력 $x_t$에 대한 가중치이다.
2. Forget Gate(f): 이전 Cell의 정보를 얼마나 잊는지에 대한 가중치이다.
3. Output Gate(o): Cell State를 얼마나 밖에 노출할 지에 대한 가중치이다.
4. Gate Gate(g): Input Cell을 얼마나 포함시킬지에 대한 가중치이다.
5. i, f, o는 Sigmoid를 사용하고, g는 tanh를 사용한다.

![CS231n%20LEC10%20eb9e2b5eba9b421c8ec6ef272fd20e0e/cbbf2cf5-0515-420f-921d-20b653cee8db.pdf-0098.jpg](CS231n%20LEC10%20eb9e2b5eba9b421c8ec6ef272fd20e0e/cbbf2cf5-0515-420f-921d-20b653cee8db.pdf-0098.jpg)

1. $h_{t-1}$, $c_{t-1}$ 두 개를 입력받아 Stack 한다.
2. 가중치 행렬 W와 행렬 곱 연산을 진행하여 Gates를 만든다.
3. i, f, o, g 4개의 Gates로 Element-wise Multiplication 등을 수행하여  $c_t$를 업데이트한다. Gates의 출력은 $h_t$ 크기와 같다.
4. $c_t$와 o Gate를 계산하여 $h_t$를 업데이트한다.

$$c_t = f \times c_{t-1} + i \times g$$

$f \times c_{t-1}$: 이전 Cell State를 기억 여부를 결정한다.

$i \times g$: ±1까지 Cell State의 각 요소를 증가시키거나 감소시킬 수 있다.

Cell State의 각 요소는 Scaler Integer Counters처럼 값이 증가하거나 감소할 수 있다.

![CS231n%20LEC10%20eb9e2b5eba9b421c8ec6ef272fd20e0e/cbbf2cf5-0515-420f-921d-20b653cee8db.pdf-0099.jpg](CS231n%20LEC10%20eb9e2b5eba9b421c8ec6ef272fd20e0e/cbbf2cf5-0515-420f-921d-20b653cee8db.pdf-0099.jpg)

![CS231n%20LEC10%20eb9e2b5eba9b421c8ec6ef272fd20e0e/cbbf2cf5-0515-420f-921d-20b653cee8db.pdf-0102.jpg](CS231n%20LEC10%20eb9e2b5eba9b421c8ec6ef272fd20e0e/cbbf2cf5-0515-420f-921d-20b653cee8db.pdf-0102.jpg)

Cell State의 Upstream Gradient는 두 갈래로 복사된다.

따라서 Element-wise Multipy로 전달되고, Gradient는 Upstream Gradient와 f Gate의 Element_wise 곱이다.

장점.

1. Matrix Multipy가 아닌 Element-wise Multipy이다.
2. 매 Step 다른 값의 f Gate와 곱해질 수 있다.

    RNN처럼 동일한 가중치 행렬만을 곱하는 것이 아니기 때문에, Exploding & Vanishing Gradients 문제가 해결된다.

3. LSTM에서는 tanh를 한 번만 거치면 된다.

- Question.

    Q: "궁극적으로 W를 업데이트해야 할 텐데, W에 대한 Gradient는 어떻게 되는지?"

    각 Step마다 W에 대한 Local Gradient는 현재 Cell State와 Hidden State로부터 전달된다.

    LSTM은 Cell State로 잘 전달되므로 Local Gradient도 깔끔하다.

    Q: "Non Linearities가 있으므로 여전히 Vanishing Gradient가 생길 수 있는지?"

    가능성은 존재한다. 그래서 f Gate의 Biases를 양수로 초기화하곤 한다.

    학습 초기 흐름이 원활하면, 학습이 진행되면서 맞는 흐름을 찾아갈 것이다.

    (+ ResNet의 Backprop에서 Identity Mapping이 Highway 역할을 한 것처럼, LSTM도 같다.)

    (+ Highway Networks라는 LSTM와 ResNet의 중간 버전도 있다.

    + 모든 Layer에서 Candidate Activation과 Gating Function을 계산한다.

    + Gating Function은 이전 입력과 CNN등에서 산출된 Candidate Activation 사이에서 Interprelates 역할을 한다.)

## Other RNN Variations.

![CS231n%20LEC10%20eb9e2b5eba9b421c8ec6ef272fd20e0e/cbbf2cf5-0515-420f-921d-20b653cee8db.pdf-0103.jpg](CS231n%20LEC10%20eb9e2b5eba9b421c8ec6ef272fd20e0e/cbbf2cf5-0515-420f-921d-20b653cee8db.pdf-0103.jpg)

1. GRU(Gated Recurrent Unit).

    Element-wise Multipy를 확인할 수 있다.

2. LSTM: A Search Space Odyssey.

    LSTM의 수식을 바꾸며 실험하는 논문이다.

3. Google Paper.

    Evolutionary Search 기법으로 RNN Architecture를 실험한다.

## Summary.

![CS231n%20LEC10%20eb9e2b5eba9b421c8ec6ef272fd20e0e/cbbf2cf5-0515-420f-921d-20b653cee8db.pdf-0104.jpg](CS231n%20LEC10%20eb9e2b5eba9b421c8ec6ef272fd20e0e/cbbf2cf5-0515-420f-921d-20b653cee8db.pdf-0104.jpg)

## 링크.

[https://www.youtube.com/watch?v=6niqTuYFZLQ&list=PL3FW7Lu3i5JvHM8ljYj-zLfQRF3EO8sYv&index=10](https://www.youtube.com/watch?v=6niqTuYFZLQ&list=PL3FW7Lu3i5JvHM8ljYj-zLfQRF3EO8sYv&index=10)

[CS231n: Convolutional Neural Networks for Visual Recognition](http://cs231n.stanford.edu/2017/)

[](http://cs231n.stanford.edu/slides/2017/cs231n_2017_lecture10.pdf)

[soline013/CS231N_17_KOR_SUB](https://github.com/soline013/CS231N_17_KOR_SUB/blob/master/kor/Lecture%2010%20%20%20Recurrent%20Neural%20Networks.ko.srt)