# CS231n LEC10.
## Stanford University CS231n, Spring 2017.
**Recurrent Neural Networks.**

## Recall from last time.

![Untitled](https://user-images.githubusercontent.com/66259854/106297222-d1890880-6295-11eb-939b-6eacef85b61d.png)
![Untitled 1](https://user-images.githubusercontent.com/66259854/106297221-d0f07200-6295-11eb-80b3-91b9dd1be445.png)
![cbbf2cf5-0515-420f-921d-20b653cee8db pdf-0006](https://user-images.githubusercontent.com/66259854/106296905-938be480-6295-11eb-86b9-2aa06d16ee4a.jpg)
![cbbf2cf5-0515-420f-921d-20b653cee8db pdf-0007](https://user-images.githubusercontent.com/66259854/106296916-9555a800-6295-11eb-8950-4b145ac105b6.jpg)
![cbbf2cf5-0515-420f-921d-20b653cee8db pdf-0008](https://user-images.githubusercontent.com/66259854/106296918-95ee3e80-6295-11eb-933f-1ffc54c7142e.jpg)
![cbbf2cf5-0515-420f-921d-20b653cee8db pdf-0009](https://user-images.githubusercontent.com/66259854/106296922-9686d500-6295-11eb-9de6-bbbe85678480.jpg)

## Recurrent Neural Networks: Process Sequences.

![cbbf2cf5-0515-420f-921d-20b653cee8db pdf-0011](https://user-images.githubusercontent.com/66259854/106296926-9686d500-6295-11eb-8519-63d6f4288be2.jpg)

![cbbf2cf5-0515-420f-921d-20b653cee8db pdf-0012](https://user-images.githubusercontent.com/66259854/106296930-97b80200-6295-11eb-815d-46d0f895bfb5.jpg)

![cbbf2cf5-0515-420f-921d-20b653cee8db pdf-0013](https://user-images.githubusercontent.com/66259854/106296934-98509880-6295-11eb-9ece-d05d512614b0.jpg)

![cbbf2cf5-0515-420f-921d-20b653cee8db pdf-0014](https://user-images.githubusercontent.com/66259854/106296937-98e92f00-6295-11eb-8781-f6b9f6591953.jpg)

![cbbf2cf5-0515-420f-921d-20b653cee8db pdf-0015](https://user-images.githubusercontent.com/66259854/106296938-9981c580-6295-11eb-85d0-f798a3098053.jpg)

가변 입력과 가변 출력으로 다양한 선택지를 제공한다.

## Sequential Processing of Non-Sequence Data.

![cbbf2cf5-0515-420f-921d-20b653cee8db pdf-0016](https://user-images.githubusercontent.com/66259854/106296941-9981c580-6295-11eb-94ad-68c6dd1932f6.jpg)

![cbbf2cf5-0515-420f-921d-20b653cee8db pdf-0017](https://user-images.githubusercontent.com/66259854/106296943-9a1a5c00-6295-11eb-9da0-5e1de0a26df2.jpg)

입출력은 고정된 길이지만, 가변 과정인 경우에도 사용할 수 있다.

## RNN.

![cbbf2cf5-0515-420f-921d-20b653cee8db pdf-0019](https://user-images.githubusercontent.com/66259854/106296946-9ab2f280-6295-11eb-9c00-0e7f26ba6644.jpg)

1. RNN이 입력을 받는다.
2. Hidden State를 업데이트한다.
3. 출력 값을 내보낸다.

![cbbf2cf5-0515-420f-921d-20b653cee8db pdf-0020](https://user-images.githubusercontent.com/66259854/106296947-9ab2f280-6295-11eb-92cb-3ea3caf7402b.jpg)

RNN Block은 재귀적인 관계를 함수 f로 연산할 수 있다.

1. $h_{t-1}$: 이전 상태의 Hidden State.
2. $x_t$: 현재 상태의 입력.
3. $h_t$: 다음 상태의 Hidden State.

![cbbf2cf5-0515-420f-921d-20b653cee8db pdf-0021](https://user-images.githubusercontent.com/66259854/106296948-9b4b8900-6295-11eb-8e3d-6c2183d481ba.jpg)

RNN에서 Y를 가지려면 $h_t$를 입력받는 FC-layer가 필요하다.

함수 f와 Parameter W는 매 스텝 동일하다.

![cbbf2cf5-0515-420f-921d-20b653cee8db pdf-0022](https://user-images.githubusercontent.com/66259854/106296952-9be41f80-6295-11eb-8132-23eaec538c71.jpg)

1. 가중치 행렬 $W_{xh}$와 입력 $x_t$
2. 가중치 행렬 $W_{hh}$와 이전 Hidden State $h_{t-1}$
3. Non-linearity 구현을 위한 tanh → `LSTM에서 부가 설명.`
4. 가중치 행렬 $W_{hy}$와 Hidden State $h_t$

## RNN Computational Graph.

### Vanilla.

![cbbf2cf5-0515-420f-921d-20b653cee8db pdf-0023](https://user-images.githubusercontent.com/66259854/106296957-9c7cb600-6295-11eb-906a-9445a0f81c19.jpg)

![cbbf2cf5-0515-420f-921d-20b653cee8db pdf-0024](https://user-images.githubusercontent.com/66259854/106296961-9d154c80-6295-11eb-811b-138130ff4c8a.jpg)

![cbbf2cf5-0515-420f-921d-20b653cee8db pdf-0025](https://user-images.githubusercontent.com/66259854/106296964-9dade300-6295-11eb-9b7d-8f4ba4a9d64d.jpg)

![cbbf2cf5-0515-420f-921d-20b653cee8db pdf-0026](https://user-images.githubusercontent.com/66259854/106296967-9e467980-6295-11eb-8d0e-4cbb50534a50.jpg)

가중치 행렬 W는 항상 동일하다.

### Many to Many.

![cbbf2cf5-0515-420f-921d-20b653cee8db pdf-0027](https://user-images.githubusercontent.com/66259854/106296969-9e467980-6295-11eb-87c4-79ca5e5d4e0d.jpg)

![cbbf2cf5-0515-420f-921d-20b653cee8db pdf-0028](https://user-images.githubusercontent.com/66259854/106296989-a2729700-6295-11eb-8c9f-596721db0906.jpg)

![cbbf2cf5-0515-420f-921d-20b653cee8db pdf-0029](https://user-images.githubusercontent.com/66259854/106296998-a56d8780-6295-11eb-9d7e-a7f032199e78.jpg)

Back Prop에서는 $\frac{dL}{dW}$를 구해야 한다.

1. RNN에서 Backward를 위한 W의 Gradient를 구하려면 각 스텝의 Local Gradient를 계산하고, 모두 더한다.
2. 각 스텝의 개별 Loss를 구하면, RNN의 Loss는 개별 Loss의 합이다.

### Many to One.

![cbbf2cf5-0515-420f-921d-20b653cee8db pdf-0030](https://user-images.githubusercontent.com/66259854/106297001-a6061e00-6295-11eb-83fb-8d97a86cf8b3.jpg)

### One to Many.

![cbbf2cf5-0515-420f-921d-20b653cee8db pdf-0031](https://user-images.githubusercontent.com/66259854/106297004-a69eb480-6295-11eb-8487-8992fd18388d.jpg)

고정 입력은 모델의 Initial Hidden State를 초기화하는 용도.

$h_0$는 대부분 0으로 초기화하는데, 이때도 0으로 초기화할까?

### seq2seq(Sequence to Sequence).

![cbbf2cf5-0515-420f-921d-20b653cee8db pdf-0032](https://user-images.githubusercontent.com/66259854/106297007-a7374b00-6295-11eb-9048-f44b2a431426.jpg)

![cbbf2cf5-0515-420f-921d-20b653cee8db pdf-0033](https://user-images.githubusercontent.com/66259854/106297010-a7cfe180-6295-11eb-80ad-8f99e323f936.jpg)

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

![cbbf2cf5-0515-420f-921d-20b653cee8db pdf-0034](https://user-images.githubusercontent.com/66259854/106297014-a8687800-6295-11eb-872b-ecb0b389ed2b.jpg)

![cbbf2cf5-0515-420f-921d-20b653cee8db pdf-0035](https://user-images.githubusercontent.com/66259854/106297018-a8687800-6295-11eb-9d6b-e603a5bafc0f.jpg)

![cbbf2cf5-0515-420f-921d-20b653cee8db pdf-0036](https://user-images.githubusercontent.com/66259854/106297023-aacad200-6295-11eb-988e-aa7983d6944e.jpg)

## Example: Character Level Language Model Sampling.

![cbbf2cf5-0515-420f-921d-20b653cee8db pdf-0037](https://user-images.githubusercontent.com/66259854/106297026-ab636880-6295-11eb-9edc-c48f80157512.jpg)

![cbbf2cf5-0515-420f-921d-20b653cee8db pdf-0038](https://user-images.githubusercontent.com/66259854/106297027-ab636880-6295-11eb-8167-073ea75696ab.jpg)

![cbbf2cf5-0515-420f-921d-20b653cee8db pdf-0039](https://user-images.githubusercontent.com/66259854/106297029-abfbff00-6295-11eb-9e8f-fec06225b2d8.jpg)

![cbbf2cf5-0515-420f-921d-20b653cee8db pdf-0040](https://user-images.githubusercontent.com/66259854/106297030-ac949580-6295-11eb-82e3-a233fa9b0ecb.jpg)

모든 문자에 대한 Score를 Sampling에 이용한다.

Score를 확률분포로 표현하기 위해 Softmax를 사용한다.

'e'의 확률은 13%에 불과하지만, 'e'가 Sampling 되었다.

### Question.

Q: "가장 높은 스코어를 선택하지 않고 확률분포에서 샘플링하는 이유?"

Argmax Probability, Sampling 모두 사용할 수 있지만, Sampling을 사용하면 다양한 결과를 얻을 수 있다.

위의 예제의 경우, Sampling을 사용했기에 올바른 결과를 얻을 수 있었다.

Q: "Test Time에 One Hot 대신 Softmax를 입력으로 사용할 수 있는가?"

첫 번째 문제는 입력이 Train에서의 입력과 달라지는 것이다.

두 번째 문제는 실제로 Vocabularies가 매우 크다는 것이다.

실제로는 One Hot Vector를 Sparse Vector로 처리한다.

Sparse Vector Operation: 공간 절약을 위해 0이 아닌 값만 저장.

## Truncated Backpropagation.

![cbbf2cf5-0515-420f-921d-20b653cee8db pdf-0041](https://user-images.githubusercontent.com/66259854/106297034-ad2d2c00-6295-11eb-8878-475f97f1d47e.jpg)

Sequence가 긴 경우 학습이 느려지는 문제가 발생한다.

![cbbf2cf5-0515-420f-921d-20b653cee8db pdf-0042](https://user-images.githubusercontent.com/66259854/106297037-adc5c280-6295-11eb-9000-6ac981474552.jpg)

![cbbf2cf5-0515-420f-921d-20b653cee8db pdf-0043](https://user-images.githubusercontent.com/66259854/106297042-adc5c280-6295-11eb-93ad-36fa6f63eaac.jpg)

![cbbf2cf5-0515-420f-921d-20b653cee8db pdf-0044](https://user-images.githubusercontent.com/66259854/106297044-ae5e5900-6295-11eb-9e59-e909ebe161b9.jpg)

따라서 Truncated Backpropagation을 사용하는데, Sequence를 일정 길이로 나누고 Loss를 구한다.

SDG에서 사용하는 Mini Batch와 같은 방식이다.

### Question.

Q: "RNN이 Markov Assumption(마르코프 가정)을 따르는가?"

RNN은 이전 Hidden State를 계속해서 앞으로 가져가기 때문에 따르지 않는다.

### 번외. Markov Assumption.

상태가 연속적인 시간에 따라 이어질 때 어떤 시점의 상태는 그 시점 바로 이전의 상태에만 영향을 받는다는 가정.

e.g. 오늘 날씨는 어제 날씨에만 영향을 받는다.

[컴공의 공부노트 : 네이버 블로그](https://blog.naver.com/kkang9901/222029504981)

[Markov Model](http://blog.daum.net/hazzling/15605818)

## min-char-rnn, Andrej.

![cbbf2cf5-0515-420f-921d-20b653cee8db pdf-0045](https://user-images.githubusercontent.com/66259854/106297048-b0281c80-6295-11eb-972f-09d89b3de05e.jpg)

Vocabulary를 만들고, Truncated Backpropagation을 수행하는 모델.

### Shakespeare RNN.

![cbbf2cf5-0515-420f-921d-20b653cee8db pdf-0046](https://user-images.githubusercontent.com/66259854/106297053-b0c0b300-6295-11eb-8dcd-cc6bddb13c42.jpg)

![cbbf2cf5-0515-420f-921d-20b653cee8db pdf-0047](https://user-images.githubusercontent.com/66259854/106297057-b1594980-6295-11eb-894e-57564f495e98.jpg)

![cbbf2cf5-0515-420f-921d-20b653cee8db pdf-0048](https://user-images.githubusercontent.com/66259854/106297058-b1f1e000-6295-11eb-8429-94463cf55165.jpg)

### Algebraic Topology.

![cbbf2cf5-0515-420f-921d-20b653cee8db pdf-0049](https://user-images.githubusercontent.com/66259854/106297059-b1f1e000-6295-11eb-9691-7f56fb2c3fd9.jpg)

![cbbf2cf5-0515-420f-921d-20b653cee8db pdf-0050](https://user-images.githubusercontent.com/66259854/106297061-b28a7680-6295-11eb-851e-e7d5bcb6711e.jpg)

![cbbf2cf5-0515-420f-921d-20b653cee8db pdf-0051](https://user-images.githubusercontent.com/66259854/106297064-b3230d00-6295-11eb-82cc-a229c6c3d73c.jpg)

### Linux Kernel.

![cbbf2cf5-0515-420f-921d-20b653cee8db pdf-0052](https://user-images.githubusercontent.com/66259854/106297068-b3bba380-6295-11eb-8ba3-3b6f7bb92737.jpg)

![cbbf2cf5-0515-420f-921d-20b653cee8db pdf-0053](https://user-images.githubusercontent.com/66259854/106297071-b3bba380-6295-11eb-998b-54b60200cd5e.jpg)

![cbbf2cf5-0515-420f-921d-20b653cee8db pdf-0054](https://user-images.githubusercontent.com/66259854/106297074-b4543a00-6295-11eb-9d95-f52c23ca24e2.jpg)

![cbbf2cf5-0515-420f-921d-20b653cee8db pdf-0055](https://user-images.githubusercontent.com/66259854/106297088-b61dfd80-6295-11eb-8220-40e96f96ae08.jpg)

## Searching for Interpretable Cells.

![cbbf2cf5-0515-420f-921d-20b653cee8db pdf-0056](https://user-images.githubusercontent.com/66259854/106297090-b6b69400-6295-11eb-8626-e6c1b0cb09f5.jpg)

Hidden Layer의 Vector를 추출하면 해석 가능한 어떤 의미 있는 값이 나오지 않을까 추측하였다.

![cbbf2cf5-0515-420f-921d-20b653cee8db pdf-0057](https://user-images.githubusercontent.com/66259854/106297094-b74f2a80-6295-11eb-9c1f-2843c772f79e.jpg)

Hidden State 대부분은 의미 없는 값이 나온다.

Vector 하나를 뽑고, Sequence를 다시 Forward 한다.

각 색깔은 Sequence를 진행하는 동안 앞에서 뽑은 Vector이다.

1. 따옴표를 만나면 값이 켜져 빨간색이 되고, 따옴표가 끝나면 파란색이 된다.

    ![cbbf2cf5-0515-420f-921d-20b653cee8db pdf-0058](https://user-images.githubusercontent.com/66259854/106297096-b7e7c100-6295-11eb-8784-68ec5e3542b6.jpg)

2. 줄 바꿈을 위해 현재 줄의 단어 수를 세는 듯, 점점 빨간색으로 변하다가 다음 줄에서 파란색으로 초기화된다.

    ![cbbf2cf5-0515-420f-921d-20b653cee8db pdf-0059](https://user-images.githubusercontent.com/66259854/106297100-b8805780-6295-11eb-8d26-4e3cc80def15.jpg)

3. Linux 코드를 학습시킬 때 발견한 것으로, if문의  조건부에서 값이 켜져 빨간색이 된다.

    ![cbbf2cf5-0515-420f-921d-20b653cee8db pdf-0060](https://user-images.githubusercontent.com/66259854/106297107-b918ee00-6295-11eb-96ba-5f786ffd3e67.jpg)

4. Linux 코드 내에서 Quote나 Comment를 찾는다.

    ![cbbf2cf5-0515-420f-921d-20b653cee8db pdf-0061](https://user-images.githubusercontent.com/66259854/106297108-b9b18480-6295-11eb-8815-5a3f77e0c302.jpg)

5. Linux 코드 내에서 들여쓰기에 따라 빨간색으로 변한다.

    ![cbbf2cf5-0515-420f-921d-20b653cee8db pdf-0062](https://user-images.githubusercontent.com/66259854/106297109-ba4a1b00-6295-11eb-9222-0341ee30be81.jpg)

## Image Captioning By using RNN.


![cbbf2cf5-0515-420f-921d-20b653cee8db pdf-0063](https://user-images.githubusercontent.com/66259854/106297112-bb7b4800-6295-11eb-8f61-c45bb13149ab.jpg)

![cbbf2cf5-0515-420f-921d-20b653cee8db pdf-0064](https://user-images.githubusercontent.com/66259854/106297121-bcac7500-6295-11eb-9c72-6c841f7e981a.jpg)

1. CNN은 요약된 이미지 정보 Vector를 출력하고, 이 Vector는 RNN의 h가 된다.
2. RNN은 Caption에 사용할 문자들을 만들어낸다.

---

1. Softmax를 사용하지 않고 4,096-dim Vector를 사용한다.

    ![cbbf2cf5-0515-420f-921d-20b653cee8db pdf-0067](https://user-images.githubusercontent.com/66259854/106297125-bd450b80-6295-11eb-81a1-90e29bb95fb2.jpg)

2. RNN의 입력은 "Hey, this is the strat of a sentence. Please start generating some text conditioned on this image information."이라는 START Token이다.

    ![cbbf2cf5-0515-420f-921d-20b653cee8db pdf-0068](https://user-images.githubusercontent.com/66259854/106297127-bddda200-6295-11eb-9732-bbd17e6c2649.jpg)

3. h는 기존 가중치 행렬에 이미지 정보를 나타내는 가중치 행렬을 더한다.

    ![cbbf2cf5-0515-420f-921d-20b653cee8db pdf-0069](https://user-images.githubusercontent.com/66259854/106297130-be763880-6295-11eb-817b-fd65bfc85a96.jpg)

4. END Token이 Sampling되면 종료되고 Caption이 완성된다.

    ![cbbf2cf5-0515-420f-921d-20b653cee8db pdf-0070](https://user-images.githubusercontent.com/66259854/106297131-bf0ecf00-6295-11eb-901d-be0275f80132.jpg)

    ![cbbf2cf5-0515-420f-921d-20b653cee8db pdf-0071](https://user-images.githubusercontent.com/66259854/106297133-bfa76580-6295-11eb-8ff5-e28d99c4371b.jpg)

    ...

    ![cbbf2cf5-0515-420f-921d-20b653cee8db pdf-0074](https://user-images.githubusercontent.com/66259854/106297135-c03ffc00-6295-11eb-8c25-ed22028eefcf.jpg)

    Train에서 모든 Caption의 마지막에 END Token을 추가한다.

    그러면 Test에서 자동으로 마지막에 END Token을 샘플링한다.

5. Example.

    ![cbbf2cf5-0515-420f-921d-20b653cee8db pdf-0075](https://user-images.githubusercontent.com/66259854/106297136-c03ffc00-6295-11eb-933a-febf574628cf.jpg)

    ![cbbf2cf5-0515-420f-921d-20b653cee8db pdf-0076](https://user-images.githubusercontent.com/66259854/106297141-c2a25600-6295-11eb-9074-16f6a011167d.jpg)

## Attention.

![cbbf2cf5-0515-420f-921d-20b653cee8db pdf-0077](https://user-images.githubusercontent.com/66259854/106297143-c33aec80-6295-11eb-9cdc-94aa23ddc719.jpg)

1. CNN으로 하나의 벡터가 아닌 각 벡터가 공간 정보를 갖는 Grid of Vector, $L \times D$를 만든다.

    ![cbbf2cf5-0515-420f-921d-20b653cee8db pdf-0078](https://user-images.githubusercontent.com/66259854/106297145-c33aec80-6295-11eb-82b8-11cca0677422.jpg)

2. $h_0$에서 이미지 위치에 대한 분포, $a1$을 계산한다.

    ![cbbf2cf5-0515-420f-921d-20b653cee8db pdf-0079](https://user-images.githubusercontent.com/66259854/106297146-c3d38300-6295-11eb-8e43-9ad341f5f5d4.jpg)

3. Grid of Vector와 분포를 계산하여 이미지 Attention, $z1$을 만든다.

    ![cbbf2cf5-0515-420f-921d-20b653cee8db pdf-0080](https://user-images.githubusercontent.com/66259854/106297149-c46c1980-6295-11eb-8899-4b2127228d0a.jpg)

4. Attention $z1$과 First Word $y1$이 다음 Step의 입력으로 들어간다.

    ![cbbf2cf5-0515-420f-921d-20b653cee8db pdf-0081](https://user-images.githubusercontent.com/66259854/106297151-c46c1980-6295-11eb-8a97-ef35ba0dc222.jpg)

5. 이미지 위치에 대한 분포 $a2$와, 각 단어들의 분포 $d1$을 계산한다.

    ![cbbf2cf5-0515-420f-921d-20b653cee8db pdf-0082](https://user-images.githubusercontent.com/66259854/106297154-c504b000-6295-11eb-8734-8d07ae2713af.jpg)

6. 또 다시 Grid of Vector와 분포를 계산하여 이미지 Attention, $z2$를 만들고 Step을 반복한다.

    ![cbbf2cf5-0515-420f-921d-20b653cee8db pdf-0083](https://user-images.githubusercontent.com/66259854/106297158-c59d4680-6295-11eb-9bf9-4a4bab49c2e3.jpg)

    ![cbbf2cf5-0515-420f-921d-20b653cee8db pdf-0084](https://user-images.githubusercontent.com/66259854/106297160-c635dd00-6295-11eb-99ef-4dc0aefa4f02.jpg)

![cbbf2cf5-0515-420f-921d-20b653cee8db pdf-0085](https://user-images.githubusercontent.com/66259854/106297164-c6ce7380-6295-11eb-83a8-bfc44587d19b.jpg)

![cbbf2cf5-0515-420f-921d-20b653cee8db pdf-0086](https://user-images.githubusercontent.com/66259854/106297169-c8983700-6295-11eb-85b7-ecd14449c590.jpg)

모델이 Caption을 만들기 위해 이미지의 Attention을 이동시킨다

다양한 위치에 Attention을 주는데, 의미 있는 부분에 Attention을 준다.

### 번외. Soft/Hard Attention.

["Soft & hard attention"](https://jhui.github.io/2017/03/15/Soft-and-hard-attention/)

[Show, Attend and Tell : Image Captioning에서 Soft Attention, Hard Attention](https://ahjeong.tistory.com/8)

1. Soft Attention.

    모든 특징(Grid of Vector)과 모든 이미지 위치 사이에 Weighted Combination을 취한다.

2. Hard Attention.

    하나의 특징과 하나의 이미지 위치 사이에서만 Weighted Combination을 취한다.

### Question.

Q: "CNN Output과 RNN Output을 어떻게 합치는지?, Encoded Image와 Encoded Question을 어떻게 조합하는지?"

더 복잡한 조합도 있지만, concat으로 붙여서 FC-layer의 입력으로 만드는 것이 가장 흔한 방법이다.

## Visual Question Answering.

![cbbf2cf5-0515-420f-921d-20b653cee8db pdf-0087](https://user-images.githubusercontent.com/66259854/106297170-c930cd80-6295-11eb-9d0f-c73577d41cce.jpg)

![cbbf2cf5-0515-420f-921d-20b653cee8db pdf-0088](https://user-images.githubusercontent.com/66259854/106297175-c9c96400-6295-11eb-9094-a40035d02eff.jpg)

RNN + Attention은 Image Captioning 뿐만 아니라 VQA 같은 것을 가능하게 한다.

## Multi-layer RNN.

![cbbf2cf5-0515-420f-921d-20b653cee8db pdf-0089](https://user-images.githubusercontent.com/66259854/106297181-ca61fa80-6295-11eb-8107-b7aedec5dc8c.jpg)

일반적으로 2, 3, 4 Layer RNN이 적합하다.

## Vanilla RNN Gradient Flow.

![cbbf2cf5-0515-420f-921d-20b653cee8db pdf-0090](https://user-images.githubusercontent.com/66259854/106297185-cafa9100-6295-11eb-9449-d7b0c8b2d84d.jpg)

Forwardpropagation.

1. $h_{t-1}$과 $x_t$를 입력받아 Stack 한다.
2. 가중치 행렬 W와 Dot Product, Matmul 한다.
3. tanh을 적용하여 $h_t$를 만든다.

![cbbf2cf5-0515-420f-921d-20b653cee8db pdf-0091](https://user-images.githubusercontent.com/66259854/106297189-cb932780-6295-11eb-9b6e-9068e60bb9ab.jpg)

Backwardpropagation.

1. $h_t$에 대한 Loss의 미분값을 얻는다.

2. Loss에 대한 $h_{t-1}$의 미분값을 얻는다.

3. tanh Gate를 타고 Dot Product, Matmul Gate를 통과한다.

    행렬 곱 연산의 Backprop는 Transpose W를 곱하게 된다.

![cbbf2cf5-0515-420f-921d-20b653cee8db pdf-0094](https://user-images.githubusercontent.com/66259854/106297191-cb932780-6295-11eb-9177-5d26524fac35.jpg)

![cbbf2cf5-0515-420f-921d-20b653cee8db pdf-0095](https://user-images.githubusercontent.com/66259854/106297194-cc2bbe00-6295-11eb-8170-b1d432693f4a.jpg)

각 Cell을 통과할 때마다 Transpose W를 곱하기 때문에 깊은 Layer에서 문제가 발생한다.

1. Largest Singular Value > 1 → Exploding Gradients.

    Gradeint Clipping: Heuristics한 방법으로, L2 norm이 특정 임계값을 넘지 못하도록 조정한다. 그리 좋은 방법은 아니지만, RNN에서 많이 사용된다.

2. Largest Singular Value < 1 → Vanishing Gradients.

    RNN Architecture를 LSTM과 같이 바꿔야 한다.

## LSTM(Long Short Term Memory).

![cbbf2cf5-0515-420f-921d-20b653cee8db pdf-0096](https://user-images.githubusercontent.com/66259854/106297197-cd5ceb00-6295-11eb-9c1e-23bc32a6c979.jpg)

LSTM은 Fancier한 RNN이다.

Exploding & Vanishing Gradients 문제를 해결하고자 만들어졌다.

![cbbf2cf5-0515-420f-921d-20b653cee8db pdf-0097](https://user-images.githubusercontent.com/66259854/106297199-cdf58180-6295-11eb-9112-5c76978fd3d7.jpg)

LSTM은 Cell 하나에 Hidden State가 두 개이다.

1. $h_t$(Hidden State): RNN과 같다.
2. $c_t$(Cell State): LSTM 내부에만 존재하고, 밖에 노출되지 않는다.

W에는 총 4개의 Gates가 있고, Gates 가중치 행렬을 합쳐 놓은 것.

1. Input Gate(i): 입력 $x_t$에 대한 가중치이다.
2. Forget Gate(f): 이전 Cell의 정보를 얼마나 잊는지에 대한 가중치이다.
3. Output Gate(o): Cell State를 얼마나 밖에 노출할 지에 대한 가중치이다.
4. Gate Gate(g): Input Cell을 얼마나 포함시킬지에 대한 가중치이다.
5. i, f, o는 Sigmoid를 사용하고, g는 tanh를 사용한다.

![cbbf2cf5-0515-420f-921d-20b653cee8db pdf-0098](https://user-images.githubusercontent.com/66259854/106297203-ce8e1800-6295-11eb-97aa-ff6d372a62cd.jpg)

1. $h_{t-1}$, $c_{t-1}$ 두 개를 입력받아 Stack 한다.
2. 가중치 행렬 W와 행렬 곱 연산을 진행하여 Gates를 만든다.
3. i, f, o, g 4개의 Gates로 Element-wise Multiplication 등을 수행하여  $c_t$를 업데이트한다. Gates의 출력은 $h_t$ 크기와 같다.
4. $c_t$와 o Gate를 계산하여 $h_t$를 업데이트한다.

$$c_t = f \times c_{t-1} + i \times g$$

$f \times c_{t-1}$: 이전 Cell State를 기억 여부를 결정한다.

$i \times g$: ±1까지 Cell State의 각 요소를 증가시키거나 감소시킬 수 있다.

Cell State의 각 요소는 Scaler Integer Counters처럼 값이 증가하거나 감소할 수 있다.

![cbbf2cf5-0515-420f-921d-20b653cee8db pdf-0099](https://user-images.githubusercontent.com/66259854/106297207-ce8e1800-6295-11eb-972f-bbc55797d641.jpg)

![cbbf2cf5-0515-420f-921d-20b653cee8db pdf-0102](https://user-images.githubusercontent.com/66259854/106297208-cf26ae80-6295-11eb-9f16-49f8ba6269d3.jpg)

Cell State의 Upstream Gradient는 두 갈래로 복사된다.

따라서 Element-wise Multipy로 전달되고, Gradient는 Upstream Gradient와 f Gate의 Element_wise 곱이다.

장점.

1. Matrix Multipy가 아닌 Element-wise Multipy이다.
2. 매 Step 다른 값의 f Gate와 곱해질 수 있다.

    RNN처럼 동일한 가중치 행렬만을 곱하는 것이 아니기 때문에, Exploding & Vanishing Gradients 문제가 해결된다.

3. LSTM에서는 tanh를 한 번만 거치면 된다.

### Question.

Q: "궁극적으로 W를 업데이트해야 할 텐데, W에 대한 Gradient는 어떻게 되는지?"

각 Step마다 W에 대한 Local Gradient는 현재 Cell State와 Hidden State로부터 전달된다.

LSTM은 Cell State로 잘 전달되므로 Local Gradient도 깔끔하다.

Q: "Non Linearities가 있으므로 여전히 Vanishing Gradient가 생길 수 있는지?"

가능성은 존재한다. 그래서 f Gate의 Biases를 양수로 초기화하곤 한다.

학습 초기 흐름이 원활하면, 학습이 진행되면서 맞는 흐름을 찾아갈 것이다.

(+ ResNet의 Backprop에서 Identity Mapping이 Highway 역할을 한 것처럼, LSTM도 같다.)

(+ Highway Networks라는 LSTM와 ResNet의 중간 버전도 있다.

모든 Layer에서 Candidate Activation과 Gating Function을 계산한다.

Gating Function은 이전 입력과 CNN등에서 산출된 Candidate Activation 사이에서 Interprelates 역할을 한다.)

## Other RNN Variations.

![cbbf2cf5-0515-420f-921d-20b653cee8db pdf-0103](https://user-images.githubusercontent.com/66259854/106297210-cfbf4500-6295-11eb-86eb-f5078d054c58.jpg)

1. GRU(Gated Recurrent Unit).

    Element-wise Multipy를 확인할 수 있다.

2. LSTM: A Search Space Odyssey.

    LSTM의 수식을 바꾸며 실험하는 논문이다.

3. Google Paper.

    Evolutionary Search 기법으로 RNN Architecture를 실험한다.

## Summary.

![cbbf2cf5-0515-420f-921d-20b653cee8db pdf-0104](https://user-images.githubusercontent.com/66259854/106297214-d057db80-6295-11eb-93f1-3cb9a77141fa.jpg)

## 링크.

[Lecture 10 | Recurrent Neural Networks |](https://www.youtube.com/watch?v=6niqTuYFZLQ&list=PL3FW7Lu3i5JvHM8ljYj-zLfQRF3EO8sYv&index=10)

[CS231n: Convolutional Neural Networks for Visual Recognition](http://cs231n.stanford.edu/2017/)

[CS231n 2017 Lecture10 PDF](http://cs231n.stanford.edu/slides/2017/cs231n_2017_lecture10.pdf)

[soline013/CS231N_17_KOR_SUB](https://github.com/soline013/CS231N_17_KOR_SUB/blob/master/kor/Lecture%2010%20%20%20Recurrent%20Neural%20Networks.ko.srt)