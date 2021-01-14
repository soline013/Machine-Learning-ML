"Attention is all you need"(Vaswani et al. 2017).

[Attention Is All You Need](https://arxiv.org/abs/1706.03762)

NLP에서 사용되는 모델로, 2017년 NIPS에서 Google이 소개하였다.

RNN에서 벗어나 Attention만을 사용하는 신경망을 고안하여 Multi-head Self-attention을 사용하는 모델이다.

RNN의 순차적 계산을 행렬곱을 사용하여 한 번에 처리함으로써 모든 중요 정보를 Embedding 한다.

Self-attention을 통해 같은 문장 내 모든 단어 쌍 사이의 의미, 문법 관계를 알 수 있다.

그러나 Positional Encoding만으로는 위치, 순서 정보 제공에 어려움이 있어, BERT가 등장하게 되었다. BERT에서는 Positional Embedding이다.

## Model

Inputs: Encoder / Outputs: Decoder로 나눌 수 있다.

![image](https://user-images.githubusercontent.com/66259854/104466229-56b6c100-55f8-11eb-9662-1acfd8adf05c.png)

- Input(Output) Embedding
- Positional Encoding
- (Encoder-Decoder) (Masked) Multi Head (Self) Attention
- Scaled Dot Product Attention
- Dropout
- Layer Normalization
- Sub-layer & Residual Connection
- Feed Forward
- Linear & Softmax

---

![image](https://user-images.githubusercontent.com/66259854/104466270-61715600-55f8-11eb-8a2a-ad56d09a66a0.png)

논문에서는 Encoder와 Decoder를 6개씩 쌓아 Encoding 부분과 Decoding 부분을 만들었다.

## Layer

Dropout에 대한 설명은 생략한다.

Self Attention이 아니거나, Output Embedding인 경우가 있으나 위에서 한 번 표기한 이후로 생략한다.

또한 Tensorflow와 Pytorch가 혼재되어 있다.

### Input Embedding

Word Embedding

[tf.keras.layers.Embedding | TensorFlow Core v2.3.0](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Embedding)

[Glossary of Deep Learning: Word Embedding](https://medium.com/deeper-learning/glossary-of-deep-learning-word-embedding-f90c3cec34ca)

Convert 2D sequence (batch_size, input_length)

→ 3D (batch_size, input_length, $d_{model}$)

Word Embedding은 6개 중 가장 밑에 있는 Encoder에서만 일어난다.

### Positional Encoding

![image](https://user-images.githubusercontent.com/66259854/104466284-67673700-55f8-11eb-8399-a97d4b1da32f.png)

RNN은 단어를 순차적으로 입력받기 때문에, 단어의 위치에 따라 위치 정보를 가질 수 있다.

이 특징으로 자연어 처리에서 RNN이 자주 사용되었다.

그리고 Transformer에서는 위치 정보를 위해 Positional Encoding을 사용한다.

---

$$PE_{(pos,2i)}=sin(pos/10000^{2i/d_{model}})$$

$$PE_{(pos,2i+1)}=cos(pos/10000^{2i/d_{model}})$$

$2i$ = 짝수 / $2i+1$ = 홀수

pos = Embedding Vector의 위치. ~~Sequence Batch 수.~~

index = Embedding Vector 내 Dimension Index. ~~Embedding Dimension.~~ 

### Multi Head Self Attention

![image](https://user-images.githubusercontent.com/66259854/104466694-db094400-55f8-11eb-98a6-55e0a2eb4ca3.png)

Multi Head Self Attention은 Query, Key, Value head로 나뉘고, 각각 다른 Linear Projection, Scaled Dot-Product를 진행한다. (Split을 진행하기 때문에 각각 다른 Scaled Dot-Product를 진행한다.)

이후 Concat, Linear Projection을 하는데, 이는 tf.reshape, tf.transform으로 한 번에 연산할 수 있다.

### Scaled Dot Product Attention

![image](https://user-images.githubusercontent.com/66259854/104466713-df356180-55f8-11eb-9e5c-5cd6faac972c.png)

Learnable Parameter가 없다.

기존 Additive Attention은 Attention score를 구하는 구간에 Feed Forward Layer가 있지만, Dot Product 연산으로 대체하였다.

Encoder에서 Padding을 사용하지 않도록 Padding Mask를 추가해야 한다.

### Sub-layer & Residual Connection, Layer Normalization

![image](https://user-images.githubusercontent.com/66259854/104466733-e492ac00-55f8-11eb-873e-e56a6bc92906.png)

Encoder에는 2개의 Sub-layer가 있고, Decoder에는 3개의 Sub-layer가 있다.

이 Sub-layer는 Residual Connection을 거친다.

---

[Layer Normalization](https://arxiv.org/abs/1607.06450)

$$\bar{x} = \frac{a}{\sigma}(x - \mu) + b$$

$$LayerNorm(x_i)=\gamma \ \frac{x_{i, k} - \mu_i}{\sqrt{\sigma_i^2} + \epsilon} + \beta&&
&&(\gamma=1,\ \beta=0)$$

이후 $LayerNorm(x + Sublayer(x))$ 이 식처럼 Layer Normalization을 적용한다.

LN은 Tensor의 마지막 차원에 대해서 평군과 분산을 구하고 위의 수식을 통해 값을 정규화한다.

### Feed Forward

$$FFN(x)=max(0,xW_1+b_1)W_2+b_2$$

Multi Head Attention에서 나온 Attention 정보를 정리하는 역할.

`FF - Relu - Dropout - FF` 순서의 Sequential한 구조이다.

### Linear & Softmax

![image](https://user-images.githubusercontent.com/66259854/104466753-eb212380-55f8-11eb-9f12-2e00a64433b1.png)

1. Linear Layer.

    Fully-connected로 마지막 Decoder Output을 Logits Vector에 투영시킨다.

    Logits Vector는 Output Vocabulary로 많은 단어가 들어가 있고, Vector의 각 셀은 대응하는 단어의 점수이므로 출력을 해석할 수 있다.

2. Softmax Layer.

    이 점수들을 Softmax를 통해 확률값으로 변환한다.

    Argmax를 이용하여 나온 가장 높은 확률값이 최종 Output이 된다.

## Encoder

`- Multi Head Attention(Mask)`

`- Dropout1`

`- LayerNorm with Residual Connection1 → Output`

`-`

`- Position wise Feed Forward`

`- Dropout2`

`- LayerNorm with Residual Connection2 → Output`

tf.add()로 ResNet에서 사용하는 Residual Connection을 한다.

Padding을 적용하지 못하게 Padding Mask를 넣는다.

## Decoder

`- Masked Multi Head Attention`

`- Dropout1`

`- LayerNorm with Residual Connection1 → Query`

`-`

`- Encoder-Decoder Multi Head Attention
{Input: Query, Key(Encoder Output), Value(Encoder Output)}`

`- Dropout2`

`- LayerNorm with Residual Connection2 → En/Decoder Output`

`-`

`- Position wise Feed Forward`

`- Dropout3`

`- LayerNorm with Residual Connection3 → Output`

Encoder의 Output이 Key, Value가 된다.

Masked Multi Head Attention에는 Look Ahead Mask을 넣고,

Encoder-Decoder Multi Head Attention에는 또 Padding Mask를 넣는다.

## Multi Head Self Attention &  Scaled Dot Product Attention

Layer의 Multi Head Self Attention과 Scaled Dot Product Attention을 상세히 서술한다.

### Vector Calculation.

문장 내 특정 단어에 대한 Self-attention을 계산하려면, 문장의 다른 단어들과 각각 점수를 계산해야 한다.

실제 구현에서는 빠른 속도를 위해 Vector가 아닌 Matrix를 사용한다.

1. Create Head Vector

    ![image](https://user-images.githubusercontent.com/66259854/104466771-f07e6e00-55f8-11eb-9e0c-871e9fdeb7aa.png)

    3가지 Head Vector는 Input Vector와 3개의 학습 가능한 행렬을 각각 곱해서 만들어진다.

    Input Vector의 크기가 $d_{model}=512$일 때, Head Vector의 크기는 64인데, Attention의 계산 복잡도를 일정하게 만들고자 하는 구조 때문이다.

    (또한 64라는 값은 $num-heads=8$로 결정되는데, $d_{model}$을 $num-heads$로 나눈 값이다.)

2. Matmul

    ![image](https://user-images.githubusercontent.com/66259854/104466792-f411f500-55f8-11eb-98a7-c705415e851c.png)

    현재 단어의 Q Vector와 모든 단어의 K Vertor를 Matmul 한다.

3. Scale & Softmax (Mask 과정을 거칠 수 있다.)

    ![image](https://user-images.githubusercontent.com/66259854/104466808-f83e1280-55f8-11eb-81b8-96cc4ab4b8b5.png)

    1. 점수들을 8로 나누는데, Key Vector의 크기 64의 제곱근이다.

        이를 Attention Score라고 부른다.

    2. Softmax를 취하고, 이 점수는 현재 위치의 단어에서 각 단어들의 표현이 얼마나 포함되는지 결정한다.

        당연히 현재 위치의 단어가 가장 높은 점수를 보이나, 다른 단어에 대한 정보도 포함되어 있다.

4. Matmul

    ![image](https://user-images.githubusercontent.com/66259854/104466823-fbd19980-55f8-11eb-8eff-73760822a595.png)

    점수에 V Vector를 곱한다.

    이를 Attention Value 혹은 Context Vector라고 부른다.

    Attention을 위해 관련이 있는 단어는 남겨두고, 관련이 없는 단어는 0.001과 같은 아주 작은 숫자를 곱해 없앤다.

5. Concat

    점수와 V Vector가 곱해진 Attention Value를 Concat 한다.

    현재 위치에 대한 Self Attention의 출력이다.

### Matrix Calcualtion.
1. Query, Key, Value Matrix

    ![image](https://user-images.githubusercontent.com/66259854/104466832-ff652080-55f8-11eb-9c85-7ed4d95ed7a1.png)

    Input Vector or Embedding Vector를 하나의 행렬 X로 쌓아 올리고, 학습할 Weight 행렬을 곱해 Q, K, V를 계산한다.

    행렬 X의 각 행은 입력 문장의 각 단어에 해당한다.

2. One Equation

    ![image](https://user-images.githubusercontent.com/66259854/104466853-0429d480-55f9-11eb-83c7-73efc5047835.png)

    행렬을 사용하면 Vector Calculation 2~5를 하나의 식으로 압축할 수 있다.

### Multi Head.

![image](https://user-images.githubusercontent.com/66259854/104466868-07bd5b80-55f9-11eb-9245-14ad6a9c555c.png)

H개의 Query, Key, Value Weight 행렬을 갖고 있다.

논문에서는 $num-heads=8$개의 Attention Head를 갖는다.

각 Attention Head에서 Query, Key, Value는 랜덤으로 초기화되어 학습된다.

![image](https://user-images.githubusercontent.com/66259854/104466881-0b50e280-55f9-11eb-93f3-48fd582581e0.png)

Self Attention 과정을 거치면 8개(논문 기준)의 Z 행렬이 나온다.

그러나 Feed Foward는 한 위치에 대해 한 개의 행렬만 받을 수 있으므로 문제가 발생한다.

![image](https://user-images.githubusercontent.com/66259854/104466916-1277f080-55f9-11eb-91f6-2f8ec0b1ab45.png)

문제를 해결하기 위해 8개의 Z 행렬을 이어 붙여서 하나의 행렬을 만들고, 또 다른 Weight 행렬인 $W_O$을 곱한다.

![image](https://user-images.githubusercontent.com/66259854/104466933-173ca480-55f9-11eb-9b9b-c811931eefb9.png)

모두 요약하면 다음과 같은 그림이 된다.

Multi Head는 Self Attention을 병렬적으로 사용한다는 의미이다.

### Difference of Attention.

![image](https://user-images.githubusercontent.com/66259854/104466947-1b68c200-55f9-11eb-9659-cf072d28d071.png)

1. Encoder Self Attention은 Encoder에서 이루어진다.

2. Masked Decoder Self Attention은 Decoder에서 이루어진다.

3. Encoder-Decoder Attention도 Decoder에서 이루어지나, Self-attention이 아니다.

    Query: Decoder Vector / Key, Value: Encoder Vector로, Head의 출처가 같지 않으므로 동일하지 않기 때문이다.

## Mask

### Padding Mask.

```python
def attention(query, key, value, mask=None, dropout=0.0):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)

    # (Dropout described below)
    p_attn = F.dropout(p_attn, p=dropout)
    return torch.matmul(p_attn, value), p_attn
    ```

    ```python
    def scaled_dot_product_attention(query, key, value, mask):
    '''중략'''
        logits += (mask * -1e9) # 어텐션 스코어 행렬인 logits에 mask*-1e9 값을 더해주고 있다.
    '''중략'''
```

입력 문장에 <PAD> Token이 있을 때, 유사도를 구하지 않도록  Masking을 하여 Attention에서 제외한다.

1. -1e9와 같은 작은 음수값을 곱한다.
2. Softmax를 지나기 이전에 작은 음수값이 있으므로, Softmax를 지나면 0에 가까워진다.

### Look-ahead Mask.

![image](https://user-images.githubusercontent.com/66259854/104466965-202d7600-55f9-11eb-9ab9-195fbc75c5a5.png)

seq2seq Decoder와 달리, Transformer Decoder는 문장 행렬로 입력을 한 번에 받는다.

따라서 현재 단어를 예측할 때, 미래 시점의 단어도 참고하는 일이 발생할 수 있다.

이를 방지하기 위해 Decoder의 첫 번째 서브층에서 Look-ahead Mask를 추가한다.

---

![image](https://user-images.githubusercontent.com/66259854/104466981-24f22a00-55f9-11eb-97e4-4bef39ae6da9.png)

```python
    def subsequent_mask(size):
        "Mask out subsequent positions."
        attn_shape = (1, size, size)
        subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
        return torch.from_numpy(subsequent_mask) == 0
```

```python
    @staticmethod
        def make_std_mask(tgt, pad):
            "Create a mask to hide padding and future words."
            tgt_mask = (tgt != pad).unsqueeze(-2)
            tgt_mask = tgt_mask & Variable(
                subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
            return tgt_mask
```

1. Self-attention을 통해 Attention Score Matrix를 얻는다.
2. 다음 그림과 같이 Masking하여 미래 시점의 단어를 참고하지 못하도록 바꾼다.

## 🎸

### Auto Regressive & Teacher Forcing.

[Transformer model for language understanding | TensorFlow Core](https://www.tensorflow.org/tutorials/text/transformer#training_and_checkpointing)

[고작 인간 : 네이버 블로그](https://blog.naver.com/just_nlp/222136930059)

```python
class Batch:
    "Object for holding a batch of data with mask during training."
    def __init__(self, src, trg=None, pad=1):
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)
        if trg is not None:
            self.trg = trg[:, :-1]
            self.trg_y = trg[:, 1:]
            self.trg_mask = \
                self.make_std_mask(self.trg, pad)
            self.ntokens = (self.trg_ != pad).data.sum()
```

1. Transformer의 Decoder는 Generate 할 때 무조건 1개의 Token을 생성하도록 학습되어 있다. Sequence를 Generate 하려면 Sequence 길이만큼 Decoder를 반복해서 실행한다.

2. Inference 상황에서 학습을 생각해 보자. 학습 단계에서도 1개씩 생성하도록 학습해야 한다. Source - Target 두 문장은 Pair로 잘 가지고 있다. Decoder를 학습할 때는 Target에서 첫 번째 글자를 안다고 생각하고 두 번째 글자를 학습한다. 세 번째 글자는 앞의 두 글자를 사용하여 학습한다. 이렇게 한 글자씩 예측하고 Loss를 측정하는 방식을 Auto Regressive라고 한다.

3. "두 번째 토큰이 잘못 생성되었으면 세 번째 토큰을 만들고 Loss를 측정할 때 문제가 생기는가?"라는 의문이 들 수 있다. 학습 단계에서 잘못 생성되어도 그 값을 그대로 사용하지 않고, 원본 문장을 다시 넣는다. 생성한 토큰을 직접 넣는 Inference와 차이를 보인다. 이렇게 잘못 생성되어도 다음 토큰을 만들 때 올바른 데이터를 넣는 방식을 Teacher Forcing이라고 한다.

4. 따라서, Decoder는 End Token 직전까지 생성 모델을 태워서 End Token이 잘 만들어지는지 확인하면 한 Sequence를 학습한 것이다. RNN에서도 하나를 만들고, 그 다음 토큰을 만들기 위해 이전의 Vector를 전부 이용하는 것을 생각하면 된다.

Original Comments.

    transformer 의 decoder 는 generate 할때 무조건 '1개의 token' 을 생성하게 학습되어있습니다. 그럼 sequence 를 generate 할려면 sequence 길이 만큼 decoder 를 반복해서 실행해야합니다.
    위 상황은 inference 상황이라고 해보면, 그럼 학습은 어떻게 이뤄질까 생각해보시면 됩니다. 학습할때도 1개씩 생성하게 학습해야하는데, source - target 두 문장의 pair 는 쌍으로 잘 들고 있고, 그럼 decoder 를 학습할때는 target에서 맨 앞 글자만 안다 치고 그 다음 글자를 학습하고, 그 다음에는 3번째 글자를 만들기 위해 앞에 두 글자만 사용하겠죠?
    이렇게 한 글자씩 예측하고 loss 를 측정하는 방식을 auto regressive 방식 이라고 합니다.

    그럼 다음 의문이 들 수 있는데,
    '만약 첫 번째 토큰 기반으로 두 번째 토큰을 생성했는데, 이게 잘못 만들어졌으면 세 번째 토큰을 만들고 로스를 측정할때 잘못 만들어진 두 번째 토큰을 사용하면 문제가 되는게 아닐까?' 라는 의문이 생길 수 있는데, 학습 단계에서는 잘못 생성했어도 그 값을 그대로 사용하지 않고, 원본 문장을 다시 넣습니다. 인퍼런스 할때와 다른 점이죠(인퍼런스는 생성한 토큰을 직접 넣음.)
    이렇게 잘못 만들었어도 바로 다음 토큰을 생성할때 데이터를 올바른 데이터를 넣어주는 방식을 teacher forcing 이라고 합니다. (근데 위 링크 번역이 교사 강제... 라고 되어있네요.. 여러분 영어로 같이 보세요...)

    따라서, decoder 는 문장의 마지막을 알리는 토큰 직전까지 생성 모델을 태워서 마지막 토큰이 잘 만들어지는지 확인하면 한 시퀀스를 전부 학습한게 됩니다. auto regressive 하게 말이죠.

    RNN 에서도 하나를 만들고, 그 다음 토큰을 만들기 위해 직전 vector 들을 전부 이용하는것을 생각하시면 됩니다 :)

## 링크

[BERT](https://www.notion.so/BERT-c61ed6193e85436c8916c641c9372188)

[Google Colaboratory](https://colab.research.google.com/drive/1cwhr9FD4Ogmr8zNieCRICGD5V9kuT_jv#scrollTo=11l-Oq-Mjfg7)

[soline013/transformer-tensorflow2.0](https://github.com/soline013/transformer-tensorflow2.0/blob/master/transformer_implement_tf2_0.ipynb)

[The Annotated Transformer](https://nlp.seas.harvard.edu/2018/04/03/attention.html)

[위키독스](https://wikidocs.net/31379)

[Transformer model for language understanding | TensorFlow Core](https://www.tensorflow.org/tutorials/text/transformer)

[The Illustrated Transformer](https://nlpinkorean.github.io/illustrated-transformer/)

[TorchText로 언어 번역하기 - PyTorch Tutorials 1.6.0 documentation](https://tutorials.pytorch.kr/beginner/torchtext_translation_tutorial.html)

[11주차(2) - Attention is All You Need (Transformer)](https://www.quantumdl.com/entry/11%EC%A3%BC%EC%B0%A82-Attention-is-All-You-Need-Transformer)
