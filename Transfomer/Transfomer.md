# Transfomer
"Attention is all you need"(Vaswani et al. 2017)
https://arxiv.org/abs/1706.03762

NLP에서 사용되는 모델로, RNN에서 벗어나 Attention만을 사용하는 신경망을 고안하여 Self-Attention을 사용하는 모델이다.

RNN의 순차적 계산을 행렬곱을 사용하여 한 번에 처리함으로써 모든 중요 정보를 Embedding 한다.

Self-Attention을 통해 같은 문장 내 모든 단어 쌍 사이의 의미, 문법 관계를 알 수 있다.

그러나 행렬곱만을 사용하여 위치, 순서 정보를 제공할 수 없어, BERT가 등장하게 되었다.

## Layer
아래 그림은 Inputs:Encoder / Outputs:Decoder로 나눌 수 있다.

![image](https://user-images.githubusercontent.com/66259854/93798739-1e6e2a00-fc79-11ea-8a00-5d9fbf9467e4.png)

Input Embedding

Positional Encoding

(Masked) Multi Head Attention

Dropout

Layer Normalization

Scaled Dot Product Attention

Feed Forward

## Input Embedding
Word Embedding

https://www.tensorflow.org/api_docs/python/tf/keras/layers/Embedding

Convert 2D sequence (batch_size, input_length)

→ 3D (batch_size, input_length, $d_{model}$)

## Positional Encoding
$$PE_{(pos,2_i)}=sin(pos/10000^{2_i/d_{model}})$$

$$PE_{(pos,2_{i+1})}=cos(pos/10000^{2_i/d_{model}})$$

$2_i$ = 짝수 / $2_i+1$ = 홀수

pos = Sequence Batch 수 / index = Embedding Dimention.

## Multi Head Attention
![image](https://user-images.githubusercontent.com/66259854/93799274-e6b3b200-fc79-11ea-8d2b-5a962887f66c.png)

Multi Head Attention은 Query, Key, Value head로 나뉘고, 각각 다른 Linear Projection, Scaled Dot-Product를 진행한다. (Split을 진행하기 때문에 각각 다른 Scaled Dot-Product를 진행한다.)

이후 Concat, Linear Projection을 하는데, 이는 tf.reshape, tf.transform으로 한 번에 연산할 수 있다.

## Scaled Dot Product Attention
![image](https://user-images.githubusercontent.com/66259854/93799283-ea473900-fc79-11ea-908f-d70f6f3d22f4.png)

Learnable Parameter가 없다.

기존 Additive Attention은 Attention score를 구하는 구간에 Feed Forward Layer가 있지만, Dot-Product 연산으로 대체하였다.

Encoder에서 Padding을 사용하지 않도록 Padding Mask를 추가해야 한다.

## Feed Forward
$$FFN(x)=max(0, xW_1+b_1)W_2+b_2$$

Multi Head Attention에서 나온 Attention 정보를 정리하는 역할.

`FF - Relu - Dropout - FF` 순서의 Sequential한 구조이다.

## Encoder
`Multi Head Attention(Mask)`

`- Dropout1`

`- LayerNorm with Residual Connection1 → Output`

`- Position wise Feed Forward`

`- Dropout2`

`- LayerNorm with Residual Connection2 → Output`

tf.add()로 Residual Connection을 한다.

Padding을 적용하지 못하게 Padding Mask를 넣는다.

## Decoder
`Masked Multi Head Attention`

`- Dropout1`

`- LayerNorm with Residual Connection1 → Query`

`- Encoder Multi Head Attention
{Input: Query, **Encoder** Output, **Encoder** Output}`

`- Dropout2`

`- LayerNorm with Residual Connection2 → **En**/Decoder Output`

`- Position wise Feed Forward`

`- Dropout3`

`- LayerNorm with Residual Connection3 → Output`

Encoder의 Output이 Key, Value가 된다.

Masked Multi Head Attention에는 Look Ahead Padding을 넣고,

Encoder Multi Head Attention에는 또 Padding Mask를 넣는다. (Encoder 부분을 끌어오니, Padding을 사용하지 않아야 하기 때문으로 보인다.)

## 링크
https://nlp.seas.harvard.edu/2018/04/03/attention.html

Pytorch를 사용한다.
