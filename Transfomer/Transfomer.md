# Transfomer
"Attention is all you need"(Vaswani et al. 2017)
https://arxiv.org/abs/1706.03762

NLP에서 사용되는 모델로, RNN에서 벗어나 Attention만을 사용하는 신경망을 고안하여 Self-Attention을 사용하는 모델이다.

RNN의 순차적 계산을 행렬곱을 사용하여 한 번에 처리함으로써 모든 중요 정보를 Embedding 한다.

Self-Attention을 통해 같은 문장 내 모든 단어 쌍 사이의 의미, 문법 관계를 알 수 있다.

그러나 행렬곱만을 사용하여 위치, 순서 정보를 제공할 수 없어, BERT가 등장하게 되었다.

##Layer
아래 그림은 Inputs:Encoder / Outputs:Decoder로 나눌 수 있다.

![image](https://user-images.githubusercontent.com/66259854/93798739-1e6e2a00-fc79-11ea-8a00-5d9fbf9467e4.png)

Input Embedding

Positional Encoding

(Masked) Multi-Head Attention

Dropout

Layer Normalization

Scaled Dot Product Attention

Feed Forward

## Input Embedding
Word Embedding

https://www.tensorflow.org/api_docs/python/tf/keras/layers/Embedding

Convert 2D sequence (batch_size, input_length)

→ 3D (batch_size, input_length, $d_{model}$)

![image](https://user-images.githubusercontent.com/66259854/93799274-e6b3b200-fc79-11ea-8d2b-5a962887f66c.png)

![image](https://user-images.githubusercontent.com/66259854/93799283-ea473900-fc79-11ea-908f-d70f6f3d22f4.png)
