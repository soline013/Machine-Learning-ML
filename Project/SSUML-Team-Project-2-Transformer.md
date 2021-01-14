## Project Guidance.

[프로젝트 2 - 번역해보기 - transformer](https://www.notion.so/2-transformer-b8e778867ef44c46ad905ed32bdf3449)

## Harvard NLP.

[The Annotated Transformer](https://nlp.seas.harvard.edu/2018/04/03/attention.html#training-loop)

---

[Google Colaboratory - Github](https://colab.research.google.com/drive/1FHXP-a0rjsgHV4kbwlF2mxijkTz3Rnfe)

Harvard NLP와 Google Colab.

Github에 있는 .ipynb 파일은 Harvard NLP와 똑같다.

---

[Google Colaboratory - Google Drive](https://colab.research.google.com/drive/1_5Ho9ldrTZFay2Z8Qfdb3QXUkx6zNRVT#scrollTo=wamR3SPdUgRo)

Harvard NLP와 차이점이 있는 Google Colab.

Harvard NLP에 있는 Colab 링크로 가면 Google Drive와 연결된다.

## What Is the Transformer?

[Transfomer](https://www.notion.so/asollie/Transfomer-d692df0aa2c64b9c9c24f6a1dcbfb0e3)

Transformer에 대한 정리는 링크 참조.

## Error.
### 1.
"TorchText, Multi30k, pytorch로 transformer 학습하기"
`torch==1.7.1`, `torchtext==0.8.1`

"The Annotated _Attention is All You Need_Different_CP"
"The Annotated Transformer_CP"
`torch==0.3.0.post4`, `torchtext==0.2.3`

```python
# For data loading.
from torchtext import data, datasets
'''
AttributeError: module 'torch' has no attribute 'float32'
'''
```

Harvard NLP Colab을 돌리는 중, torchtext에 대한 Version 지정은 없었지만 오류가 발생했다.

Harvard NLP Comments에 있는 `pip install torchtext=0.2.3`으로 해결할 수 있었다.

### 2.
"TorchText, Multi30k, pytorch로 transformer 학습하기"
`torch==1.7.1`, `torchtext==0.8.1`

```python
return crit(Variable(predict.log()),
                 Variable(torch.LongTensor([1]))).data #.data[0]

return loss.data * norm #return loss.data[0] * norm
'''
Error Content: IndexError: invalid index of a 0-dim tensor. Use tensor.item() to convert a 0-dim tensor to a Python number
'''
```

Pytorch 1.7.1에서 실행하면 오류가 발생한다.

0.5 이전의 Pytorch와 0.5 이후의 Pytorch의 데이터 자료 구조가 바뀌었기 때문이다.

Var.data[0]을 Var.data로 변경하면 해결된다.

## Project Notebook.

[Google Colaboratory](https://colab.research.google.com/drive/1UDJoqzwlfZVjdQu_COpUyGT8TppVcQQA#scrollTo=M1Pwa3-pEJGX)

## 링크.

[Transformer (Attention Is All You Need) 구현하기 (1/3)](https://paul-hyun.github.io/transformer-01/)

조금 더 직접 코드를 건드리기 좋을 것 같은 Transformer 구현 게시글이 있었다. `구현 예정.`
