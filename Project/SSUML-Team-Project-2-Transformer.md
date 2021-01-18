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

## Timeline.
- 01/03~01/17 | 프로젝트 기한.
- 01/11 | Comments 마무리.
- 01/12~01/13 | Train 부분 구현.
- 01/13 | Transformer 정리 마무리.
- 01/14~01/16 | Train 시키며 실험 진행.

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

### 3.
```python
train, val, test = datasets.IWSLT.splits(
        exts=('.de', '.en'), fields=(SRC, TGT), 
        filter_pred=lambda x: len(vars(x)['src']) <= MAX_LEN and 
            len(vars(x)['trg']) <= MAX_LEN)
```

```python
---------------------------------------------------------------------------
OSError                                   Traceback (most recent call last)
/usr/lib/python3.6/tarfile.py in gzopen(cls, name, mode, fileobj, compresslevel, **kwargs)
   1644         try:
-> 1645             t = cls.taropen(name, mode, fileobj, **kwargs)
   1646         except OSError:

12 frames
OSError: Not a gzipped file (b'<!')

During handling of the above exception, another exception occurred:

ReadError                                 Traceback (most recent call last)
/usr/lib/python3.6/tarfile.py in gzopen(cls, name, mode, fileobj, compresslevel, **kwargs)
   1647             fileobj.close()
   1648             if mode == 'r':
-> 1649                 raise ReadError("not a gzip file")
   1650             raise
   1651         except:

ReadError: not a gzip file
```

[Error while trying to run colab notebook from readme.md · Issue #38 · fastaudio/fastai2_audio](https://github.com/fastaudio/fastai2_audio/issues/38)

[Read error at Google Colab notebook error · Issue #13 · fastaudio/fastaudio](https://github.com/fastaudio/fastaudio/issues/13)

[tweaks code to support new url for IWSLT dataset by ghlai9665 · Pull Request #1115 · pytorch/text](https://github.com/pytorch/text/pull/1115)

A Real World Example부터는 Projet Notebook에서 Multi30K로 진행하였다.

당장 해결하기 어려운 듯 보인다.

### 4.
```markdown
제공받은 코드에서 오류를 발견해 공유합니다!

Multi30k dataset의 shape가 일반적인 경우와 다르게 (tok_idx, batch_idx)의 형태여서 하나의 sequence가 row가 아닌 column 형태인 것 같습니다. 그런데 Batch class 코드에서는 일반적인 경우에 맞춰 slicing을 수행하고 있습니다.
self.trg = trg[:, :-1]
self.trg_y=trg[:, 1:]
로 되어 있는데 각각 마지막 token, 첫번째 token을 제외하는 것이 아닌 batch의 trg에서 마지막 sequence, 첫번째 sequence를 제거하는 방식으로 올바르지 않게 동작합니다.
itos는 정상적으로 동작하는데, 시작과 동시에 인자로 받은 batch를 transpose한 뒤 사용하기 때문인 것으로 보입니다.

Batch class의 __init__의 코드를
self.trg = trg[:, :-1]
self.trg_y = trg[:, :-1]
에서
self.trg = trg.T[:, :-1]
self.trg_y = trg.T[:, :-1]
로 변경하고,

itos의 코드를
batch = batch.T.tolist()
에서
batch = batch.tolist()
로 변경하면 문제가 해결됩니다.
```
### 5.
```python
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
<ipython-input-59-5d1743421a66> in <module>()
      3 src = torch.LongTensor([[SRC.stoi[w] for w in sent]]).to(dev)
      4 src = Variable(src)
----> 5 src_mask = (src != SRC.vocab.stoi["<blank>"]).unsqueeze(-2)
      6 out = greedy_decode(model, src, src_mask, 
      7                     max_len=60, start_symbol=TRG.vocab.stoi["<s>"])

AttributeError: 'Vocab' object has no attribute 'vocab'
```

한 번 돌렸던 결과를 다시 돌리면 발생하는 오류.

과거 모두를 위한 딥러닝에서도 다시 돌리면 문제가 발생한 적이 있다.

vocab을 없애면 6번 오류로 이어진다.

### 6.
```python
model.eval()
sent = "▁The ▁log ▁file ▁can ▁be ▁sent ▁secret ly ▁with ▁email ▁or ▁FTP ▁to ▁a ▁specified ▁receiver".split()
src = torch.LongTensor([[SRC.stoi[w] for w in sent]]).to(dev)
src = Variable(src)
src_mask = (src != SRC.stoi["<blank>"]).unsqueeze(-2)
out = greedy_decode(model, src, src_mask, 
                    max_len=60, start_symbol=TRG.stoi["<s>"])
print("Translation:", end="\t")
trans = "<s> "
for i in range(1, out.size(1)):
    sym = TRG.itos[out[0, i]]
    if sym == "</s>": break
    trans += sym + " "
print(trans)
```

```python
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
<ipython-input-58-396f57155dbe> in <module>()
      5 src_mask = (src != SRC.stoi["<blank>"]).unsqueeze(-2)
      6 out = greedy_decode(model, src, src_mask, 
----> 7                     max_len=60, start_symbol=TRG.stoi["<s>"])
      8 print("Translation:", end="\t")
      9 trans = "<s> "

12 frames
/usr/local/lib/python3.6/dist-packages/torch/nn/functional.py in dropout(input, p, training, inplace)
    976             return handle_torch_function(
    977                 dropout, (input,), input, p=p, training=training, inplace=inplace)
--> 978     if p < 0. or p > 1.:
    979         raise ValueError("dropout probability has to be between 0 and 1, "
    980                          "but got {}".format(p))

TypeError: '<' not supported between instances of 'Dropout' and 'float'
```

한 번 돌렸던 결과를 다시 돌리면 발생하는 오류.

과거 모두를 위한 딥러닝에서도 다시 돌리면 문제가 발생한 적이 있다.

5번 오류에서 vocab을 없애도 이 오류가 발생하여 다시 돌려야 했다.

* * *

1. 학습 속도가 매우 느리다.

    처음에는 진행조차 어려웠고, GPU를 올려도 CPU로 구동되어 Epoch를 줄일 수밖에 없었다.

    일단 NVIDIA Drive Version의 문제인 듯 보인다.

    `!ln -sf /opt/bin/nvidia-smi /usr/bin/nvidia-smi`

2. <pad> Token.

    <eos>로 끝난 이후에도 <pad> Token이 존재한다.

    추후 예상되는 부분을 건드려 볼 예정.

## Result.
### 1.

### 2.

### 3.
번역 결과는 조금 부실하였지만, Attention Visualization이 잘 나타났다.

특히 이번 Attention Visualization 결과는 Harvard NLP와 비슷한 양상을 보인다.

### 4.
마찬가지로 번역 결과는 조금 부실하였지만, Attention Visualization이 잘 나타난다.

## Project Notebook.

[Google Colaboratory](https://colab.research.google.com/drive/1UDJoqzwlfZVjdQu_COpUyGT8TppVcQQA#scrollTo=M1Pwa3-pEJGX)

## 링크.

[Transformer (Attention Is All You Need) 구현하기 (1/3)](https://paul-hyun.github.io/transformer-01/)

조금 더 직접 코드를 건드리기 좋을 것 같은 Transformer 구현 게시글이 있었다. `구현 예정.`
