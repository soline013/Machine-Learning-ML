# 모두를 위한 딥러닝 LEC12.
## *Sequence Data.*
![image](https://user-images.githubusercontent.com/66259854/99654178-50471580-2a9d-11eb-9cf7-49edd29cb141.png)

  1. We don't understand one word only.
  2. We understand based on the previous words + this word. (time series)
  3. NN/CNN can not do this.
  4. Notice: the same function and the same set of parameters are used at every time step.

## *Recurrent Neural Network.* //순환신경망
![image](https://user-images.githubusercontent.com/66259854/99654188-53420600-2a9d-11eb-8501-ffff37b43ed7.png)
![image](https://user-images.githubusercontent.com/66259854/99654195-550bc980-2a9d-11eb-8afa-f099b6e0ab94.png)

  1. Vanilla //The state consists of a single "hidden" vector h
     1. $h_t = f_w(h_{t-1}, x_t)$
     2. $h_t = tanh(W_{hh}h_{t-1}, w_{xh}x_t)$
     3. $y_t = W_{hy}h_t$

## *Character-level Language Model* Example.
![image](https://user-images.githubusercontent.com/66259854/99654207-576e2380-2a9d-11eb-8ace-993e2f780c29.png)

![image](https://user-images.githubusercontent.com/66259854/99654232-5e953180-2a9d-11eb-9f2b-efd02ecdea31.png) //"h"의 output layer에서 Error, output layer는 Softmax 사용.

Vocabulary: [h, e, l, o]

Example Training Sequence: "hello"

## RNN Applications.
![image](https://user-images.githubusercontent.com/66259854/99654247-62c14f00-2a9d-11eb-9fe1-0061d0ef89b6.png)

  - one to one → Vanilla
  - one to many → Image Captioning
  - many to one → Sentiment Classification
  - many to many 1 → Machine Translation
  - many to many 2 → Video Classification
  
  1. Language Modeling
  2. Speech Recognition
  3. Machine Translation
  4. Conversation Modeling/Question Answering
  5. Image/Video Captioning
  6. Image/Video/Dance Generation

## *Multi-layer RNN*
![image](https://user-images.githubusercontent.com/66259854/99654264-67860300-2a9d-11eb-8942-8c8b275b39e4.png)

## Training RNNs is Challenging.
  1. Several Advanced Models
     1. Long Short Term Memory (LSTM)
     2. GRU by Cho et al. 2014

# 모두를 위한 딥러닝 LAB12.md
## Different Sequence Length.
Sequence_length=[5, 2, 3]

![image](https://user-images.githubusercontent.com/66259854/99654340-8389a480-2a9d-11eb-94de-9f619287784e.png)

## *Dynamic RNN.*
![image](https://user-images.githubusercontent.com/66259854/99654398-9603de00-2a9d-11eb-881c-5a546a035dfe.png)
