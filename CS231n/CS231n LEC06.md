# CS231n LEC06.
## Stanford University CS231n, Spring 2017.
Training Neural Networks.

## Recall from last time.
![image](https://user-images.githubusercontent.com/66259854/101941803-d8978100-3c2b-11eb-96f4-522edaeea7bb.png)
![image](https://user-images.githubusercontent.com/66259854/101941902-febd2100-3c2b-11eb-9e28-100d96e79375.png)
![image](https://user-images.githubusercontent.com/66259854/101941922-07adf280-3c2c-11eb-92ec-b6e8a6a8a308.png)
![image](https://user-images.githubusercontent.com/66259854/101941938-0f6d9700-3c2c-11eb-819a-8cfb19329cc4.png)
![image](https://user-images.githubusercontent.com/66259854/101941963-1694a500-3c2c-11eb-8286-ce83187ca0f0.png)
![image](https://user-images.githubusercontent.com/66259854/101941980-1eece000-3c2c-11eb-8b57-9d950388823f.png)
![image](https://user-images.githubusercontent.com/66259854/101941998-257b5780-3c2c-11eb-8d5c-a748fc4d480a.png)

## Overview.
  1. One Time Setup.
  
     Activation Functions, Preprocessing, Weight Initialization, (Regularization, Gradient Checking.)
     
  2. Training Dynamics.
  
     Babysitting the Learning Process, Hyperparameter Optimization, (Parameter Updates.)

  3. Evaluation.
  
     (Model Ensembles.)

# One Time Setup.
## Activation Functions.
![image](https://user-images.githubusercontent.com/66259854/101942876-940ce500-3c2d-11eb-8ce7-dec71272c606.png)

![image](https://user-images.githubusercontent.com/66259854/101942892-996a2f80-3c2d-11eb-9b68-6dd79b9d206d.png)

입력된 데이터와 가중치를 곱하고 활성함수, 비선형 연산을 거친다.

다양한 Activation Function이 있다.

* * *

### Sigmoid.
![image](https://user-images.githubusercontent.com/66259854/101943184-041b6b00-3c2e-11eb-91d7-da036de77c7a.png)

입력 값을 0 ~ 1 사이의 값으로 만든다.

입력이 크면 1에 가깝고, 입력이 작으면 0에 가깝다.

0 근처의 Rigime은 선형과 비슷하다.

Sigmoid에는 세 가지 문제점이 있다.

  1. Saturation neurons kill the gradients.

     ![image](https://user-images.githubusercontent.com/66259854/101943197-0978b580-3c2e-11eb-89b0-492cf7a45d6a.png)
     
     Back Prop에서 Gradient는 $\frac{dL}{dx} = \frac{dL}{d \sigma} \frac{d \sigma}{dx}$이다.
     
     1. x의 값이 -10이라면, Gradient는 0이 계속 전달된다.
     2. X의 값이 0이라면, Gradient는 의미 있는 값이 계속 전달된다.
     3. X의 값이 10이라면, Gradient는 1이 계속 전달된다.
     
     즉, Sigmoid 함수의 수평 부분에서 Gradient는 의미 없는 값이다.
     
  2. Sigmoid outputs are not zero-centered.
  
     ![image](https://user-images.githubusercontent.com/66259854/101943211-0ed60000-3c2e-11eb-8e3a-29db09ba8195.png)

     x가 항상 양수일 때, x는 어떤 가중치랑 곱해지고 활성함수를 통과한다.
     
     W에 대한 Gradient 값은 전부 양수거나 음수이다.
     
     $\frac{dL}{df} \times \frac{df}{dw} = \frac{dL}{df} \times x$
     
     따라서 Parameter를 업데이트 하여도 W는 다 같이 증가하거나 감소하여 지그재그 모양이 된다.
     
     Gradient 업데이트를 여러 번 수행해야 한다.

  3. exp() is a bit compute expensive.
     
     그렇게 큰 문제는 아니다.
     
     오히려 내적의 계산 비용이 더 크다.

* * *

### tanh.
![image](https://user-images.githubusercontent.com/66259854/101944151-9a9c5c00-3c2f-11eb-9bc6-730d34d66d22.png)

tanh는 입력 값을 -1 ~ 1 사이로 만든다.

  1. 장점.
     
     Zero-centered로, "Sigmoid outputs are not zero-centered." 문제가 해결된다.
  
  2. 단점.
     
     그러나 "Saturation neurons kill the gradients."는 여전하다.
     
* * *

### ReLU.
![image](https://user-images.githubusercontent.com/66259854/102707779-35a4ce00-42e1-11eb-97f1-ffe2b565c432.png)

입력이 음수면 Element-wise 연산으로 값이 0이다.

  1. 장점.
     1. Sigmoid, tanh와 다르게 양의 수에서 Saturation(포화)가 없다.
     2. max 연산으로 속도가 빠르다.
  2. 단점.
     1. Zero-centered가 아니다.
     2. 음의 수에서 Saturation이 발생한다.
     
        ![image](https://user-images.githubusercontent.com/66259854/102707784-42c1bd00-42e1-11eb-87f0-9fe8d124e773.png)
     
        ReLU는 Gradient의 절반을 죽이는데, 이를 Dead ReLU라고 한다.
     
        ![image](https://user-images.githubusercontent.com/66259854/102707788-481f0780-42e1-11eb-940b-cbbbf6d83e3b.png)
     
        Data Cloud = Training Data.
     
        ReLU에서 평면의 절반만 Activate 된다.
     
        실제 네트워크의 10~20%는 Dead ReLU로 나온다.
     
        ReLU의 초기화에서 Positive Biases 값을 추가한다.   
        (그래도 대부분 Zero-bias로 초기화.)
        
        Data Cloud와 ReLU가 멀리 있는 경우에도 Dead ReLU가 발생하는데 2가지 이유가 있다.
        
        1. Wrong Initialize.
        
           가중치가 초평면을 이루는데, 초평면 자체가 멀리서 생길 수 있다.
           
        2. High High Learning Rate.
           
           가중치가 커져 데이터의 Manifold를 벗어남.

* * *

### Leaky ReLU. PReLU.
![image](https://user-images.githubusercontent.com/66259854/102707798-638a1280-42e1-11eb-9d3f-4236b3d6d02c.png)

음수 구간에 기울기를 줘서, Saturation을 방지한다.

PReLU는 기울기를 $\alpha$로 두고 Back Prop로 Parameter를 찾아간다.

* * *

### ELU.
![image](https://user-images.githubusercontent.com/66259854/102707802-697ff380-42e1-11eb-9f12-77991ae1e1b8.png)

Zero-mean에 보다 가까운 출력값이 나온다.

음수 구간에서 Saturation이 발생하지만 Noise에 강하다.

ReLU(Saturation)와 Leaky ReLU(Zero-mean)의 중간 정도..?

* * *

### Maxout.
![image](https://user-images.githubusercontent.com/66259854/102707804-6edd3e00-42e1-11eb-92b7-6bcfc7a36b15.png)

입력 형식이 정해져 있지 않다.

$w^T_1x + b_1, w^T_2x + b_2$처럼 받고 두 개 중 최댓값을 취한다.

뉴런 당 Parameter가 배가 되는 단점이 있다.

* * *

![image](https://user-images.githubusercontent.com/66259854/102707807-7270c500-42e1-11eb-9879-af35c5fbcd01.png)

## Data Preprocessing.
