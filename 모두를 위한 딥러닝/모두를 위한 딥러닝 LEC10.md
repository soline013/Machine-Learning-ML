# 모두를 위한 딥러닝 LEC10.
## *Vanishing gradient.* //NN winter2: 1986-2006.
Sigmoid를 사용하면 많은 레이어의 계산에서 기울기가 점점 사라진다.

![image](https://user-images.githubusercontent.com/66259854/95179700-f4ca0c80-07fb-11eb-8aee-427e5ea0ab3c.png)

## *ReLU: Rectified Linear Unit.*
Code: L1 = tf.nn.relu(tf.matmul(X, W1) + b1)

ReLU 함수를 사용하여 Vanishing gradient를 해결할 수 있다.

![image](https://user-images.githubusercontent.com/66259854/95179708-f7c4fd00-07fb-11eb-8cd4-36a6182f39e7.png)

## Cost Function.
![image](https://user-images.githubusercontent.com/66259854/95179712-fa275700-07fb-11eb-8b74-291cdbb7ab2f.png)

## Other *Activation Functions.*
![image](https://user-images.githubusercontent.com/66259854/95179720-fbf11a80-07fb-11eb-904d-489f6972aabc.png)

## Need to Set the Initial Weight Values Wisely.
  1. Not all 0's.
  2. Challenging issue.
  3. Hinton et al. (2006) "A Fast Learning Algorithm for Deep Belief Nets"- RBM.

## *RBM(Restricted Boatman Machine)* Structure.
![image](https://user-images.githubusercontent.com/66259854/95180191-9ea99900-07fc-11eb-9003-8392f7459f34.png)
![image](https://user-images.githubusercontent.com/66259854/95180198-a0735c80-07fc-11eb-9b96-abcf84c9166b.png)

  1. Forward를 통한 x값과 Backward를 통한 x를 비교한다.
  2. 가장 차이가 적도록 Weight를 조정한다.

## *Xavier/He Initialization.*
  1. Makes sure the weights are ‘just right’, not too small, not to big
  2. Using number of input (fan in) and output (fan out)
  3. Code: W = np.random.randn(fan_in, fan_out)/np.sqrt(fan_in) //by 2015. (fan_in/2)

## *Dropout.* //A Simple Way to Prevent NN from Overfitting (Srivastava et al. 2014)
"randomly set some neurons to zero in the forward pass."

Code: L1 = tf.nn.dorpout(_L1, dorpout_rate)

![image](https://user-images.githubusercontent.com/66259854/95180209-a406e380-07fc-11eb-8271-76667eb1fcf7.png)

## *Ensemble.*
![image](https://user-images.githubusercontent.com/66259854/95180218-a6693d80-07fc-11eb-9875-bfb3cdc2a82b.png)

## NN LEGO Play.
### *Fast Forward.*
*신호를 앞으로 당겨 계산한다.*

![image](https://user-images.githubusercontent.com/66259854/95180305-c26cdf00-07fc-11eb-99df-d479b69a6958.png)

* * *

### *Split & Merge.*
*나뉘어 계산을 하고 나중에 모인다.*

![image](https://user-images.githubusercontent.com/66259854/95180314-c567cf80-07fc-11eb-8397-b035b3035cb0.png)

* * *

### *Recurrent Network(RNN).*
![image](https://user-images.githubusercontent.com/66259854/95180328-cb5db080-07fc-11eb-8ed3-f5ed84c7b2cc.png)
