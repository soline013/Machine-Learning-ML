# 모두를 위한 딥러닝 LEC11.
## *Convolutional Neural Networks.* //1959 by Huble & Wiesel
  1. 고양이의 두뇌 뒤에 전극을 연결하여 연구를 진행하였다.
     
     ![image](https://user-images.githubusercontent.com/66259854/96874440-13ddc500-14b1-11eb-890c-bc67be284192.png)
     
  2. 여러 레이어를 쌓아 모델을 만들었다.
  
     ![image](https://user-images.githubusercontent.com/66259854/96874451-16d8b580-14b1-11eb-9ec4-c4c6afb2bb9c.png)
     
## *ConvNet. CNN.*
  1. ![image](https://user-images.githubusercontent.com/66259854/96874463-193b0f80-14b1-11eb-9478-ebef49325364.png)
  2. ![image](https://user-images.githubusercontent.com/66259854/96874470-1b9d6980-14b1-11eb-84ed-cddbbf8c7082.png)
  3. ![image](https://user-images.githubusercontent.com/66259854/96874486-1dffc380-14b1-11eb-9a49-238c3ef8a9de.png)
  4. ![image](https://user-images.githubusercontent.com/66259854/96874491-20621d80-14b1-11eb-8a7e-d4f8dfde18df.png) //padding.
  5. ![image](https://user-images.githubusercontent.com/66259854/96874665-5bfce780-14b1-11eb-8cf4-b6970a7d74a4.png) //(32-5)/1+1=28
  6. ![image](https://user-images.githubusercontent.com/66259854/96874671-5e5f4180-14b1-11eb-879b-47523dbba25d.png) //W=5x5x3x6

## *Pooling Layer (Sampling).*
![image](https://user-images.githubusercontent.com/66259854/96874685-61f2c880-14b1-11eb-9ce3-458cfed21605.png)

Sampling 과정을 통해 크기가 줄어든다.

## *Max Pooling.*
![image](https://user-images.githubusercontent.com/66259854/96874693-64edb900-14b1-11eb-953e-6d5219b37d75.png)

Pooling 중 한 방법으로 일정 크기에서 가장 큰 값을 선택한다.

## *Fully Connected(FC) Layer.*
![image](https://user-images.githubusercontent.com/66259854/96874704-67501300-14b1-11eb-9599-025819750892.png)

## *ConvNet의 활용.*

## *AlexNet.* //Krizhevsky et al. 2012
![image](https://user-images.githubusercontent.com/66259854/96874722-6b7c3080-14b1-11eb-99d7-b4ca8ab16a12.png)
![image](https://user-images.githubusercontent.com/66259854/96874730-6d45f400-14b1-11eb-8404-54bf8d017f30.png)

  1. Input: 227x227x3 Images.
  2. First Layer(CONV): 96 11x11 Filters applied at Stride 4.
     1) Output Volume: 55x55x96.
     2) Parameters: (11x11x3) x 96 = 35K.
  3. Second Layer(FOOL): 3x3 Filters applied at Stride 2.
     1) Output Volume: 27x27x96.
     2) Parameters: 0!
  4. NORM: Normalization Layer.
  5. ![image](https://user-images.githubusercontent.com/66259854/96874746-70d97b00-14b1-11eb-86b8-04e492591608.png)

## *GoogLeNet.* //Szegedy et al. 2014
![image](https://user-images.githubusercontent.com/66259854/96874758-73d46b80-14b1-11eb-893c-02abcceb8d68.png)

## *ResNet* //He et al. 2015
  1. ![image](https://user-images.githubusercontent.com/66259854/96874770-76cf5c00-14b1-11eb-9f23-4d7e041838aa.png)
  2. ![image](https://user-images.githubusercontent.com/66259854/96874780-78991f80-14b1-11eb-8422-aab5da228049.png) //Fast Forward.

## Sentence Classification. //Yoon Kim, 2014
![image](https://user-images.githubusercontent.com/66259854/96874792-7b941000-14b1-11eb-8ded-090c513700fa.png)

## DeepMind's AlphaGo.
![image](https://user-images.githubusercontent.com/66259854/96874800-7df66a00-14b1-11eb-8b1c-a0b61640d155.png)

Policy Network:
  1. [19x19x45] Input
  2. CONV1: 192 5x5 Filters, Stride1, Pad2, [19x19x192]
  3. CONV2…12: 192 3x3 Filters, Stride1, Pad1, [19x19x192]
  4. CONV: 1 1x1 Filters, Stride1 ,Pad0, [19x19]
