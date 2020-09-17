# 모두를 위한 딥러닝 LEC06.
## *Multinomial Classification.*
*Hyper Plane으로 각각의 경우를 구분하고, Matrix Multiplication을 이용한다.* //n차원-1의 선형함수.

![image](https://user-images.githubusercontent.com/66259854/93458114-dff40a80-f91a-11ea-97a5-42414373a016.png)

1. X → 「C」 → Y
2. X → 「B」 → Y
3. X → 「A」 → Y

++++++++++

![image](https://user-images.githubusercontent.com/66259854/93458141-e7b3af00-f91a-11ea-8f3f-a14fa7d71bcf.png)

∥∥∥∥∥∥∥∥∥∥

![image](https://user-images.githubusercontent.com/66259854/93458150-e97d7280-f91a-11ea-9f02-a5672f6cbb8d.png)

## *Softmax Function.*
*Softmax Function을 이용하여 값을 확률과 같이 바꾸고, One Hot Encoding을 이용한다.*

1. ![image](https://user-images.githubusercontent.com/66259854/93462306-192f7900-f921-11ea-84cc-1e82f15edd83.png)
2. ![image](https://user-images.githubusercontent.com/66259854/93462315-1b91d300-f921-11ea-9c56-4ee48cac5abf.png)
3. ![image](https://user-images.githubusercontent.com/66259854/93462949-079aa100-f922-11ea-9298-de5e1ca701ef.png)
   1) 0~1의 값
   2) ∑ = 1
   3) 확률(p)
4. ![image](https://user-images.githubusercontent.com/66259854/93462976-0ff2dc00-f922-11ea-9db4-8800367eaccd.png)
   1) **One Hot Encoding**
   
      제일 큰 값(확률)을 1로 하고 나머지는 0으로 한다. //들여쓰기 

## *Cross-Entropy* Cost Function.
![image](https://user-images.githubusercontent.com/66259854/93463789-37967400-f923-11ea-9904-da25b9fdd0a8.png)

## Logistic Cost & Cross Entropy.
![image](https://user-images.githubusercontent.com/66259854/93463801-3cf3be80-f923-11ea-9421-13b5a77a40f4.png)

## Cost Function.
![image](https://user-images.githubusercontent.com/66259854/93463812-42510900-f923-11ea-9148-ca4f0fad5258.png)

## Gradient Descent Algorithm.
-α∆L(w_1,w_2) //자세한 미분은 다루지 않음

![image](https://user-images.githubusercontent.com/66259854/93463837-4c730780-f923-11ea-8644-ba14a3694e31.png)



# 모두를 위한 딥러닝 LAB06.md
## Softmax_cross_entropy_with_logits.
![image](https://user-images.githubusercontent.com/66259854/93467284-81358d80-f928-11ea-904a-8827f3bfac3d.png)

1. LAB06 초반 코드에서의 cost.
2. .softmax_cross_entropy_with_logits()을 사용한 cost.
