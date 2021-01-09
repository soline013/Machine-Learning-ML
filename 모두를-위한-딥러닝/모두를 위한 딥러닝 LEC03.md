# 모두를 위한 딥러닝 LEC03.
## Simplified Hypothesis.
![image](https://user-images.githubusercontent.com/66259854/93184449-39bbcf80-f777-11ea-9f5b-fa99eac541a5.png)

![image](https://user-images.githubusercontent.com/66259854/93184137-dd58b000-f776-11ea-890d-b787b6d3cf49.png)

## **Minimize.** //최소화
*Cost가 제일 낮은 값을 찾는 것이다.*

## **Gradient Descent Alogorithm.** //경사하강법

![image](https://user-images.githubusercontent.com/66259854/93184184-e9dd0880-f776-11ea-8600-4bdaa0d1fad9.png)

1. Minimize cost function.
2. Gradient descent is used many minimization problems.
3. For a given cost function, cost(W, b), it will find W, b to minimize cost.
4. It can be applied to more general function: cost(w1, w2, …).

## Gradient Descent Algorithm Works.
1. Start with initial guesses
   1)	Start at 0,0 (or any other value)
   2)	Keeping changing W and b a little bit to try and reduce cost(W, b)
2. Each time you change the parameters, you select the gradient which reduces cost(W, b) the most possible
3. Repeat
4. Do so until you converge to a local minimum
5. Has an interesting property
   1) Where you start can determine which minimum you end up

## Formal Definition. //미분
![image](https://user-images.githubusercontent.com/66259854/93184235-f5c8ca80-f776-11ea-9518-e322cb637551.png)

## Convex Function. //볼록함수
![image](https://user-images.githubusercontent.com/66259854/93184247-f8c3bb00-f776-11ea-870c-76c6df31daaf.png)

Cost  /  W  /  b 

![image](https://user-images.githubusercontent.com/66259854/93184302-05e0aa00-f777-11ea-8712-4116f237efde.png)

![image](https://user-images.githubusercontent.com/66259854/93184314-0a0cc780-f777-11ea-9f6a-15d5672ee1a1.png)
