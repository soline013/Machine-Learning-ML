# 모두를 위한 딥러닝 LEC05.
## *Binary Classification.*
![image](https://user-images.githubusercontent.com/66259854/93319260-a0f68400-f84a-11ea-9e6b-8e4e4f5f47cf.png)
![image](https://user-images.githubusercontent.com/66259854/93319278-a3f17480-f84a-11ea-803a-29e5679b5de6.png)
1. **0, 1 Encoding**
   1) Spam Detection: Spam(1) or Ham(0)
   2) Facebook Feed: Show(1) or Hide(0)
   3) Credit Card Fraudulent Transaction: Legitimate(0) or Fraud(1)

## Pass(1)/Fail(0) Based on Study Hours. (If Linear Regression.)
![image](https://user-images.githubusercontent.com/66259854/93319287-a653ce80-f84a-11ea-9fa9-c29e280da06d.png)

1. 문제점
   1) 수치가 커지면(50hours) 선이 기울어진다.
   2) 선이 기울어지면 기준값(0.5)의 위치가 바뀐다.
   3) 기준값의 위치가 바뀌면 기존 결과의 판단이 달라진다.
   4) 1보다 큰 값, 0보다 작은 값이 나올수 있다.

## *Logistic Regression.* //로지스틱 회귀
![image](https://user-images.githubusercontent.com/66259854/93319420-cedbc880-f84a-11ea-8161-d3d7e234b057.png)

![image](https://user-images.githubusercontent.com/66259854/93319293-ab188280-f84a-11ea-8de7-866a84826641.png)

1. **Logistic Function, Sigmoid Function**
   1) Sigmoid: Curved in two directions, like the letter “S”, or the Greek ς(Sigma) //들여쓰기 예외
   
## Cost Function.
![image](https://user-images.githubusercontent.com/66259854/93319309-b10e6380-f84a-11ea-9684-56f4e6838da9.png)
   
1. 문제점

   **Local Cost**에서 끝나 **Global Cost**로 갈 수 없다.

2. .

   ![image](https://user-images.githubusercontent.com/66259854/93319512-e74be300-f84a-11ea-9526-1ecdfcae2758.png)

3. -log(H(x)) : y=1
   
   H(x)=1 → cost=0
   
   H(x)=0 → cost=∞
   
   ![image](https://user-images.githubusercontent.com/66259854/93319535-f29f0e80-f84a-11ea-982e-e817e606da5c.png)

4. -log⁡(1-H(x)) ∶ y=0

   H(x)=0 → cost=0
   
   H(x)=1 → cost=∞

   ![image](https://user-images.githubusercontent.com/66259854/93319542-f5016880-f84a-11ea-9681-4f734c0755a8.png)
   
## Gradient Descent Algorithm.
![image](https://user-images.githubusercontent.com/66259854/93319573-fdf23a00-f84a-11ea-9287-edb61ddc04aa.png)
