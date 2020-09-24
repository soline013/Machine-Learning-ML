# 모두를 위한 딥러닝 LEC09.
## XOR Problem Using Neural Net(NN).
1. Multiple Logisitic Regression Units.
   ![image](https://user-images.githubusercontent.com/66259854/94162514-c24a1680-fec1-11ea-9c0f-c5771e154286.png)

2. 계산 과정.
   ![image](https://user-images.githubusercontent.com/66259854/94162525-c5450700-fec1-11ea-83e7-7ea5455d7860.png)

   1) $x_1, x_2$를 대입하여 Sigmoid Function으로 $y_1, y_2$를 얻는다.
   2) $y_1, y_2$를 대입하여 Sigmoid Function으로 $y^ ̅$를 얻는다.
   3) $y^ ̅$와 XOR의 값을 확인한다.

## *Forward Propagation.* //순전파
![image](https://user-images.githubusercontent.com/66259854/94162622-e0177b80-fec1-11ea-8b18-4e8681962d0c.png)

↓↓↓↓↓↓↓↓↓↓

![image](https://user-images.githubusercontent.com/66259854/94162633-e3126c00-fec1-11ea-9ab4-0f7ceeae394d.png)

## Derivation.
![image](https://user-images.githubusercontent.com/66259854/94162651-e6a5f300-fec1-11ea-9db9-cb1a87a9e476.png)

## *Backpropagation(Chain Rule).* //역전파(합성함수의 미분)
![image](https://user-images.githubusercontent.com/66259854/94162662-e9084d00-fec1-11ea-8a90-f6f173627931.png)

$\frac{\delta f}{\delta w}=5, \frac{\delta f}{\delta x}=-2$
