# 모두를 위한 딥러닝 LEC04.
## *Multi-Variable* Linear Regression.
![image](https://user-images.githubusercontent.com/66259854/93193232-8ad0c100-f781-11ea-93d6-f66ffb066b7f.png)

## Cost Function. //손실함수
![image](https://user-images.githubusercontent.com/66259854/93193268-97edb000-f781-11ea-822e-fd144edd6f7f.png)

## *Matrix.* //행렬
![image](https://user-images.githubusercontent.com/66259854/93193708-28c48b80-f782-11ea-8acf-0f1fda13b401.png)

## *Matrix Multiplication.*
*H(X) = X W*

1. ![image](https://user-images.githubusercontent.com/66259854/93193739-34b04d80-f782-11ea-9c31-405767b78761.png)
2. ![image](https://user-images.githubusercontent.com/66259854/93193747-3712a780-f782-11ea-83a1-b41e332b97a7.png)
   1) [n, ~3~] → [~3~, m] → [n, m]
   2) 행: Instance.

## Wx & XW.
![image](https://user-images.githubusercontent.com/66259854/93193756-3a0d9800-f782-11ea-8942-c6477288d9be.png)



# 모두를 위한 딥러닝 LAB04.md
## *Slicing.*
![image](https://user-images.githubusercontent.com/66259854/93194963-a4730800-f783-11ea-9b3d-05e57fb04be0.png)

## *Indexing, Slicing, Iterating.*
![image](https://user-images.githubusercontent.com/66259854/93194971-a6d56200-f783-11ea-9c79-863fe63dd522.png)
![image](https://user-images.githubusercontent.com/66259854/93194985-a9d05280-f783-11ea-8ec1-be168e14b471.png)
1. Arrays can be indexed, sliced, iterated much like lists and other sequence types in Python
2. As with Python lists, slicing in NumPy can be accomplished with the colon (:) syntax
3. Colon instances (:) can be replaced with dots (…)

## *Queue Runners.*
![image](https://user-images.githubusercontent.com/66259854/93194998-adfc7000-f783-11ea-879c-7a93f09461be.png)
1. 여러 개의 파일을 읽을 때, Filename Queue에 쌓는다. (+Random Shuffle)
2. Reader로 연결한다.
3. Decoder를 거쳐 Example Queue에 쌓는다.
4. 일정 배치만큼 읽어와 학습시킨다.
