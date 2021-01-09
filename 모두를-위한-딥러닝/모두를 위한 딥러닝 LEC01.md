# 모두를 위한 딥러닝 LEC01

## About ML.
1. Limitations of Explicit Programming.

   Spam filter : many rules
     
   Automatic driving : too many rules → 고려할 사항이 너무 많음.

2. **Machine Learning.** //Bold: 주요 개념

   “Field of study that gives computers the ability to learn without being explicitly programmed.” Arthur Samuel(1959)

*머신러닝은 일종의 소프트웨어로, 프로그램 자체가 학습하여 배우는 명령을 갖는 소프트웨어이다.* //분명하지 않은 경우 재정리

## Learning.
1. Supervised Learning: Learning with labeled examples - Training Set
  
   ![image](https://user-images.githubusercontent.com/66259854/93174607-3b7e9680-f769-11ea-98cc-1d1dfc8f984a.png)
  
   Most commom problem type in ML
   1) Image Labeling
   2) Email Spam Filter
   3) Prediciting Exam Score
  
   Types of Supervised Learning
   1) 0~100까지의 점수     **Regression**
   2) Pass/Non-pass       **Binary Classification**
   3) A, B, C, D, E grade **Multi-Label Classification**
  
2. Unsupervised Learning: un-labeles data
  
   Google news grouping
  
   Word clustering



# 모두를 위한 딥러닝 LAB01.md

## Tensorflow.
Tensorflow is an open source software library for numerical computation using data flow graphs.

## **Data Flow Graph.**
*Data Flow Graph는 노드, 엣지로 연산이 일어나 어떤 작업을 할 수 있는 것이다.*
1. **Nodes** in the graph represent mathematical operations. //연산
2. **Edges** represent the multidimensional data arrays(tensors) communicated between them. //Tensor

![image](https://user-images.githubusercontent.com/66259854/93174674-58b36500-f769-11ea-8786-289ce2fdd635.png)

## **Tensor.**
*임의의 차원을 갖는 배열들을 의미한다. 다차원 배열.*
1. **Ranks** : n 차원

![image](https://user-images.githubusercontent.com/66259854/93174679-5a7d2880-f769-11ea-8f4e-e06496eba5be.png)

2. **Shapes** : [가장 바깥 요소의 개수, 가장 안쪽 요소의 개수]

![image](https://user-images.githubusercontent.com/66259854/93174686-5c46ec00-f769-11ea-9fd3-1791e92e1ab7.png)

3. **Types** : 자료형.

![image](https://user-images.githubusercontent.com/66259854/93174687-5ea94600-f769-11ea-9925-6c29d1678c2e.png)
