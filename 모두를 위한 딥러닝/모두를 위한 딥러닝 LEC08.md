# 모두를 위한 딥러닝 LEC08.
## *Activation Functions.* //From a Neuron
![image](https://user-images.githubusercontent.com/66259854/93873306-773ad280-fd0c-11ea-83bc-81c52df1e207.png)

다양한 Activation Functions이 있다.

e. g. Sigmoid, ReLU 등

## Logistic Regression Units.
*이전에 언급한 내용을 통해 여러 개의 출력을 동시에 낼 수 있다.*

![image](https://user-images.githubusercontent.com/66259854/93873309-786bff80-fd0c-11ea-9654-b9ef50434262.png)

## Hardware Implementations.
*Hardware로 기계학습을 구현하는 시도가 있었고, 이후 XOR 문제와 직면한다.*

![image](https://user-images.githubusercontent.com/66259854/93873325-7f930d80-fd0c-11ea-93cc-a908949d6fdd.png)

## *AND/OR/XOR Problem*: Linearly Separable? (Simple).
*AND/OR은 되나 XOR은 불가능하다.*

![image](https://user-images.githubusercontent.com/66259854/93873337-8588ee80-fd0c-11ea-9e8b-77a563861e3a.png)

## *Perceptrons.* //1969 by Marvin Minsky, founder of the MIT AI Lab
![image](https://user-images.githubusercontent.com/66259854/93873343-8883df00-fd0c-11ea-9420-0160dfa415c8.png)

We need to use MLP, multilayer perceptrons (multilayer neural nets)

No one on earth had found a viable way to train MLPs good enough to learn such simple functions.

## *Backpropagtion.* //역전파 //1974, 1982 by Paul Werbos, 1986 by Hinton
![image](https://user-images.githubusercontent.com/66259854/93873349-8ae63900-fd0c-11ea-9d3b-11df29a9dd21.png)

## *Convolutional Neural Networks.* //1959 by Hubel & Wiesel
1. ![image](https://user-images.githubusercontent.com/66259854/93873357-8d489300-fd0c-11ea-8d0f-83ce495c7aa7.png)
2. ![image](https://user-images.githubusercontent.com/66259854/93873365-8f125680-fd0c-11ea-8284-1589ceff0dbe.png)

## A BIG Problem.
![image](https://user-images.githubusercontent.com/66259854/93873372-9174b080-fd0c-11ea-8074-d32eac95b523.png)

1. Backpropagation just did not work well for normal neural nets with many layers
2. Other rising machine learning algorithms: SVM, RandomForest, etc.
3. 1995 "Comparison of Learning Algorithms For Handwritten Digit Recognition" by LeCun et al.found that this new approach worked better.

## Breakthrough. //In 2006 and 2007 by Hinton and Bengio
1. Neural networks with many layers really could be trained well, if the weights are initialized in a clever way rather than randomly.
2. Deep machine learning methods are more efficient for difficult problems than shallow methods.
3. Rebranding to Deep Nets, Deep Learning.

## *ImageNet Classification.* //2010 – 2015
*CNN의 등장으로 이미지를 분류하는 과정의 오류가 줄어들었다.*

![image](https://user-images.githubusercontent.com/66259854/93873377-946fa100-fd0c-11ea-8973-5251d0c1b675.png)
// Error 26.2% to 15.3%

![image](https://user-images.githubusercontent.com/66259854/93873380-96396480-fd0c-11ea-9241-df63ed591812.png)

## Neural Networks That Can Explain Photos.
*사진을 설명하는 수준까지 향상되었다.*

![image](https://user-images.githubusercontent.com/66259854/93873387-99345500-fd0c-11ea-9276-8116664113f1.png)

## *Deep API Learning.*
*홍콩과기대에서 교수님이 진행했던 프로젝트라 들었다.*

![image](https://user-images.githubusercontent.com/66259854/93873395-9c2f4580-fd0c-11ea-8454-dd6581d5a017.png)

## Geoffrey Hinton’s Summary of Findings Up to Today.
1. Our labeled datasets were thousands of times too small.
2. Our computers were millions of times too slow.
3. We initialized the weights in a stupid way.
4. We used the wrong type of non-linearity.
