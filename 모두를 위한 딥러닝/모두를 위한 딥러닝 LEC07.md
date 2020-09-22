## 모두를 위한 딥러닝 LEC07.
*About Machine Learning Tips.*

## *Learning Rate.* //학습률
1. Large Learning Rate: **Overshooting.**

![image](https://user-images.githubusercontent.com/66259854/93868399-59b63a80-fd05-11ea-856d-37b55664fe19.png)
![image](https://user-images.githubusercontent.com/66259854/93868405-5c189480-fd05-11ea-97ce-a1fe13880750.png)

2. Small Learning Rate: **Takes too long & Stops at local minimum.** //강조 예외

3. Try several Learning Rates
   1) Observe the Cost Function.
   2) Check it goes down in a reasonable rate.

## *Data(X) Preprocessing* for Gradient Descent. //데이터 전처리.
1. 값의 차이가 심하게 날 경우.

   ![image](https://user-images.githubusercontent.com/66259854/93868417-5f138500-fd05-11ea-86eb-6d606dd3050e.png)

2. 정상적인 경우.

   ![image](https://user-images.githubusercontent.com/66259854/93868511-7f434400-fd05-11ea-99a8-24d483d3058b.png)

3. Data(X) Preprocessing.

   ![image](https://user-images.githubusercontent.com/66259854/93868519-823e3480-fd05-11ea-9e47-68a266afa821.png)

   1) **Standardization** //표준화

      $x'_j=\frac{x_j-μ_j}{σ_j}$
      
      Code: X_std[:, 0] = (X[:, 0] - X[:, 0].mean()) / X[:, 0].std()

## *Overfitting.* //과적합
![image](https://user-images.githubusercontent.com/66259854/93868534-866a5200-fd05-11ea-974e-c997546a80da.png)

1. Our model is very good with Training Data Set. (with memorization)
2. Not good at test dataset or in real use.
3. Solutions for Overfitting
   1)	More Training Data
   2)	Reduce the number of features
   3)	**Regularization**
   
## *Regularization.* //일반화, 정규화
Let’s not have too big numbers in the weight.

1. $Loss=\frac{1}{N}\sum_j D(S(WX_i+b), L_i)+\lambda\sum W^2 (λ=Regularization Strength)$
   1) λ = 0	      No Regularization
	 2) λ = 1	      High Regularization 
	 3) λ = 0.001	  Little Regularization
2. Code: 12reg = 0.001 * tf.reduce_sum(tf.square(W))

## Evaluation using Training Set.
*모든 데이터를 학습시키는 것은 좋은 방법이 아니다.*

![image](https://user-images.githubusercontent.com/66259854/93868549-8c603300-fd05-11ea-8d76-7e70ab465f62.png)

1. 100% Correct (Accuracy)
2. Can memorize

## *Training Sets* & *Test Sets.*
![image](https://user-images.githubusercontent.com/66259854/93868558-8f5b2380-fd05-11ea-914d-13aa36682117.png)

## Training, *Validation* and Test Sets.
*α(Learning Rate) & λ(Regularization Strength)의 값을 조정한다.*

![image](https://user-images.githubusercontent.com/66259854/93868571-94b86e00-fd05-11ea-9127-0f25f3f8e3c9.png)

## *Online Learning.*
*이미 100만개가 입력되어 있을 때, 추가로 10만개의 데이터만 추가하면 된다.*

![image](https://user-images.githubusercontent.com/66259854/93868581-97b35e80-fd05-11ea-8a23-417face4d654.png)

## *MNIST* Dataset.
![image](https://user-images.githubusercontent.com/66259854/93868598-9e41d600-fd05-11ea-8bfc-54a56e55d789.png)
