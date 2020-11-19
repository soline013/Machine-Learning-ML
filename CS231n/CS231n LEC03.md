# CS231n LEC03.
## Stanford University, Spring 2017.
Loss Functions and Optimization.

## Recall from Last Time.
![image](https://user-images.githubusercontent.com/66259854/99666333-395cef00-2aae-11eb-8b48-74a5cbae10cd.png)
![image](https://user-images.githubusercontent.com/66259854/99666349-3eba3980-2aae-11eb-8788-96c65bbf76c3.png)
![image](https://user-images.githubusercontent.com/66259854/99666358-411c9380-2aae-11eb-815a-c4cd66b5dd6c.png)

## Start LEC03, Linear Classifier.
![image](https://user-images.githubusercontent.com/66259854/99666371-4548b100-2aae-11eb-93ca-fb1a2978f761.png)

지난 시간, 임의의 행렬 W를 가지고 예측한 10 Class Scores.

  1. 고양이 사진 - Cat 카테고리가 2.9점이나, 점수가 더 높은 다른 카테고리가 존재한다.
  2. 자동차 사진 - Automobile 카테고리가 6.04점으로 제일 높다. 잘 예측한 것.
  3. 개구리 사진 - Frog 카테고리가 -4.34이다.

가장 나은 W를 찾기 위한 방법 → Loss Function.

* * *

![image](https://user-images.githubusercontent.com/66259854/99666381-48dc3800-2aae-11eb-94a1-f27ff815cd35.png)

X=Input, Y=Label, Target (In CIFAR10, 1~10 or 0~9)

$L = 1/N \sum_i L_i(f(x_i, W), y_i)$

![image](https://user-images.githubusercontent.com/66259854/99666384-4bd72880-2aae-11eb-86f8-63637aa336b9.png)
![image](https://user-images.githubusercontent.com/66259854/99666387-4e398280-2aae-11eb-8e51-386d964b561a.png)
![image](https://user-images.githubusercontent.com/66259854/99666390-50034600-2aae-11eb-97cb-09e20ff38273.png)
![image](https://user-images.githubusercontent.com/66259854/99666398-51cd0980-2aae-11eb-9219-c9091c5d3453.png)
![image](https://user-images.githubusercontent.com/66259854/99666404-542f6380-2aae-11eb-84e9-7d1e5830456f.png)
![image](https://user-images.githubusercontent.com/66259854/99666415-572a5400-2aae-11eb-92ad-5e7a19c75cd5.png)
![image](https://user-images.githubusercontent.com/66259854/99666427-5a254480-2aae-11eb-90a3-bf7329504de1.png)
![image](https://user-images.githubusercontent.com/66259854/99666431-5bef0800-2aae-11eb-9fdc-e746d52cf021.png)
![image](https://user-images.githubusercontent.com/66259854/99666442-5e516200-2aae-11eb-9be9-5e63487ec406.png)
![image](https://user-images.githubusercontent.com/66259854/99666455-614c5280-2aae-11eb-9b12-36778e0f53da.png)
![image](https://user-images.githubusercontent.com/66259854/99666462-63161600-2aae-11eb-93ea-5ec3a5eca9ea.png)
![image](https://user-images.githubusercontent.com/66259854/99666481-67daca00-2aae-11eb-8d4c-68f2fa4b185d.png)
![image](https://user-images.githubusercontent.com/66259854/99666492-6ad5ba80-2aae-11eb-9653-9654153bb3a2.png)
![image](https://user-images.githubusercontent.com/66259854/99666501-6c9f7e00-2aae-11eb-9893-d9114639c2b7.png)
![image](https://user-images.githubusercontent.com/66259854/99666507-6e694180-2aae-11eb-9fa8-3b299c114971.png)
![image](https://user-images.githubusercontent.com/66259854/99666514-70cb9b80-2aae-11eb-8ede-1ce5d9175e7a.png)
![image](https://user-images.githubusercontent.com/66259854/99666519-72955f00-2aae-11eb-9d43-319cfb668371.png)
![image](https://user-images.githubusercontent.com/66259854/99666529-74f7b900-2aae-11eb-8eb3-3f0e5b25a895.png)
![image](https://user-images.githubusercontent.com/66259854/99666539-775a1300-2aae-11eb-9394-79f7e788bf82.png)
![image](https://user-images.githubusercontent.com/66259854/99666553-79bc6d00-2aae-11eb-8a95-ecc85eec6e9c.png)
![image](https://user-images.githubusercontent.com/66259854/99666558-7c1ec700-2aae-11eb-930f-7b8b39ec14db.png)
![image](https://user-images.githubusercontent.com/66259854/99666569-7de88a80-2aae-11eb-89e6-296ed91011a9.png)
![image](https://user-images.githubusercontent.com/66259854/99666575-7fb24e00-2aae-11eb-878c-d9af4a3d74b2.png)
![image](https://user-images.githubusercontent.com/66259854/99666578-8214a800-2aae-11eb-88b2-eb176e23d573.png)
