# 이미지는 수정 예정.

# Image Style Transfer Using Convolutional Neural Networks
## 1. Introduction.

2015년에 먼저 발표된 논문은 Image Style Transfer Using Convolutional Neural Networks과 같은 연구팀의 논문이다.

[A Neural Algorithm of Artistic Style](https://arxiv.org/abs/1508.06576)

2016년 발표된 이 논문은 "A Neural Algorithm of Artistic Style"과 많은 내용이 중복된다.

[](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf)

---

![Image%20Style%20Transfer%20Using%20Convolutional%20Neural%20Ne%2063322dbe3c3d4809a339e4c77401a812/_2021-02-14_02.44.46.png](Image%20Style%20Transfer%20Using%20Convolutional%20Neural%20Ne%2063322dbe3c3d4809a339e4c77401a812/_2021-02-14_02.44.46.png)

1. Content Reconstructions → Target Image.
    1. CNN의 처리 단계마다 입력 이미지를 재구성하여 시각화한다.
    2. 하나의 Higher Layers만을 사용한다.
    3. a, b, c 같은 Lower Layers에서는 모든 Content가 남아있어 완벽한 복원을 보인다.
    4. d, e 같은 Higher Layers에서는 Detailed Pixel Information이 사라지지만, 윤곽 같은 High-level Content는 남아있다.
    5. Layers.

        `conv1_2 (a)`

        `- conv2_2 (b)`

        `- conv3_2 (c)`

        `- conv4_2 (d)`

        `- conv5_2 (e)`

        From VGG19 Network.

    - 번외. A Neural Algorithm of Artistic Style과의 차이점.

        Content Reconstructions에서 `conv'N'_2`를 사용한다.
    

2. Style Reconstructions → Artworks.
    1. Style Reconstructions를 위해 CNN Representations 위에 Feature Space를 구축했다.
    2. Style은 같은 Layer의 다른 Features Map 사이의 Correlation으로 정의된다.
    3. 여러 Layer를 포함하는 Multi-scale이다.
    4. Lower Layer에서는 '원본 Content' 정보는 대부분 무시하고 Texture만 복원한다.

        복원이 잘 이루어지므로 Correlation이 높다.

    5. Higher Layer에서는 '원본 Content' 정보를 포함하여 미술 작품과 비슷한 이미지가 나온다.

        복원이 잘 이루어지지 않으므로 Correlation이 낮다.

    6. Layers.

        `conv1_1 (a)`

        `- conv1_1, conv2_1 (b)`

        `- conv1_1, conv2_1, conv3_1 (c)`

        `- conv1_1, conv2_1, conv3_1, conv4_1 (d)`

        `- conv1_1, conv2_1, conv3_1, conv4_1, conv5_1 (e)`

3. Content Representations와 Style Representations는 분리할 수 있다.

    같은 Network를 사용하여 Content와 Style을 복원하고, 그 둘로 새로운 의미 있는 이미지를 만들 수 있다.

### 2. Deep Image Representations.

![https://sanghyukchun.github.io/images/post/92-6.png](https://sanghyukchun.github.io/images/post/92-6.png)

논문에서 CNN Base Network로 VGG19를 사용한다.

VGG19의 구조를 모두 사용하지는 않고, 16개의 Conv Layer와 5개의 Pooling Layer를 사용한다.

Pooling Layer는 Image Reconstruction이라는 목적에 맞게, 기존의 Max Pooling이 아닌 Average Pooling을 사용한다.

- 번외. Average Pooling.

    ![Image%20Style%20Transfer%20Using%20Convolutional%20Neural%20Ne%2063322dbe3c3d4809a339e4c77401a812/_2021-02-13_22.43.25.png](Image%20Style%20Transfer%20Using%20Convolutional%20Neural%20Ne%2063322dbe3c3d4809a339e4c77401a812/_2021-02-13_22.43.25.png)

    [[Part Ⅵ. CNN 핵심 요소 기술] 3. Stochastic Pooling - 라온피플 머신러닝 아카데미 -](https://m.blog.naver.com/PostView.nhn?blogId=laonple&logNo=220830178487&proxyReferer=https:%2F%2Fwww.google.co.kr%2F)

    Average Pooling은 윈도우 영역에서 평균값을 취하는 방식이다.

VGG의 자세한 내용은 아래 논문을 참고하자.

[Very Deep Convolutional Networks for Large-Scale Image Recognition](http://arxiv.org/abs/1409.1556)

---

![Image%20Style%20Transfer%20Using%20Convolutional%20Neural%20Ne%2063322dbe3c3d4809a339e4c77401a812/_2021-02-14_02.53.33.png](Image%20Style%20Transfer%20Using%20Convolutional%20Neural%20Ne%2063322dbe3c3d4809a339e4c77401a812/_2021-02-14_02.53.33.png)

### 2.1. Content Representation.

$$\mathcal{L}_{content}(\vec{p}, \vec{x}, l) = \frac{1}{2} \sum_{i,j} (F^l_{ij} - P^l_{ij})^2. \\ F^l \in R^{N_l \times M_l}$$

$x$: Input Image. White Noise.

$p$: Content Image.

$l$: Layer.

$i,j$: $i$ 번째 필터, $j$ 번째 위치.

$F$: $x$를 통과시킨 Feature Map Matrix.

$P$: $p$를 통과시킨 Feature Map Matrix.

Image는 Filter를 통과하면 Encoding 된다.

- 번외. Frobenius Norm.

    [Matrix norm](https://en.wikipedia.org/wiki/Matrix_norm#Frobenius_norm)

- 번외. Mean Squared Error(MSE) & Mean Squared Deviation(Difference)(MSD).

    [블루래더 - 경제적 자유를 향한 푸른 사다리 : 네이버 블로그](https://blog.naver.com/resumet/221751314721)

    [ADONIS EntertaINMent & Fairies : 네이버 블로그](https://blog.naver.com/sanghan1990/221156229802)

    [Mean squared error](https://en.wikipedia.org/wiki/Mean_squared_error)

---

$$\frac{\delta \mathcal{L}_{content}}{\delta F^l_{ij}} = \begin{cases} (F^l-P^l)_{ij} \ \text{if} \ F^l_{ij} > 0 \\ 0 \qquad \qquad \ \   \text{if} \ F^l_{ij} < 0.
    \end{cases}$$

$x$와 $p$의 Content가 얼마나 다른지 나타내는 Loss Function을 최소화하는 값을 찾는다.

해당 식은 Loss를 Layer의 Activations로 미분한 것이다.

Standard Error Back-propagation을 통해 $x$의 Gradient를 계산한다.

그리고 Initially Random Image $x$를 계속 Update한다.

### 2.2. Style Representation.

$$G^l_{ij} = \sum_k F^l_{ik}F^l_{jk} \qquad (G^l \in R^{N_l \times N_l}). \\ E_l = \frac{1}{4N^2_lM^2_l} \sum_{i,j} (G^l_{ij} - A^l_{ij})^2 \\ \mathcal{L}_{style}(\vec{a}, \vec{x}) = \sum^L_{l=0} w_l E_l$$

$x$: Input Image. White Noise.

$a$: Style Image.

$G$: Gram Matrix. 

$A$: $a$를 통과시킨 Feature Map Matrix.

$E$: Style Loss Contribution.

$i,j$: $i$ Feature Map, $j$ Feature Map.

$N_l$: $l$ Layer의 Filter 개수.

$M_l$: $l$ Layer에서 Filter의 가로 세로 곱.

1. Gram Matrix.

    Style은 같은 Layer의 다른 Features Map 사이의 Correlation이고, Filter의 개수는 $N_l$이므로, $R^{N_l \times N_l}$이다.

    $l$ Layer에서, $i$ Feature Map과 $j$ Feature Map을 내적하여 Correlation을 구한다.

    이렇게 계산하여 나온 Matrix가 Gram Matrix이다.

    Expectation에 대한 내용은 인용문을 참고하자.

    > 이때, correlation을 계산하기 위하여 각각의 filter의 expectation 값을 사용하여 correlation matrix를 계산한다고 한다. 즉, l번째 layer에서 필터가 100개 있고, 각 필터별로 output이 400개 있다면, 각각의 100개의 필터마다 400개의 output들을 평균내어 값을 100개 뽑아내고, 그 100개의 값들의 correlation을 계산했다는 것이다.

    [그람 행렬](https://ko.wikipedia.org/wiki/%EA%B7%B8%EB%9E%8C_%ED%96%89%EB%A0%AC)

    [Gramian matrix](https://en.wikipedia.org/wiki/Gramian_matrix)

2. Style Loss Contribution & Style Loss.

    $L$은 Loss에 영향을 주는 Layer 개수이다.

    $w_l$은 활성화할 Layer를 나타내는 가중치로, 합은 1이다.

    선택한 레이어를 제외하고 모두 0으로 만드는 방법을 사용한다.

    e.g. Figure 1의 a 그림은 conv1_1만 선택되므로 conv1_1의 $w_l=1$이고, 나머지는 0이다.

---

$$\frac{\delta E_l}{\delta F^l_{ij}} = \begin{cases} \frac{1}{N^2_l M^2_l} ((F^l)^T (G^l - A^l))_{ji} \ \text{if} \ F^l_{ij} > 0 \\ 0 \qquad \qquad \qquad \qquad \qquad \text{if} \ F^l_{ij} < 0. \end{cases}$$

$x$와 $a$의 Style이 얼마나 다른지 나타내는 Loss Function을 최소화하는 값을 찾는다.

해당 식은 Loss를 Layer의 Activations로 미분한 것이다.

Standard Error Back-propagation을 통해 $x$의 Gradient를 계산한다.

### 2.3. Style Transfer.

$$\mathcal{L}_{total}(\vec{p}, \vec{a}, \vec{x}) = \alpha \mathcal{L}_{content}(\vec{p}, \vec{x}) + \beta \mathcal{L}_{style}(\vec{a}, \vec{x})$$

Content Loss와 Style Loss의 합을 최소화한 값이다.

$\alpha$와 $\beta$는 각각 Content, Style의 가중치로, 숫자가 클수록 Content의 특성이 잘 나타난다.

보통 $\alpha/\beta = 10^{-3}, \, 10^{-4}$을 사용한다.

---

$$\frac{\delta \mathcal{L}_{total}}{\delta \vec{x}} \qquad \vec{x} := \vec{x} - \lambda \frac{\delta \mathcal{L}_{total}}{\delta \vec{x}}$$

Total Loss의 미분은 왼쪽과 같다.

Gradient는 Numerical Optimisation Strategy인 L-BFGS의 입력으로 사용된다.

그리고 $x$를 Update( : ) 한다.

L-BFGS의 자세한 내용은 링크를 참고하자.

[위키독스](https://wikidocs.net/22155)

[[기계학습] L-BFGS-B 란?](https://m.blog.naver.com/PostView.nhn?blogId=sw4r&logNo=221475376608&proxyReferer=https:%2F%2Fwww.google.co.kr%2F)

[L-BFGS](https://www.notion.so/L-BFGS-bea4d310410349488993c12158560fa4)

비슷한 Scale로 이미지 정보를 추출하는 것이 좋다.

따라서 Style Image의 Feature Representations를 계산하기 전, Style Image를 Content Image와 비슷한 크기로 조절한다.

# 추가 예정.

## 3. Results.

![Image%20Style%20Transfer%20Using%20Convolutional%20Neural%20Ne%2063322dbe3c3d4809a339e4c77401a812/_2021-02-14_02.57.33.png](Image%20Style%20Transfer%20Using%20Convolutional%20Neural%20Ne%2063322dbe3c3d4809a339e4c77401a812/_2021-02-14_02.57.33.png)

1. A: Neckarfront in Tübingen, Germany.
2. B: The Shipwreck of the Minotaur by J.M.W. 5 Turner, 1805.
3. C: The Starry Night by Vincent van Gogh, 1889.
4. D: Der Schrei by Edvard Munch, 1893.
5. E: Femme nue assise by Pablo Picasso, 1910.
6. F: Composition VII by Wassily Kandinsky, 1913.

## 링크.

[Style Transfer](https://blog.lunit.io/2017/04/27/style-transfer/)

[CNN을 활용한 스타일 전송(Style Transfer) | 꼼꼼한 딥러닝 논문 리뷰와 코드 실습](https://www.youtube.com/watch?v=va3e2c4uKJk&fbclid=IwAR05YuKVXga_kOD-0W-YO42SCIUN7REu20YmQCoEaztrh9Is29o3ule_874)