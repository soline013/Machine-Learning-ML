# Swin Transformer: Hierarchical Vision Transformer using Shifted Windows

[Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://arxiv.org/pdf/2103.14030.pdf)

## Abstract.

- 2021년 3월 마이크로소프트 아시아에서 발표하였고, 2021년 3월 25일 Arixv에 올라왔다.
- 일반적인 Transformer와 다르게 Hierarchical Transformer 구조를 제시하고, Swin(Shifted Window)로 계산된다.
- 다양한 Vision Tasks에서 Backbone으로 사용할 수 있다.
    - Image Classification(86.4 top-1 accuracy on ImageNet-1K)
    - Object Detection(58.7 box AP and 51.1 mask AP on COCO test-dev)
    - Sementic Segmentation(53.5 mIoU on ADE20K val)
    - 이전의 SOTA보다 +2.7 box AP and +2.6 maxk AP on COCO이고, +3.2 mIoU on ADE20K이다.

## 1. Introduction.

![Swin%20Transformer%20Hierarchical%20Vision%20Transformer%20u%20ab32c632ddb84ed991fecd1307f67057/Swin_Transformer1.png](Swin%20Transformer%20Hierarchical%20Vision%20Transformer%20u%20ab32c632ddb84ed991fecd1307f67057/Swin_Transformer1.png)

1. ViT는 각각의 Patch가 나머지 Patch와 Self-attention을 수행한다.

    Quadratic Computational Complexity로 비용이 크고, 모든 Patch의 크기가 동일하다.

2. Swin Transformer는 Window로 나누고, Window의 Patch만 Self-attention을 수행한다.

    Linear Computational Complexity이고, 처음에는 작은 Patch로 시작하여 점점 Patch를 Merge 한다. (4X) → (8X) → (16X)

    Computational Complexity이 이미지 크기에 따라 선형으로 나타난다고 하는데, 자세한 설명은 나오지 않는다.

    Feature Pyramid Network(FPN)이나 U-Net과 같은 Dense Prediction이 가능하다.

![Swin%20Transformer%20Hierarchical%20Vision%20Transformer%20u%20ab32c632ddb84ed991fecd1307f67057/Swin_Transformer2.png](Swin%20Transformer%20Hierarchical%20Vision%20Transformer%20u%20ab32c632ddb84ed991fecd1307f67057/Swin_Transformer2.png)

1. Layer L에서 Self-attention이 일어나고, Layer L+1에서 Window Shift 이후 Self-attention이 일어난다.
2. Layer L+1은 Layer L에서 진행한 Self-attention의 영향을 받는다.

## 2. Related Work.

1. CNN and Variants.
2. Self-attention Based Backbone Architectures.
3. Self-attention/Transformers to Complement CNNs.
4. Transformer Based Vision Backbones.

## 3. Method.

- 3.1. Overall Architecture.

    ![Swin%20Transformer%20Hierarchical%20Vision%20Transformer%20u%20ab32c632ddb84ed991fecd1307f67057/Swin_Transformer3.png](Swin%20Transformer%20Hierarchical%20Vision%20Transformer%20u%20ab32c632ddb84ed991fecd1307f67057/Swin_Transformer3.png)

    (a) Architectures.

    1. Patch Partition.

        ![Swin%20Transformer%20Hierarchical%20Vision%20Transformer%20u%20ab32c632ddb84ed991fecd1307f67057/1.png](Swin%20Transformer%20Hierarchical%20Vision%20Transformer%20u%20ab32c632ddb84ed991fecd1307f67057/1.png)

        ViT와 같은 Patch Splitting Module을 사용하여 Input Image를 Patch로 분리한다.

        각 Patch는 Token으로 처리된다.

        논문에서 사용된 Input Image의 크기는 224X224이다.

    2. Linear Embedding.

        Linear Embedding Layer는 Feature Dimension을 Anarbitrary Dimension C로 만든다.

        C는 Swin-T를 기준으로 96-d, 192-d, 384-d, 768-d이다.

    3. Swin Transformer Block.

        Swin Transformer Block의 구조는 (b)와 같다.

        (b)-Left

        1. Layer Norm
        2. Windowing Multi Head Self Attention
        3. Layer Norm
        4. GELU MLP(Multilayer Perceptron)

        (b)-Right

        1. Layer Norm
        2. Shifted Windowing Multi Head Self Attention
        3. Layer Norm
        4. GELU MLP(Multilayer Perceptron)

    4. Patch Merging.

        Patch Merging을 통해 (4X) → (8X) → (16X)로 변한다.

        논문 Figure 기준으로 Window 4개가 하나의 Window가 된다.

    - 번외. GELU(Gaussian Error Linear Unit).

        ![Swin%20Transformer%20Hierarchical%20Vision%20Transformer%20u%20ab32c632ddb84ed991fecd1307f67057/Untitled.png](Swin%20Transformer%20Hierarchical%20Vision%20Transformer%20u%20ab32c632ddb84ed991fecd1307f67057/Untitled.png)

        $$\text{GELU}(x) = xP(X \le x) = x \Phi(x) = x \cdot \frac{1}{2} [1 + \text{erf}(x / \sqrt{2})] \\ \text{GELU}(x) = 0.5x(1 + \text{tanh}(\sqrt{2/\pi}(x + 0.044715 x^3)))$$

        비선형 활성화 함수 중 하나이다.

        [GELU (Gaussian Error Linear Unit)](https://hongl.tistory.com/236)

        [[Computer Vision] GELU](https://velog.io/@tajan_boy/Computer-Vision-GELU)

- 3.2. Shifted Window Based Self Attention.
    1. Self-attention in non-overlapped windows.

        $$\Omega(\text{MSA}) = 4hxC^2 + 2(hw)^2C, \qquad (1) \\ \Omega(\text{W-MSA}) = 4hwC^2 + 2M^2 hwC, \; (2)$$

        1. (1)의 경우, ViT에서 사용하는 일반적인 MSA로 이미지 크기의 제곱 연산, Quadratic한 연산이다.
        2. (2)의 경우, Swin Transformer에서 사용하는 W-MSA로 Window Size M의 제곱 연산, Linear한 연산이다.
            1. ViT에 비해 Image Size가 크더라도 연산량이 많이 줄어든다.
            2. 논문에서 M은 7로 고정된다. (논문 Figure 기준으로 M = 4이다.)

        Stage 1 기준 설명

        논문의 Figure는 실제와 다르기 때문에,
        실제 논문을 기준으로 부가 설명을 추가하였다.

        Patch Size = 1 X 1
        Patch Number = 56 X 56 개
        한 Window에 들어가는 Patch Number = 7 X 7 개

        M = 7
        Window Size = M X M = 7 X 7
        Window Number = 8 X 8 개

    2. Shifted window partitioning in successive blocks.

        $$\hat{z}^l = \text{W-MSA}(\text{LN}(z^{l-1})) + z^{l-1}, \\ z^l = \text{MLP}(\text{LN}(\hat{z}^l)) + \hat{z}^l, \\ \hat{z}^{l+1} = \text{SW-MSA}(\text{LN}(z^l)) + z^l, \\ z^{l+1} = \text{MLP}(\text{LN}(\hat{z}^{l+1})) + \hat{z}^{l+1},$$

        1. Window의 기반의 Self Attention은 독립적으로 수행되기 때문에 Window 사이의 연결이 필요하다.
        2. 따라서 Figure 2.처럼 Swin Transformer Block에서 Window Shift를 수행한다.

    3. Efficient batch computation for shifted configuration.

        ![Swin%20Transformer%20Hierarchical%20Vision%20Transformer%20u%20ab32c632ddb84ed991fecd1307f67057/Swin_Transformer4.png](Swin%20Transformer%20Hierarchical%20Vision%20Transformer%20u%20ab32c632ddb84ed991fecd1307f67057/Swin_Transformer4.png)

        1. 하지만 Figure 2.처럼 Window Shift를 수행하게 되면 문제가 생기는데, 논문 Figure를 기준으로 Window의 개수가 4개에서 9개로 늘어나고, Window Size 또한 4X4, 4X2, 2X2로 달라진다.
        2. Window Size를 맞추기 위해 Padding을 사용할 수 있으나 비용이 크기 때문에, Cyclic Shift를 사용하여 Figure 4.처럼 A, B, C를 이동한다.
        3. 그리고 그대로 Self Attention을 수행하는데, A, B, C는 반대편에 있었기 때문에 Masked MSA를 수행한다.
        4. Masked MSA이 끝나면 Reverse Cyclic Shift로 원래대로 이동한다.

    4. Relative position bias.

        ![Swin%20Transformer%20Hierarchical%20Vision%20Transformer%20u%20ab32c632ddb84ed991fecd1307f67057/Untitled%201.png](Swin%20Transformer%20Hierarchical%20Vision%20Transformer%20u%20ab32c632ddb84ed991fecd1307f67057/Untitled%201.png)

        ![Swin%20Transformer%20Hierarchical%20Vision%20Transformer%20u%20ab32c632ddb84ed991fecd1307f67057/_2021-07-18_12.32.29.png](Swin%20Transformer%20Hierarchical%20Vision%20Transformer%20u%20ab32c632ddb84ed991fecd1307f67057/_2021-07-18_12.32.29.png)

        $$\text{Attention}(Q, K, V)  = \text{Softmax}(QK^T / \sqrt{d} + B) V \\ B \in \R^{M^2 \times M^2}$$

        Transformer와 ViT를 보면 Positional Encoding을 더하지만 Swin Transformer에는 이 Positional Encoding이 없다.

        대신 Self Attention 과정에서 Relative Position Bias를 더한다.

        (0, 0)에서 (2, 2)에 대한 상대좌표는 (2, 2)이고,

        (2, 2)에서 (0, 0)에 대한 상대좌표는 (-2, -2)이다.

- 3.3. Architecture Variants.

    논문에서 제안하는 Swin Transformer의 Base Model은 Swin-B이다.

    M = 7, d(Query Dimension of Each Head) = 32, MLP $\alpha$ = 4로 고정된다.

    Swin-T, Swin-S, Swin-L은 Model Size와 Computational Complexity에서 각각 Swin-B의 (0.25X), (0.5X), (2X)이다.

    - Swin-T: C = 96, Layer Numbers = {2, 2, 6, 2}
    - Swin-S: C = 96, Layer Numbers = {2, 2, 18, 2}
    - Swin-B: C = 128, Layer Numbers = {2, 2, 18, 2}
    - Swin-L: C = 192, Layer Numbers = {2, 2, 18, 2}

## 4. Experiments.

![Swin%20Transformer%20Hierarchical%20Vision%20Transformer%20u%20ab32c632ddb84ed991fecd1307f67057/Swin_Transformer5.png](Swin%20Transformer%20Hierarchical%20Vision%20Transformer%20u%20ab32c632ddb84ed991fecd1307f67057/Swin_Transformer5.png)

(a)

1. ViT-B/16 보다 Parameter 수가 적으나, 성능은 3.4% 높다.
2. 가장 높은 성능은 84.2%로 EffcientNet-B7과 84.3% 비슷한 성능을 보인다.

![Swin%20Transformer%20Hierarchical%20Vision%20Transformer%20u%20ab32c632ddb84ed991fecd1307f67057/Swin_Transformer6.png](Swin%20Transformer%20Hierarchical%20Vision%20Transformer%20u%20ab32c632ddb84ed991fecd1307f67057/Swin_Transformer6.png)

COCO Object Detection, Instance Segmentation에서 높은 성능을 보인다.

![Swin%20Transformer%20Hierarchical%20Vision%20Transformer%20u%20ab32c632ddb84ed991fecd1307f67057/Swin_Transformer7.png](Swin%20Transformer%20Hierarchical%20Vision%20Transformer%20u%20ab32c632ddb84ed991fecd1307f67057/Swin_Transformer7.png)

ADE20K에 대한 Semantic Segmentation도 높은 성능을 보인다.

![Swin%20Transformer%20Hierarchical%20Vision%20Transformer%20u%20ab32c632ddb84ed991fecd1307f67057/Swin_Transformer8.png](Swin%20Transformer%20Hierarchical%20Vision%20Transformer%20u%20ab32c632ddb84ed991fecd1307f67057/Swin_Transformer8.png)

1. W-MSA만 사용했을 때와, SW-MSA까지 사용했을 때의 성능 변화.
2. 상대좌표와 절대좌표 여부에 따른 성능 변화.

![Swin%20Transformer%20Hierarchical%20Vision%20Transformer%20u%20ab32c632ddb84ed991fecd1307f67057/Swin_Transformer9.png](Swin%20Transformer%20Hierarchical%20Vision%20Transformer%20u%20ab32c632ddb84ed991fecd1307f67057/Swin_Transformer9.png)

각 Window 기반 방식에 따른 Real Speed 차이.

![Swin%20Transformer%20Hierarchical%20Vision%20Transformer%20u%20ab32c632ddb84ed991fecd1307f67057/Swin_Transformer10.png](Swin%20Transformer%20Hierarchical%20Vision%20Transformer%20u%20ab32c632ddb84ed991fecd1307f67057/Swin_Transformer10.png)

Sliding Window와 Shifted Window의 성능 차이.

![Swin%20Transformer%20Hierarchical%20Vision%20Transformer%20u%20ab32c632ddb84ed991fecd1307f67057/Swin_Transformer11.png](Swin%20Transformer%20Hierarchical%20Vision%20Transformer%20u%20ab32c632ddb84ed991fecd1307f67057/Swin_Transformer11.png)

A1. Detailed Architectures.

Architectures에 대해 자세한 설정이 나와 있다.

![Swin%20Transformer%20Hierarchical%20Vision%20Transformer%20u%20ab32c632ddb84ed991fecd1307f67057/Swin_Transformer12.png](Swin%20Transformer%20Hierarchical%20Vision%20Transformer%20u%20ab32c632ddb84ed991fecd1307f67057/Swin_Transformer12.png)

A3. More Experiments에서 A3.1. Image classification with different input size.

A2. Detailed Experimental Settings에서 A2.1. Image classification on ImageNet-1K.

ImageNet-1K Classification에서 다양한 Input Size에 따른 결과.

![Swin%20Transformer%20Hierarchical%20Vision%20Transformer%20u%20ab32c632ddb84ed991fecd1307f67057/Swin_Transformer13.png](Swin%20Transformer%20Hierarchical%20Vision%20Transformer%20u%20ab32c632ddb84ed991fecd1307f67057/Swin_Transformer13.png)

A3. More Experiments에서 A3.2. Different Optimizers for ResNe(X)t on COCO.

A2. Detailed Experimental Settings에서 A2.2. Object detection on COCO와 A2.3. Semantic segmentation on ADE20K.

## 5. Conclusion.

Swin Transformer의 주요 특징을 정리하면 다음과 같다.

- Hierarchical Transformer 구조이다.
- Shifted Window 방식을 사용한다.
- Cyclic Shift를 이용한다.
- Positional Encoding이 없고, Relative Position Bias를 더한다.
- Linear Computational Complexity를 갖는다.
- 다양한 Vision Tasks에서 Backbone으로 사용할 수 있다.
- Dense Prediction이 가능하다.

## Link.

[[논문리뷰] Swin Transformer](https://visionhong.tistory.com/31)