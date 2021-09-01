# Wiki.

Wiki는 간단한 이해와 복기를 돕습니다.

[Loss](##loss)

[Object Detection](##ObjectDetection)


## <div id="loss">Loss</div>
    
- Cross-Entropy Loss

    $$\text{CE} = - \sum_i^C t_i log(s_i)$$

    Loss Funcion 중 하나로, $t_i$는 Ground Truth, $s_i$는 각 클래스 i에 대한 Output Score 벡터의 i번째 요소이다.

    [Cross-entropy 의 이해: 정보이론과의 관계](https://3months.tistory.com/436)

- Categorical Cross-Entropy Loss

    $$f(s)_i = \frac{e^{s_j}}{\sum^C_j e^{s_j}}, \quad \text{CE} = - \sum_i^C t_i log(f(s)_i)$$

    Softmax Activation Function 뒤에서 주로 사용된다.

    Multi-class Classification에 사용하는 Loss이다.

- Binary Cross-Entropy Loss

    $$f(s_i) = \frac{1}{1+e^{-s_i}}, \quad \text{CE} = - \sum_{i=1}^{C'=2} t_i log(f(s_i)) = -t_1log(f(s_1)) - (1-t)log(1-f(s_1))$$

    Sigmoid Activation Function 뒤에서 주로 사용된다.

    Binary-class Classification에 사용하는 Loss이다.

## Object Detection

- Sliding Window

- Region Proposal

- Window Shift

- Bounding Box

    객체의 위치를 표시하기 위해 사용하는, 객체를 둘러싼 직사각형 박스를 의미한다.
    
- Two Stage Method

    Two Stage Method는 Region Proposal 단계와 Detector 단계로 구분된다.

    Region Proposal 단계는 Localization을 수행한다.
    Detector 단계는 Multi-task 문제로, 추가적인 Localization을 위해 Regression을, Classification을 위해 SVM 등을 거친다.

    One Stage Method에 비해 정확도는 높지만 속도가 느리다.

    e.g. R-CNN 계열

- One Stage Method

    One Stage Method는 Single Network로 해결한다.

    Localization과 Classification이 동시에 이루어진다.

    Two Stage Method에 비해 정확도는 낮지만 속도가 빠르다.

    e.g. YOLO, SSD

## Computer Vision

- MediaPipe

    Computer Vision 라이브러리이다.

    Cross-platform과 Live and Streaming Media ML Solution을 제공한다.

- OpenCV

    Computer Vision에서 많이 사용되는 라이브러리이다.

## Activation Function

- Softmax

- Sigmoid

- ReLU

- GELU

## Fundamental Concept

- End-to-end

    입력에서 출력까지의 모든 과정이 하나의 Network에서 진행되는 것을 의미한다.
