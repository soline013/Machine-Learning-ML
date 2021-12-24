# Normal Wiki.

Wiki는 간단한 이해와 복기를 돕습니다.

[Loss](#loss)

[Activation Function](#activationfunction)

[Fundamental Concept](#fundamentalconcept)

## <div id="loss">Loss</div>

- Cross-Entropy Loss

    $$\text{CE} = - \sum_i^C t_i log(s_i)$$

    Loss Funcion 중 하나로, $t_i$는 Ground Truth, $s_i$는 각 클래스 i에 대한 Output Score 벡터의 i번째 요소이다.

    [Cross-entropy 의 이해: 정보이론과의 관계](https://3months.tistory.com/436)

</br>

- Categorical Cross-Entropy Loss

    $$f(s)_i = \frac{e^{s_j}}{\sum^C_j e^{s_j}}, \quad \text{CE} = - \sum_i^C t_i log(f(s)_i)$$

    Softmax Activation Function 뒤에서 주로 사용된다.

    Multi-class Classification에 사용하는 Loss이다.

</br>

- Binary Cross-Entropy Loss

    $$f(s_i) = \frac{1}{1+e^{-s_i}}, \quad \text{CE} = - \sum_{i=1}^{C'=2} t_i log(f(s_i)) = -t_1log(f(s_1)) - (1-t)log(1-f(s_1))$$

    Sigmoid Activation Function 뒤에서 주로 사용된다.

    Binary-class Classification에 사용하는 Loss이다.

## <div id="activationfunction">Activation Function</div>

- Softmax

- Sigmoid

- ReLU

- GELU

## <div id="fundamentalconcept">Fundamental Concept</div>

- End-to-end

    입력에서 출력까지의 모든 과정이 하나의 Network에서 진행되는 것을 의미한다.

## Framework

- Tensorflow

- Pytorch

- Sklearn


- JAX
