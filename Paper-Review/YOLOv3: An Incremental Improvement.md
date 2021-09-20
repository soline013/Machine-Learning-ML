# YOLOv3: An Incremental Improvement.

[YOLOv3: An Incremental Improvement](https://arxiv.org/pdf/1804.02767.pdf)

## Abstract.

![YOLOv3 0](https://user-images.githubusercontent.com/66259854/133923123-6f1c1479-36fc-4689-b33e-433bb2ec9af6.png)


- Figure 1은 Focal Loss를 적용한 결과이다.
- YOLOv3는 v1, v2에 비해 정확도가 향상되었고, 여전히 빠르다.
- 320 X 320 YOLOv3는 28.2mAP의 성능과 22ms의 속도를 낼 수 있다.
- SSD와 mAP는 비슷하지만 속도는 3배 더 빠르다.

## 1. Introduction.

이번 YOLOv3는 v2처럼 큰 변화는 없지만, 성능을 향상시키는 개선이 이루어졌다.

## 2. The Deal.

### 2.1. Bounding Box Prediction.

![YOLOv3 1](https://user-images.githubusercontent.com/66259854/133923127-79e41089-020d-4039-96db-df2e317f0895.png)

1. Bounding Box를 예측하는 방법은 YOLOv2와 같다.

    $$b_x = \sigma(t_x) + c_x \\ b_y = \sigma(t_y) + c_y \\ b_w = p_we^{t_w} \\ b_h = p_he^{t_h} \\ Pr(\text{object}) * IOU(\text{b, object}) = \sigma(t_o)$$

    - $t_x, \, t_y, \, t_w, \, t_h, \, t_o$ : Bounding Box의 요소.
    - $(c_x, \, c_y)$ : Cell의 왼쪽 상단 Offset.
    - $(p_w, \, p_h)$ : Anchor Box의 사전 Width, Height.
    - Network는 각 Cell 마다 5개의 Bounding Box를 예측한다.
    - Offset $t_x, \, t_y$의 범위를 [0, 1]로 제한하고, Logistic Activation(Sigmoid)을 사용한다.
    - Loss는 SSE를 사용한다.

2. 기존 YOLO는 Bounding Box를 예측하기 위해 Gird 기반을 사용했다. YOLOv2, v3는 모두 Anchor Box를 사용한다.

3. Localization Loss의 경우, 기존 YOLO는 SSE, YOLOv2도 아래 자료를 보아 SSE, YOLOv3까지 SSE를 사용하는 것 같다.

    ![YOLOv3 8](https://user-images.githubusercontent.com/66259854/133923164-c887ac48-40d8-4947-93d1-b425c11b5581.png)

    [YOLOv2/YOLOv2.md at master · leetenki/YOLOv2](https://github.com/leetenki/YOLOv2/blob/master/YOLOv2.md)

### 2.2. Class Prediction.
1. 기존의 YOLO는 Class Prediction에서 SSE Loss를 사용한다.

    Activation Function은 마지막에는 Linear, 나머지는 Leaky ReLU를 사용한다.

2. YOLOv2도 Class Prediction에서 SSE Loss를 사용한다.

    Activation Function은 Softmax를 사용하는데 Loss는 Cross-entropy를 사용하지 않는 것 같다.

    ![YOLOv3 8](https://user-images.githubusercontent.com/66259854/133923164-c887ac48-40d8-4947-93d1-b425c11b5581.png)

    [YOLOv2/YOLOv2.md at master · leetenki/YOLOv2](https://github.com/leetenki/YOLOv2/blob/master/YOLOv2.md)

    [YOLO의 loss function에 대해](https://brunch.co.kr/@kmbmjn95/35)

3. YOLOv3는 위와 다르게 Binary Cross-entropy Loss를 사용한다.

    Binary인 만큼 Activation Function도 Softmax가 아닌 독립적인 Logistic Classifier를 사용하는데, Sigmoid로 보인다.

    이 방법이 Open Image Dataset과 같은 복잡한 데이터셋으로 학습할 때 좋다고 한다. (e.g. Woman and Person)

### 2.3. Predictions Across Scales.

![YOLOv3 9](https://user-images.githubusercontent.com/66259854/133923165-ee122f15-b33d-4f37-9a44-973cdac9966c.png)

![YOLOv3 10](https://user-images.githubusercontent.com/66259854/133923166-e00c6ae4-a1ca-43ff-8100-bca56f44af0e.jpg)

1. YOLOv3는 세 가지의 다른 Scale에서 Box를 예측한다.
2. Scale에서 FPN와 비슷하게 Feature를 추출한다.

<br>

3. Base Feature Extractor로부터 Darknet-53에 몇 개의 Conv Layer를 추가한다.
4. 이 마지막에서 3-d Tensor의 형태로 Bounding Box, Objectness, Class Prediction을 예측한다. - `1st Scale`
    - Tensor: $N \times N \times [3 * (4+1+80)]$
    - Bounding Box: 3
    - Bounding Box Offset: 4(x, y, w, h)
    - Objectness: 1
    - COCO Dataset Class: 80
5. 이전 2개 Layer의 Feature Map을 가져와서 2배씩 Upsampling 하고, 초기 Feature Map과 Concat 한다.
    - 이 과정을 통해 Semantic Information(Upsampling)과 Finer-grained Information(Earlier Feature Map)을 얻을 수 있다.

<br>

6. 다시 몇 개의 Conv Layer을 추가하고, 비슷하지만 크기가 2배인 Tensor를 예측하고, 5번을 수행한다. - `2nd Scale`
7. 6번의 과정(3-5번)을 한 번 더 반복하여 마지막 Scale에서 Tensor를 예측한다. - `3rd Scale`

<br>

8. YOLOv2에 이어 Anchor Box를 생성하기 위해 K-means Clustering을 사용한다.

    3개의 Scale에서 3개씩 Box를 생성하기 때문에, 9개의 Cluster가 필요하다.

    (10×13), (16×30), (33×23), (30×61), (62×45), (59×119), (116 × 90), (156 × 198), (373 × 326).

```
More Detail

- 416 X 416 이미지를 입력으로 52X52, 26X26, 13X13에서 Feature Map을 추출한다.
- FCN의 Output Channel이 512가 될 때, 추출하여 Upsampling 한다.
- 각 Scale의 Feature Map의 Output Channel이 255가 되도록 1X1의 Channel을 조정한다.
```

#### 번외. Fine-grained Information.
1. Coarse-grained는 결이 거칠다. Information이라면 상세하지 않은 정보이고, Classification이라면 큰 범주의 분류를 생각하면 될 것 같다.
2. Fine-grained는 결이 곱다. Information이라면 상세한 정보이고, Classification이라면 개의 품종을 분류하는 등의 상세한 분류를 생각하면 될 것 같다.

[딥러닝 용어 정리, Coarse-grained classification과 Fine-grained classification의 차이와 이해](https://light-tree.tistory.com/215)

### 2.4. Feature Extractor.

![YOLOv3 2](https://user-images.githubusercontent.com/66259854/133923128-dd0060a5-1f8c-43f5-a1a5-6a6b32823fdd.png)

Backbone으로 Darknet-53을 사용한다.

Darknet-53은 53개의 Conv Layer로, ResNet에 사용했던 Shortcut Connections을 추가했다.

테스트에 사용한 이미지는 256 X 256에 Single Crop Accuracy이다.

![YOLOv3 3](https://user-images.githubusercontent.com/66259854/133923129-f5007eae-a001-4099-a4ef-e7807e0e2a1b.png)

ImageNet의 결과로 봤을 때, Top-5에서 ResNet-152와 같은 정확도를 보였으며, Darknet-19만큼은 아니지만 여전히 빠르다.

### 2.5. Training.
- Full Image를 사용한다.
- Multi-scale Training, Data Augmentation, Batch Normalization, Standard Stuff를 사용한다.
- Darknet Neural Network Framework를 사용한다.

## 3. How We Do.

![YOLOv3 4](https://user-images.githubusercontent.com/66259854/133923134-90db798a-9aba-412f-b855-b6f7a00ce3b9.png)

- YOLOv3 608 X 608은 RetinaNet 보다 약 3.8배 빠르다.
- $AP_{50}$에서 정확도가 높고, 빠르기 때문에 좋은 성능이 나온다.
- $AP_S$에서 약한 모습이 보인다.

---

![YOLOv3 5](https://user-images.githubusercontent.com/66259854/133923154-8e82b3d1-ff1d-4607-bca5-232d77f362ab.png)

- 0.5 IoU에서 성능을 비교한 것이다.
- inference time 그래프를 벗어난 높은 정확도의 YOLOv3의 모습이 인상적이다.
- YOLOv2의 데이터 로딩 버그를 고쳤고, 2mAP 정도 도와주었다고 한다.

## 4. Things We Tried That Didn't Work.

1. Anchor box x, y offset predictions.

    Linear Activation을 사용하여 Box의 w, h의 배수로 x, y Offset을 예측하는, 일반적인 Anchor Box 예측을 시도했으나 Model의 안정성을 떨어뜨린다.

2. Linear x, y predictions instead of logistic.

    Linear Activation으로 x, y Offset을 예측하려고 했지만, mAP 성능이 떨어졌다.

3. Focal loss.

    Focal Loss를 사용해봤지만 2 Point의 mAP 성능이 떨어졌다.

    YOLOv3는 자체적으로 Focal Loss 문제를 해결하려고 하는데, Objectness Predictions과 Conditional Class Prediction이 분리되어 있기 때문이다.

    따라서 Class Prediction에서 Loss가 생기지 않을 것이라 예측하지만, 확신할 수 없다.

4. Dual IOU thresholds and truth assignment.

    Faster R-CNN은 학습 중 2가지 IoU Threshold를 사용한다.

    [0.3 - 0.7]이 기준일 때, GT와 IoU 값이 0.7이라면 Positive, 0.3 미만이라면 Negative가 된다.

    논문에서 설정한 기준이 이미 Local Optima에 근접하게 보인다.

## 5. What This All Means(Conclusion).

> YOLOv3 is a good detector. It’s fast, it’s accurate. It’s
not as great on the COCO average AP between .5 and .95
IOU metric. But it’s very good on the old detection metric
of .5 IOU.

> Why did we switch metrics anyway? The original
COCO paper just has this cryptic sentence: “A full discus-
sion of evaluation metrics will be added once the evaluation
server is complete”. Russakovsky et al report that that hu-
mans have a hard time distinguishing an IOU of .3 from .5!
“Training humans to visually inspect a bounding box with
IOU of 0.3 and distinguish it from one with IOU 0.5 is surprisingly difficult.” [18] If humans have a hard time telling
the difference, how much does it matter?

> But maybe a better question is: “What are we going to
do with these detectors now that we have them?” A lot of
the people doing this research are at Google and Facebook.
I guess at least we know the technology is in good hands
and definitely won’t be used to harvest your personal infor-
mation and sell it to.... wait, you’re saying that’s exactly
what it will be used for?? Oh.

> Well the other people heavily funding vision research are
the military and they’ve never done anything horrible like
killing lots of people with new technology oh wait.....

> I have a lot of hope that most of the people using com-
puter vision are just doing happy, good stuff with it, like
counting the number of zebras in a national park [13], or
tracking their cat as it wanders around their house [19]. But
computer vision is already being put to questionable use and
as researchers we have a responsibility to at least consider
the harm our work might be doing and think of ways to mit-
igate it. We owe the world that much.

> In closing, do not @ me. (Because I finally quit Twitter).

## Rebuttal.

1. Reviewer #2.

    <img width="928" alt="YOLOv3 6" src="https://user-images.githubusercontent.com/66259854/133923155-e40f07a7-d69e-4e92-8404-3f0bd243963d.png">

2. Reviewer #4.

    COCO Metrics.

    <img width="452" alt="YOLOv3 7" src="https://user-images.githubusercontent.com/66259854/133923156-02bbedfd-2005-4af8-908d-e4b386ea47b3.png">

## Link.

[One-stage object detection](https://machinethink.net/blog/object-detection/)

[Dive Really Deep into YOLO v3: A Beginner's Guide](https://towardsdatascience.com/dive-really-deep-into-yolo-v3-a-beginners-guide-9e3d2666280e)

[[Deeplearning] Yolov3: An Incremental Improvement](https://dhhwang89.tistory.com/138)