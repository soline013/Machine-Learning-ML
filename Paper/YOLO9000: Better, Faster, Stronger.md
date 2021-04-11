# YOLO9000: Better, Faster, Stronger

[YOLO9000: Better, Faster, Stronger](https://arxiv.org/pdf/1612.08242.pdf)

![YOLO9000%20Better,%20Faster,%20Stronger%20816596cb0d024569b2c301b25a4931b4/Computer_Vision_Timeline.png](YOLO9000%20Better,%20Faster,%20Stronger%20816596cb0d024569b2c301b25a4931b4/Computer_Vision_Timeline.png)

## Abstract.

- 🖼️ Figure 1.

    ![YOLO9000%20Better,%20Faster,%20Stronger%20816596cb0d024569b2c301b25a4931b4/YOLO9000_6.png](YOLO9000%20Better,%20Faster,%20Stronger%20816596cb0d024569b2c301b25a4931b4/YOLO9000_6.png)

- Real-time Object Detection System.

    Detect Over 9000 Object Categories.

- Standard Detection Tasks Like PASCAL VOC & COCO.

    Outperforming Faster R-CNN With ResNet & SSD.

- Jointly Train on Object Detection & Classification.

    COCO Detection Dataset & ImageNet Classification Dataset.

## Introduction.

1. Neural Networks의 등장으로, Object Detection은 빠르고 정확해졌다.

    그러나 Classification, Tagging 등과 비교했을 때, Dataset이 너무 작다는 문제가 있다.

2. Classification 수준에서 Detection을 하고 싶지만, Detection을 위한 Labeling은 어렵다.

    따라서 Classification Dataset을 활용하는 두 가지 방법이 제안되었다.

    1. Hierarchical View of Object Classification.

        Combine distinct datasets together(Classification Dataset, Detcetion Dataset).

    2. Joint Training Algorithm.

        Leverages labeled detection images to learn to precisely localize objects.

        Leverages classification images to increase its vocabulary and robustness.

3. YOLO를 개선한 YOLOv2를 제안하고, 위의 두 방법을 적용한 YOLO9000을 제안했다.

## Better.

- 기존 YOLO의 단점.
    - YOLO는 Fast R-CNN과 비교했을 때, 중요한 Localization Error가 발생한다.
    - Region Proposal 기반의 방식들과 비교했을 때, Recall이 낮다.
    - Localization Error와 Recall을 잡으면서, Classification Accuracy를 유지하는데 집중한다.

- Batch Normalization.

    Dropout을 제거하고 Batch Normalization을 추가하였고, mAP가 2% 증가했다.

    - 번외. mAP.

        Mutiple Object Detection 알고리즘에 대한 성능을 1개의 Scalar Value로 표현한 것이다.

        [[Deep Learning] mAP (Mean Average Precision) 정리](https://eehoeskrap.tistory.com/m/237)

- High Resolution Classifier.
    1. State-of-the-art Detection은 ImageNet 기반의 Classifier Pre-trained Network를 사용한다. 여기서 대부분의 입력은 256X256 보다 작다.
    2. 기존 YOLO는 224X224 크기의 Classifier Pre-trained Network를 사용하며, 448X448 크기로 증가시켜 사용했다.
    3. YOLOv2는 448X448 크기로 ImageNet에서 10 Epochs를 수행하였고, mAP가 4% 증가했다.

- Convolutional With Anchor Boxes.
    1. YOLO는 Bounding Boxes의 Coordinates를 FC Layer로 직접 예측한다.
    2. FC Layer를 삭제하고 Convolutional Layer로 바꾸었고, Anchor Box를 사용해서 Box의 Offset을 예측한다.
    3. Conv Layer 출력의 해상도를 높이기 위해 Pooling Layer 하나를 제거한다.
    4. 448X448 크기를 416X416으로 변경하여, 최종 Feature Map에서 Width, Height가 홀수가 된다. 보통 큰 이미지는 가운데 Cell이 있는데, 이때 홀수라면 가운데에 Single Cell 생성되어 더 좋은 성능이 난다.
    5. YOLOv2의 Downsample Factor는 32이고(2X2 Max Pooling$^5$), 최종 Feature Map의 크기는 13X13이다.
    6. 기존 YOLO는 Grid 기반으로 Class를 예측하였으나, YOLOv2는  Anchor Box 기반이다.
    7. 기존 YOLO가 입력 이미지 당 98개의 Box를 예측했다면, YOLv2는 Anchor Box로 천 개이상의 Box를 예측한다.
    8. mAP는 69.5에서 69.2로 감소했지만, Recall은 81%에서 88%로 증가했다.

- Dimension Clusters.

    ![YOLO9000%20Better,%20Faster,%20Stronger%20816596cb0d024569b2c301b25a4931b4/YOLO9000_2.png](YOLO9000%20Better,%20Faster,%20Stronger%20816596cb0d024569b2c301b25a4931b4/YOLO9000_2.png)

    ![YOLO9000%20Better,%20Faster,%20Stronger%20816596cb0d024569b2c301b25a4931b4/YOLO9000_3.png](YOLO9000%20Better,%20Faster,%20Stronger%20816596cb0d024569b2c301b25a4931b4/YOLO9000_3.png)

    1. Anchor Box를 사용하면서 2가지 문제가 생겼는데, 하나는 Box Dimension이 Hand-pick 된다는 것이다.

    2. 따라서 YOLO9000은 K-mean을 사용하는데, Euclidian Distance 대신 아래의 식을 사용한다.

        $$d(\text{box, centroid}) = 1 - \text{IOU(box, centroid)}$$

        > ... larger boxes generate more error than smaller boxes. ... we really want are priors that lead to good IOU scores, which is independent of the size of the box.

    3. K-mean은 Parameter $K$의 값이 중요한데, 논문에서는 $K=5$로 설정하였다.

        성능과 연산속도는 Trade-off 관계에 있어 적당한 값이 중요하고, 성능 증가에도 한계가 있다.

    4. Cluster SSE(Error Sum of Squares, Euclidian Distance), Cluster IOU(Intersection Over Union), Anchor Boxes(Hand-pick)를 비교했을 때, $K=5$인 Cluster IOU를 선택하였다.

    - 번외. IOU(Intersection Over Union).

        ![YOLO9000%20Better,%20Faster,%20Stronger%20816596cb0d024569b2c301b25a4931b4/IOU.png](YOLO9000%20Better,%20Faster,%20Stronger%20816596cb0d024569b2c301b25a4931b4/IOU.png)

        ![YOLO9000%20Better,%20Faster,%20Stronger%20816596cb0d024569b2c301b25a4931b4/IOU_2.png](YOLO9000%20Better,%20Faster,%20Stronger%20816596cb0d024569b2c301b25a4931b4/IOU_2.png)

        [IoU, Intersection over Union 개념을 이해하자](https://ballentain.tistory.com/12)

- Direct Location Prediction.

    ![YOLO9000%20Better,%20Faster,%20Stronger%20816596cb0d024569b2c301b25a4931b4/YOLO9000_4.png](YOLO9000%20Better,%20Faster,%20Stronger%20816596cb0d024569b2c301b25a4931b4/YOLO9000_4.png)

    1. Anchor Box를 사용하면서 생긴 두 번째 문제는 모델이 불안정하다는 것이다.

    2. 불안정성은 Box의 (x, y) 좌표를 예측하는 과정에서 일어나는데, 아래 식을 통해 계산할 수 있다.

        $$x = (t_x * w_a) - x_a \\ y = (t_y * h_a) - y_a$$

        $t_x$가 양수라면 오른쪽, 음수라면 왼쪽으로 움직이게 되는데, 제한이 없으므로 Random Initialization에서 안정적인 Offset 값까지 오랜 시간이 걸린다.

    3. YOLO9000은 Offset의 범위를 [0, 1]로 제한하고, Logistic Activation을 사용한다.

        $$b_x = \sigma(t_x) + c_x \\ b_y = \sigma(t_y) + c_y \\ b_w = p_we^{t_w} \\ b_h = p_he^{t_h} \\ Pr(\text{object}) * IOU(\text{b, object}) = \sigma(t_o)$$

        $t_x, \, t_y, \, t_w, \, t_h, \, t_o$ : Bounding Box의 요소.

        $(c_x, \, c_y)$ : Cell의 왼쪽 상단 Offset.

        $(p_w, \, p_h)$ : Anchor Box의 사전 Width, Height.

        Network는 각 Cell 마다 5개의 Bounding Box를 예측한다.

    4. Dimension Cluster와 Direct Location Prediction을 통해 5%의 성능 향상이 일어남.

- Fine-Grained Features.

    ![YOLO9000%20Better,%20Faster,%20Stronger%20816596cb0d024569b2c301b25a4931b4/YOLO9000_0.png](YOLO9000%20Better,%20Faster,%20Stronger%20816596cb0d024569b2c301b25a4931b4/YOLO9000_0.png)

    1. 기존 YOLO는 13X13 Feature Map으로, 큰 이미지를 검출하기에는 충분하지만 작은 이미지에는 불충분하다.
    2. 이전 Layer에서 26X26 Feature Map을 가져와 26X26X512 크기를 13X13X2048로 Rescale 한다.
    3. 기존 13X13 Feature Map과 Concat한 Passthrough Layer를 만든다.
    4. 여기서 Concat은 다른 채널에 Stack하는 것으로, ResNet의 Identity Mappings과 비슷하다.
    5. 1%의 성능 향상이 일어난다.

- Multi-Scale Training.

    ![YOLO9000%20Better,%20Faster,%20Stronger%20816596cb0d024569b2c301b25a4931b4/YOLO9000_5.png](YOLO9000%20Better,%20Faster,%20Stronger%20816596cb0d024569b2c301b25a4931b4/YOLO9000_5.png)

    ![YOLO9000%20Better,%20Faster,%20Stronger%20816596cb0d024569b2c301b25a4931b4/YOLO9000_6%201.png](YOLO9000%20Better,%20Faster,%20Stronger%20816596cb0d024569b2c301b25a4931b4/YOLO9000_6%201.png)

    1. YOLOv2는 FC Layer를 제거하여 여러 Size의 이미지를 학습할 수 있고, 실행에 옮겼다.
    2. {320, 352, ..., 608}처럼 32 Pixel 간격으로 10 Batch마다 입력 이미지의 크기를 바꾼다.
    3. 다양한 크기에 대해 강해지므로 Speed와 Accuracy 사이에서 쉽게 Trade-off를 전환할 수 있다.
    4. 288X288에서는 90 FPS로 Fast R-CNN 정도의 mAP를 갖고, 608X608에서는 VOC2007에서 78.6mAP를 갖는다.

- Further Experiments.

    ![YOLO9000%20Better,%20Faster,%20Stronger%20816596cb0d024569b2c301b25a4931b4/YOLO9000_8.png](YOLO9000%20Better,%20Faster,%20Stronger%20816596cb0d024569b2c301b25a4931b4/YOLO9000_8.png)

    ![YOLO9000%20Better,%20Faster,%20Stronger%20816596cb0d024569b2c301b25a4931b4/YOLO9000_9.png](YOLO9000%20Better,%20Faster,%20Stronger%20816596cb0d024569b2c301b25a4931b4/YOLO9000_9.png)

- A Summary of Results.

    ![YOLO9000%20Better,%20Faster,%20Stronger%20816596cb0d024569b2c301b25a4931b4/YOLO9000_7.png](YOLO9000%20Better,%20Faster,%20Stronger%20816596cb0d024569b2c301b25a4931b4/YOLO9000_7.png)

## Faster.

- 기존 YOLO.
    - 많은 Detection Frameworks는 VGG-16을 사용한다. 그러나 VGG-16은 224X244 크기의 경우 30.69 Billion의 Floating Point 계산이 필요하다.
    - YOLO는 GoogleNet 기반의 독자적인 Network를 만들어 더 빠르고 계산량을 8.52 Billion으로 줄였다.
    - 같은 224X224 크기에서 Accuracy는 88%로 VGG-16의 90%와 비교하면 큰 차이가 없다.

- Darknet-19.

    ![YOLO9000%20Better,%20Faster,%20Stronger%20816596cb0d024569b2c301b25a4931b4/YOLO9000_10.png](YOLO9000%20Better,%20Faster,%20Stronger%20816596cb0d024569b2c301b25a4931b4/YOLO9000_10.png)

    YOLOv2는 Darknet이라는 새로운 Model을 사용한다.

    1. VGG와 비슷하게 3X3 Filter를 사용한다.
    2. Network In Network를 따라 Global Average Pooling을 사용한다.
    3. 1X1 Convolution Layer를 사용한다.
    4. 19개의 Convolution Layer와 5개의 Pooling Layer를 사용한다.

    > Darknet-19 only requires 5.58 billion operations to process an image yet achieves 72.9% top-1 accuracy and 91.2% top-5 accuracy on ImageNet.

- Training for Classification.
    1. ImageNet 1000 Class Classification Dataset for 160 Epochs.
    2. SGD With Learning Rate of 0.1.
    3. Polynomial Rate Decay of 4.

        [Learning rate Schedules](https://kiranscaria.github.io/general/2019/08/16/learning-rate-schedules.html)

    4. Weight Decay of 0.005.
    5. Momemtem of 0.9.
    6. Standard Data Augmentation Tricks: Random crops, Rotations, Hue, Saturation, and Exposure Shifts.

        [CNNs in Practice](https://nmhkahn.github.io/CNN-Practice)

- Training for Detection.
    1. Adding on three 3X3 convolutional layers with 1024 filters each followed by a final 1X1 convolutional layer.
    2. For VOC we predict 5 boxes with 5 coordinates each and 20 classes per box so 125 filters. $5 \times (5 +20) =125$
    3. 위에서 이미 언급된 Passthrough Layer.
    4. Train the network for 160 epochs with a starting learning rate of $10^{-3}$, dividing it by 10 at 60 and 90 epochs.
    5. 위에서 이미 언급된 Weight Decay & Momentum.
    6. 위에서 이미 언급된 Data Augmentation.
    7. Use the same training strategy on COCO and VOC.

## Stronger.

- How to do Joint Training?
    - Detection Dataset의 Class는 "dog", "boat"처럼 일반적이다.
    - Classification Dataset의 Class는 "Norfolk terrier", "Yorkshire terrier", “Bedlington terrier”처럼 세부적이다.
    - Classification은 Softmax를 사용하는데, 각 Class가 독립이라는 가정이 있다. 그러나 Dataset을 합치면 독립이라는 가정이 무너진다.
    - 따라서 Dataset은 독립이 아니라고 가정하고, Multi-label Model을 사용한다.

- Hierarchical Classification.

    ![YOLO9000%20Better,%20Faster,%20Stronger%20816596cb0d024569b2c301b25a4931b4/YOLO9000_11.png](YOLO9000%20Better,%20Faster,%20Stronger%20816596cb0d024569b2c301b25a4931b4/YOLO9000_11.png)

    ImageNet의 Label은 WordNet Language Dataset으로부터 파생되었다.

    WordNet은 Tree가 아닌 Direct Graph인데, 언어를 계층적으로 표현하기 어렵기 때문이다.

    YOLO9000은 WordTree를 만들었는데, Softmax를 없애지 않고 WordTree 내에서 독립적인 부분은 Softmax를 사용하여 여러 개의 Softmax를 볼 수 있다.

- Dataset Combination With WordTree.

    ![YOLO9000%20Better,%20Faster,%20Stronger%20816596cb0d024569b2c301b25a4931b4/YOLO9000_12.png](YOLO9000%20Better,%20Faster,%20Stronger%20816596cb0d024569b2c301b25a4931b4/YOLO9000_12.png)

    YOLO9000의 WordTree는 COCO Dataset과 ImageNet Dataset을 모두 사용했다.

    WordTree 1K를 만들기 위해 중간에 Node를 추가하였고, Label이 1369개로 늘어났다.

    ---

    $$Pr(\text{Norfolk terrier}) = Pr(\text{Norfolk terrier | terrier}) \\ *Pr(\text{terrier | hunting dog}) \\ * ... * \\ Pr(\text{mammal | Pr(\text{animal})}) \\ *Pr(\text{animal | physical object})$$

    For classification purposes we assume that the the image contains an object: $Pr(\text{physical object}) = 1$.

    1. 특정 Node를 예측할 때는 조건부 확률을 사용한다.
    2. 학습에서는 실제(Ground Truth) Label부터 Loot까지 모든 상위 값을 업데이트한다.

- Joint Classification and Detection.

    ![YOLO9000%20Better,%20Faster,%20Stronger%20816596cb0d024569b2c301b25a4931b4/YOLO9000_13.png](YOLO9000%20Better,%20Faster,%20Stronger%20816596cb0d024569b2c301b25a4931b4/YOLO9000_13.png)

    1. Dataset.

        총 9418개의 Class를 가진 Dataset이 만들어졌다.

        그중 9000개의 Class는 ImageNet에 속하고 Classification Label만 붙어있기 때문에, COCO를 위해 학습하는 이미지의 비율을 4 : 1로 맞췄다.

    2. Training.

        해당 부분에서 Output Size 문제로 5개의 Bounding Box를 3개로 조정하였다.

    3. Back Prop.

        Back Prop 시 Image가 나온 Dataset에 따라 Loss 계산이 다르다.

        - COCO Detection Dataset: Entire Loss Funtion.
        - ImageNet Classificaion Dataset: Classification Loss Function.
            - 여러 Box 중 가장 높은 확률을 뽑아 Classification Loss를 계산한다.
            - Box와 Ground Truth의 IOU가 0.3 이상이면 Entire Loss를 계산한다.

    4. Validation.

        Validation Dataset은 ImageNet Detection Dataset을 사용했고, COCO와 44개의 Class만 겹쳐있었다.

        성능은 19.7mAP으로, Labeling 되지 않은 156개의 Class를 포함하면 16.0mAP이다.

## Conclusion.

> We introduce YOLOv2 and YOLO9000, real-time de- tection systems. YOLOv2 is state-of-the-art and faster than other detection systems across a variety of detection datasets. Furthermore, it can be run at a variety of image sizes to provide a smooth tradeoff between speed and accu- racy.

> YOLO9000 is a real-time framework for detection more than 9000 object categories by jointly optimizing detection and classification. We use WordTree to combine data from various sources and our joint optimization technique to train simultaneously on ImageNet and COCO. YOLO9000 is a strong step towards closing the dataset size gap between de- tection and classification.

> Many of our techniques generalize outside of object de- tection. Our WordTree representation of ImageNet offers a richer, more detailed output space for image classification. Dataset combination using hierarchical classification would be useful in the classification and segmentation domains. Training techniques like multi-scale training could provide benefit across a variety of visual tasks.

> For future work we hope to use similar techniques for weakly supervised image segmentation. We also plan to improve our detection results using more powerful match- ing strategies for assigning weak labels to classification data during training. Computer vision is blessed with an enor- mous amount of labelled data. We will continue looking for ways to bring different sources and structures of data together to make stronger models of the visual world.

## Link.

[YOLO: Real-Time Object Detection](https://pjreddie.com/yolo9000/)