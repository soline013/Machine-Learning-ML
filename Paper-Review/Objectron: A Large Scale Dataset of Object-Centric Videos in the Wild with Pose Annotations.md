# Objectron: A Large Scale Dataset of Object-Centric Videos in the Wild with Pose Annotations.

[Objectron: A Large Scale Dataset of Object-Centric Videos in the Wild with Pose Annotations](https://arxiv.org/pdf/2012.09988v1.pdf)

## Abstract.

<img width="924" alt="Objectron1" src="https://user-images.githubusercontent.com/66259854/117652533-3f669980-b1ce-11eb-818d-3b90b2b3d5d0.png">

2020 TensorFlow Developer Summit 에서 Google이 공개하였다.

## 1. Introduction.

<img width="1040" alt="Objectron2" src="https://user-images.githubusercontent.com/66259854/117652568-4e4d4c00-b1ce-11eb-9b1d-fc3f0b865c0d.png">

- Object Detection에서 대부분의 연구는 2D에 중점을 두고 있다.
- 3D Dataset은 2D Dataset에 비해 크지 않다.
- 다른 각도, 짧은 객체 중심의 비디오 클립, 더 큰 객체 세트를 갖는 Objectron Dataset을 제시한다.
- 각 비디오 클립에는 Camera Poses, Sparse Point-clouds, Surface Planes를 포함한 AR Session Metadata가 있다.
- 데이터에는 객체의 위치, 방향, 치수를 설명하는 Manually Annotated 3D Bounding Box가 있다.
- 14,819개의 비디오 클립과 4M개의 이미지로 구성된다.

### 번외. Pose.

방향을 설명하는 데 사용되기도 하지만, 위치와 방향의 조합을 포즈라고 한다.

우리가 흔히 인식하는 포즈와 유사한 개념인 것 같다.

[Pose (computer vision) - Wikipedia](https://en.wikipedia.org/wiki/Pose_(computer_vision))

### 번외. Point-cloud.

Point-cloud, 점구름은 어떤 좌표계에 속한 점들의 집합으로, 보통 3차원 공간을 대상으로 한다.

Lidar 센서나 RGB-D 센서 등에 의해 수집되는 데이터는 점구름으로 나타난다.

![Point-cloud](https://user-images.githubusercontent.com/66259854/117652653-61f8b280-b1ce-11eb-86d0-bd54bc833c29.gif)

[점구름 - 위키백과, 우리 모두의 백과사전](https://ko.wikipedia.org/wiki/%EC%A0%90%EA%B5%AC%EB%A6%84)

[Tistory](https://23min.tistory.com/8)

## 2. Previous Work.

`추가 예정`

## 3. Data Collection and Annotation.

## Object Categories.
1. 목표는 의미 있는 객체 카테고리를 고르는 것이다.
2. 또한, 객체를 공통된 환경, Store, Indoor, Outdoor에서 상대적으로 캡처한다.
3. 객체의 크기는 컵, 의자, 자전거 등으로 다양하다.
4. 객체 카테고리는 Rigid Objects와 Non-rigid Objects를 포함한다. CAD Model을 사용하는 기술이 예상된다. 자전거, 노트북 등은 쉽게 모양이 바뀌기 때문에 언급한 것으로 보인다.
5. 많은 3D Object Detection Models는 대칭 객체의 회전을 추정하는 것이 어렵다. 1도, 2도, 3도 정도의 회전도 애매하기 때문에, 테스트를 위해 컵이나 병 카테고리를 추가하였다.
6. Vision Model이 이미지 속의 텍스트에 유의하는 것으로 보인다. 따라서 책, 시리얼 박스와 같은 분명한 텍스트의 카테고리를 추가하였다.
7. 데이터가 다양한 나라에서 수집되어 조금씩 차이가 발생하기 때문에, Baseline 실험이 정확한 추정에 어려움을 겪는다.
8. 실시간 인식을 위해 신발, 의자와 같은 카테고리를 추가하였다.

## Data Collection.

<img width="499" alt="Objectron3" src="https://user-images.githubusercontent.com/66259854/117652574-4f7e7900-b1ce-11eb-9346-f72141369d15.png">

1. 데이터 수집은 카메라가 물체 주위에서 움직이며 다른 각도로 녹화한다.
2. 또한 AR Session(ARKit or ARCore)을 통해 Camera Poses, Sparse Point-clouds, Surface Planes을 캡처한다.
3. 모든 비디오는 1440 X 1920 크기로 High-end 핸드폰의 뒤 카메라를 사용하여 30fps로 녹화한다.
4. 데이터 수집에 핸드폰을 사용함으로써 10개 국가에서 데이터를 모을 수 있었다.

## Data Annotation.

<img width="503" alt="Objectron4" src="https://user-images.githubusercontent.com/66259854/117652576-50170f80-b1ce-11eb-8f01-437ff7c29d49.png">

1. 각 이미지에 3D Bounding Box를 다는 것은 시간이 오래 걸리고 비용이 크다.
2. 대신 비디오 클립의 3D 객체에 Annotation을 달아 모든 프레임에 채우고, Annotation Process를 확장한다.
3. Figure 4a처럼 3D World Map을 비디오의 이미지와 같이 Annotator에게 보여 준다.
4. Annotator는 3D World Map에서 3D Bounding Box를 그리고 Annotator Tool은 AR Sessions(ARKit or ARCore)에서 해당하는 모든 프레임에 3D Bounding Box를 투영한다.
5. Annotator는 위치, 방향, 치수를 조절하고 내용을 저장한다.

## Annotation Variance.

Annotation의 정확성은 두 가지 요인에 달려 있다.

1. 비디오 전체에서 추정된 카메라 포즈의 Drift(이동) 양.

    1. 경험적으로 Videos Drift < 2% 임을 관찰했다.
    2. Drift를 줄이기 위해, 보통 10초 미만의 시퀀스를 캡처한다.

    <img width="499" alt="Objectron5" src="https://user-images.githubusercontent.com/66259854/117652578-50afa600-b1ce-11eb-9bcd-6aa8e184c43b.png">

2. 3D Bounding Box를 나타내는 Rater(평가자)의 정확도.

    1. Rater의 정확도를 평가하기 위해 8명의 Annotator에게 동일한 시퀀스를 다시 처리하게 했다.
    2. 의자의 경우 의자 방향, 변형, 스케일의 표준 편차는 각각 4.6°, 1cm, 4cm이다.

    <img width="1039" alt="Objectron6" src="https://user-images.githubusercontent.com/66259854/117652580-50afa600-b1ce-11eb-9a00-bd217056ba5d.png">

## 4. Objectron Dataset.

<img width="1038" alt="Objectron7" src="https://user-images.githubusercontent.com/66259854/117652585-51483c80-b1ce-11eb-97cd-3d62b98a1f9f.png">

<img width="907" alt="Objectron8" src="https://user-images.githubusercontent.com/66259854/117652588-52796980-b1ce-11eb-822a-1c50bbeff49b.png">

- Objectron Dataset의 카테고리는 자전거, 책, 병, 카메라, 시리얼 상자, 의자, 컵, 노트북, 신발이다.
- 이 카테고리 중 일부는 Non-rigid하고 비디오 녹화 동안 정지된 상태를 유지한다.
- 비디오에서 카메라는 물체 주위를 움직이며 다른 각도에서 촬영한다.
- Dataset은 10개 국가에서 수집된다.
- 각 카테고리에는 Train Set과 Test Set이 있다.
- 좌표계는 +y 축이 위인 Left-hand 규칙을 따른다.
- 객체의 주석에는 Scale 뿐만 아니라 Rotation, Translation, w.r.t, Camera Center가 포함된다.
- Viewpoint Distribution(방위 분포)을 잘 이해하기 위해 각 객체의 방위각을 계산했다. 방위 0도는 물체가 앞을 보고 있다는 의미이다.
- Figure 7-top은 Viewpoint Distribution을 보여주고, Figure 7-bottom은 Elevation Distribution(고도 분포)을 보여준다.

## 5.  Baseline Experiments and Evaluations.

## 3D Intersection Over Union(IoU).

<img width="499" alt="Objectron9" src="https://user-images.githubusercontent.com/66259854/117652591-53120000-b1ce-11eb-99bb-3bd2f299293a.png">

1. Sutherland-Hodgman Algorithm.

    Sutherland-Hodgman Algorithm을 사용하여 두 Box의 면 사이의 교차점을 계산한다.

    x는 Predicted Box를 나타내고 y는 Annotation Label을 나타낸다.

    > To compute the intersecting points between the boxes x and y, first transform both boxes using the inverse transformation of the box x. The transformed box x will be axis-aligned and centered around the origin while the box y is brought to the coordinate system of the box x and remains oriented. Volume remains invariant under rigid-body transformation. We can compute the intersecting points in the new coordinate system and estimate the volume from the transformed intersection points. Using this coordinate system allows for more efficient and simpler polygon clipping against boxes since each surface is perpendicular to one of the coordinate axes.

    ### 번외. Sutherland-Hodgman Algorithm.

    Polygon Clipping 중 하나로, 다각형의 선분을 확장하여 다각형 내부만 존재하도록 하는 알고리즘이다.

    예시에서는 반시계 방향으로 수행한다.

    ![Sutherland-Hodgman_clipping_sample](https://user-images.githubusercontent.com/66259854/117652657-6329df80-b1ce-11eb-910e-4d1380556864.png)

    [Sutherland-Hodgman algorithm - Wikipedia](https://en.wikipedia.org/wiki/Sutherland%E2%80%93Hodgman_algorithm)

    [[클리핑] Sutherland-Hodgman 다각형 클리핑](https://playground10.tistory.com/80)

2. Convex Hull Algorithm.

    Clipping을 수행하기 위해서 Sutherland-Hodgman Algorithm를 사용한다.

    그리고 Convex Hull Algorithm을 사용하여 Intersection Volume을 계산한다.

    > We iterate over each edge in the polygon in clockwise order and determine whether that edge intersects with any faces in the axis-aligned box x. For each vertex in the box y, we check whether any of them are inside the box x. We add those vertices to the intersecting vertices as well. We repeat the whole process swapping the box x and y. We refer the readers to [9] for the details of the polygon clipping algorithm. Figure 8a shows an example of a polygon clipping.
    The volume of the intersection is computed by the convex hull of all the clipped polygons, as shown in Figure 8b. Finally, the IoU is computed from the volume of the intersection and volume of the union of two boxes. We are releasing the evaluation metrics source code along with the dataset.

    ### 번외. Convex Hull Algorithm.

    2차원 평면에 있는 여러 점 중 일부를 사용하여 모든 점을 포함하는 하나의 다각형을 만드는 알고리즘이다.

    이렇게 만들어진 것을 Convex Hull(볼록 껍질)이라고 한다.

    ![hull_algorithm](https://user-images.githubusercontent.com/66259854/117652633-5e652b80-b1ce-11eb-97ee-6cb275887014.gif)

    [컨벡스 헐 알고리즘(Convex Hull Algorithm)](https://www.crocus.co.kr/1288)

    [볼록 껍질 알고리즘 - 위키백과, 우리 모두의 백과사전](https://ko.wikipedia.org/wiki/%EB%B3%BC%EB%A1%9D_%EA%BB%8D%EC%A7%88_%EC%95%8C%EA%B3%A0%EB%A6%AC%EC%A6%98)

3. Figure 9.

    <img width="499" alt="Objectron10" src="https://user-images.githubusercontent.com/66259854/117652593-53aa9680-b1ce-11eb-93ea-e684ddd52159.png">

    병, 컵과 같은 대칭 객체의 경우 3D IoU가 제대로 정의되지 않는다.

    이 경우 Bouning Box를 대칭 축을 따라 회전시키고, 각 회전 결과를 평가하여 IoU를 최대화할 수 있는 Bouning Box를 선택한다.

## Baselines for 3D object detection.

## MobliePose.

![objectron_MobilePose_architecture](https://user-images.githubusercontent.com/66259854/117652651-61f8b280-b1ce-11eb-9916-8633ad5d18b5.png)

<img width="915" alt="Objectron11" src="https://user-images.githubusercontent.com/66259854/117652594-54432d00-b1ce-11eb-855b-ae7ff32db640.png">

1. 3D Bounding Box를 감지하기 위해 Dataset에 대해 SOTA Model을 훈련시켰다.

    MobliePose는 모바일 장치에서 실시간으로 작동하도록 설계된 경량 네트워크이다.

    Evaluation Code로 네트워크 출력을 평가하고, 3D IoU Average Precision, 2D Pixel Projection Error, Azimuth, Elevation 등 다양한 메트릭을 보고한다.

    Pre-training이나 Hyperparameter Optimization 없이 훈련했다.

    모델은 8개의 V100 GPU에서 100 Epoch 동안 훈련하였다.

2. MobileNetV2를 백엔드로 사용하고 네트워크에 두 개의 Head를 추가했다.
    1. 3D 경계 상자의 중앙 키포인트에서 Attention Mask를 생성하는 Attention Head.
    2. 중앙 키포인트에서 다른 8개의 키포인트의 x-y 조정을 예측하는 Regression Head.

    네트워크는 9개의 2D 예상 키포인트를 예측하고, 나중에 EPnP(Established Pose Estimation) 알고리즘을 사용하여 3D로 확장한다.

## Two-stage.

![objectron_2stage_network_architecture](https://user-images.githubusercontent.com/66259854/117652647-61601c00-b1ce-11eb-82c8-5184f99056ce.png)

<img width="915" alt="Objectron12" src="https://user-images.githubusercontent.com/66259854/117652597-54432d00-b1ce-11eb-9cb4-e6dc74fc0111.png">

1. 3D Object Detection을 위해 새로운 2단계 Architecture를 설계했다.

    첫 번째 단계는 SSD 같은 2D Object Detector를 사용하여 224 X 224 크기의 객체에서 2D Crop을 추정한다.

    두 번째 단계는 2D Crop을 사용하여 3D Bounding Box의 핵심 포인트로 회귀시키는 EfficientNet-Lite를 사용한다.

    입력 이미지를 7 X 7 X 1152 임베딩 벡터로 인코딩하고, 9개의 2D 키포인트를 회귀시키기 위해 Fully Connected Layer를 사용한다.

    EPnP를 사용하여 2D 예측 키포인트를 3D로 올린다.

    이 네트워크는 5.2MB이며 삼성 S20 모바일 GPU에서 83fps로 실행된다.

## Baselines for 3D object detection.

<img width="948" alt="Objectron13" src="https://user-images.githubusercontent.com/66259854/117652599-54dbc380-b1ce-11eb-8645-1263db12f8c1.png">

<img width="946" alt="Objectron14" src="https://user-images.githubusercontent.com/66259854/117652601-54dbc380-b1ce-11eb-8704-a2579193d332.png">

<img width="728" alt="Objectron15" src="https://user-images.githubusercontent.com/66259854/117652602-55745a00-b1ce-11eb-9e09-e8a786ff88dc.png">

Average Precision(평균 정밀도)를 위해 Detector가 상자의 중심을 사용하여 3D Bounding Box를 감지하고, 계속 다른 메트릭을 계산해야 한다.

실험 결과는 Dataset의 방위 분포는 45º에 치우쳐 있지만 고도 분포가 균일하기 때문에, 모델은 고도 추정에 더 정확하다.

### 번외. EPnP 알고리즘.

[EPnP: Efficient Perspective-n-Point Camera Pose Estimation](https://www.epfl.ch/labs/cvlab/software/multi-view-stereo/epnp/)

### 번외. MobileNetV2.

[MobileNetV2(모바일넷 v2), Inverted Residuals and Linear Bottlenecks](https://gaussian37.github.io/dl-concept-mobilenet_v2/)

[MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://arxiv.org/abs/1801.04381)

## 6. Details of the Objectron data format.

`추가 예정`

## 7. Conclusion.

> By releasing this dataset, we hope to enable the research community to push the limits of 3D object geometry understanding and foster new research and applications in 3D understanding, video models, object retrieval, view synthetics, and 3D reconstruction.

- Objectron Dataset.
- AR Session.
- MobilePose & Two-stage.

## Link.

[Home](https://google.github.io/mediapipe/)

[[Object Detction] 3D Object Detection, Google Objectron](https://eehoeskrap.tistory.com/435)

[Brunch](https://brunch.co.kr/@synabreu/64)

[Real-Time 3D Object Detection on Mobile Devices with MediaPipe](https://ai.googleblog.com/2020/03/real-time-3d-object-detection-on-mobile.html)

[google-research-datasets/Objectron](https://github.com/google-research-datasets/Objectron)
