# Computer Vision Wiki.

Wiki는 간단한 이해와 복기를 돕습니다.

[Object Detection](#objectdetection)

[Computer Vision](#computervision)

## <div id="objectdetection">Object Detection</div>

- Sliding Window

- Region Proposal

- Window Shift

- Bounding Box

    객체의 위치를 표시하기 위해 사용하는, 객체를 둘러싼 직사각형 박스를 의미한다.

</br>

- Two Stage Method

    Two Stage Method는 Region Proposal 단계와 Detector 단계로 구분된다.

    Region Proposal 단계는 Localization을 수행한다.
    Detector 단계는 Multi-task 문제로, 추가적인 Localization을 위해 Regression을, Classification을 위해 SVM 등을 거친다.

    One Stage Method에 비해 정확도는 높지만 속도가 느리다.

    e.g. R-CNN 계열

</br>

- One Stage Method

    One Stage Method는 Single Network로 해결한다.

    Localization과 Classification이 동시에 이루어진다.

    Two Stage Method에 비해 정확도는 낮지만 속도가 빠르다.

    e.g. YOLO, SSD

## <div id="computervision">Computer Vision</div>

- MediaPipe

    Computer Vision 라이브러리이다.

    Cross-platform과 Live and Streaming Media ML Solution을 제공한다.

</br>

- OpenCV

    Computer Vision에서 많이 사용되는 라이브러리이다.