# OpenCV.

1. `cv2.imread(fileName, flag)`
    - Parameters

        fileName: str, 이미지의 경로.

        flag: int, 이미지를 읽는 Option.

    - Return

        numpy.ndarray

    - Flag

        IMREAD_COLOR = 1: 이미지를 Color로 읽는다. 투명한 값은 무시한다. Default 값이다.

        IMREAD_GRAYSCALE = 0: 이미지를 Grayscale로 읽는다.

        IMREAD_UNCHANGED = -1: 이미지를 Alpha Channel까지 포함하여 읽는다.

2. `cv2.imshow(title, image)`
    - Parameters

        title: str, 이미지의 제목.

        image: numpy.ndarray.

3. `cv2_imshow(image)`

    Google Colab에서는 `cv2.imshow` 대신 `cv2_imshow`를 사용한다.

    - Parameters

        image: numpy.ndarray.

4. `cv2.imwrite(fileName, image)`
    - Parameters.

        fileName: str, 저장될 파일명.

        image: numpy.ndarray, 저장할 이미지.

5. `cv2.waitKey(0)`

    키보드 입력을 대기하는 함수, 0이면 입력할 때까지 대기한다.

6. `cv2.destroyAllWindows()`

    화면에 나타난 윈도우를 종료한다.