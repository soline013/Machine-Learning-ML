# OpenCV.

## Iamge Handling.
1. `cv2.imread(fileName, flag)`
    - Return
    
        numpy.ndarray
    
    - Flag

        IMREAD_COLOR = 1: 이미지를 Color로 읽는다. 투명한 값은 무시한다. Default 값이다.

        IMREAD_GRAYSCALE = 0: 이미지를 Grayscale로 읽는다.
    
        IMREAD_UNCHANGED = -1: 이미지를 Alpha Channel까지 포함하여 읽는다.

2. `cv2.imshow(title, image)`
3. `cv2_imshow(image)`
4. `cv2.imwrite(fileName, image)`
5. `cv2.waitKey(0)`
6. `cv2.destroyAllWindows()`