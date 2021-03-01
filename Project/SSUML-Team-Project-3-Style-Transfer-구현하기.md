# SSUML Team Project3. - Style Transfer 구현하기.
**Style Transfer 구현하기.**

## Project Guidance.

[Assignment 3 with 프로젝트](https://www.notion.so/Assignment-3-with-ddf98e9578f04cc085dc8adcac5734c9)

## Paper.

[Notion | A Neural Algorithm of Artistic Style](https://www.notion.so/A-Neural-Algorithm-of-Artistic-Style-59083a0faed34444a2545d78e488c0a6)

[A Neural Algorithm of Artistic Style](https://arxiv.org/abs/1508.06576)

2015년 논문.

---

[Notion | Image Style Transfer Using Convolutional Neural Networks](https://www.notion.so/Image-Style-Transfer-Using-Convolutional-Neural-Networks-e38f3655565949caa22358fd387dc848)

[Image Style Transfer Using Convolutional Neural Networks](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf)


2016년 논문. 내용이 조금 더 많다.

---

### 두 논문의 차이점.
1. 2016년 논문은 Content Reconstructions에서 `conv'N'_2`를 사용한다.
2. 2016년 논문이 최적화 방식, 입력 이미지에 따른 차이, 레이어 깊이에 따른 차이, Discussion 등 더 많은 내용을 다룬다.

## Error & Trouble.

1. `torch.tensor` to `torch.as_tensor`

    [Copy Construct warning · Issue #467 · pytorch/text](https://github.com/pytorch/text/issues/467)

---

1. Loss 값이 순간적으로 튀긴다.

    덕분에 이런 이미지를 얻을 수 있다.

    ![Image_1614432527](https://user-images.githubusercontent.com/66259854/109471569-45484c00-7ab4-11eb-9e02-c6ac1de4f2bd.png)


## Result.

Framework는 Tensorflow와 Pytorch를 모두 사용하였고, Result에서는 Tensorflow의 결과를 이용하였다.

- 최적화 이전의 결과.

    ![Neural_Style_Transfer_tf](https://user-images.githubusercontent.com/66259854/109472526-96a50b00-7ab5-11eb-867f-599a2b5c2dc0.png)

- 최적화 이후의 결과.

    ![Neural_Style_Transfer_tf_2](https://user-images.githubusercontent.com/66259854/109472521-9442b100-7ab5-11eb-86e6-13487db1a306.png)


## Different Weight Rate.

    Setting.

    Framework = Pytorch
    Input Image = Content Image
    Content Weight = 1
    Epoch = 400

- Style Weight = 100.

    ![Neural_Style_Transfer_pytorch_Content_100](https://user-images.githubusercontent.com/66259854/109471579-48433c80-7ab4-11eb-978e-f63fa67aa62e.png)

- Style Weight = 10,000.

    ![Neural_Style_Transfer_pytorch_Content_10000](https://user-images.githubusercontent.com/66259854/109471581-49746980-7ab4-11eb-9c00-10d2f7df0946.png)

- Style Weight = 1,000,000.

    ![Neural_Style_Transfer_pytorch_Content](https://user-images.githubusercontent.com/66259854/109471583-4a0d0000-7ab4-11eb-95ce-2a146ee7edf8.png)

## Different Input Image.

    Setting.

    Framework = Pytorch
    Style Weight = 1,000,000
    Content Weight = 1
    Epoch = 400

- Content Image.

    ![Neural_Style_Transfer_pytorch_Content](https://user-images.githubusercontent.com/66259854/109471583-4a0d0000-7ab4-11eb-95ce-2a146ee7edf8.png)

- Style Image.

    ![Neural_Style_Transfer_pytorch_Style](https://user-images.githubusercontent.com/66259854/109471593-4b3e2d00-7ab4-11eb-980f-331ba956b96d.png)

- Noise Image.

    ![Neural_Style_Transfer_pytorch_Noise_2](https://user-images.githubusercontent.com/66259854/109471585-4aa59680-7ab4-11eb-847c-59c133dc92b2.png)

    ![Neural_Style_Transfer_pytorch_Noise](https://user-images.githubusercontent.com/66259854/109471591-4b3e2d00-7ab4-11eb-91bb-4ef6d75159dc.png)

### Project Notebook.

- Pytorch.

    [Google Colaboratory](https://colab.research.google.com/drive/1txPZfTw1jCpELF1L9EA6VEwiTu2pa_DF#scrollTo=-nDFn2Yr4ROd)

    [soline013/Machine-Learning-ML](https://github.com/soline013/Machine-Learning-ML/blob/master/Style-Transfer/Neural_Style_Transfer_pytorch.ipynb)

---

- Tensorflow.

    [Google Colaboratory](https://colab.research.google.com/drive/1NmpvPqndFaleIZ4G-uODj9QOYGGp8Svr)

    [soline013/Machine-Learning-ML](https://github.com/soline013/Machine-Learning-ML/blob/master/Style-Transfer/Neural_Style_Transfer_tf.ipynb)

---

- Keras.

    [Google Colaboratory](https://colab.research.google.com/drive/1X5nIxIi4dR_en_lgtH-zxofMj-hrcUQW)

    [soline013/Machine-Learning-ML](https://github.com/soline013/Machine-Learning-ML/blob/master/Style-Transfer/Neural_Style_Transfer_keras.ipynb)

## 링크.

[고흐의 그림을 따라그리는 Neural Network, A Neural Algorithm of Artistic Style (2015) - README](http://sanghyukchun.github.io/92/)

[CNN을 활용한 스타일 전송(Style Transfer) | 꼼꼼한 딥러닝 논문 리뷰와 코드 실습](https://www.youtube.com/watch?v=va3e2c4uKJk&fbclid=IwAR05YuKVXga_kOD-0W-YO42SCIUN7REu20YmQCoEaztrh9Is29o3ule_874)

[Neural Transfer Using PyTorch - PyTorch Tutorials 1.7.1 documentation](https://pytorch.org/tutorials/advanced/neural_style_tutorial.html)

[tf.keras를 사용한 Neural Style Transfer | TensorFlow Core](https://www.tensorflow.org/tutorials/generative/style_transfer?hl=ko)

[[딥러닝]Neural Style Transfer](https://ssungkang.tistory.com/entry/%EB%94%A5%EB%9F%AC%EB%8B%9DNeural-Style-Transfer)