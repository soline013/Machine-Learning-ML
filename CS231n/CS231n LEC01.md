# CS231n LEC01.
## Stanford University, Spring 2017.
**Introduction to Convolutional Neural Networks for Visual Recognition.**

## Intro.
  1. CISCO에서 2015~2017년 사이에 조사한 내용에 따르면, 인터넷 트래픽의 80%는 비디오 데이터이다.
  2. 이런 시각 데이터를 잘 활용하는 알고리즘이 중요하지만, 시각 데이터는 해석하기 까다로워 Dark Matter라고 부르곤 한다.
  3. YouTube에는 매초 5시간 분량의 비디오가 업로드 된다. 사람이 이 분량을 감당할 순 없으므로 자동으로 시각 데이터를 활용하는 알고리즘이 필요하다.
  4. Computer Vision 주변에는 많은 분야가 존재한다. e. g. 물리학, 생물학, 수학, etc.

## History of Vision.
  1. 생물학적 비전, 비전의 태동.
  2. 공학적 비전, 1600년 르네상스 시대의 Camera Obscura.
  3. 컴퓨터 비전.

## History of Computer Vision.
  1. Hubel & Wiesel, 1959.
  
     ![image](https://user-images.githubusercontent.com/66259854/95469859-0e658280-09bb-11eb-8175-69eb7ff4e0c4.png)
     
     포유류의 시각적 처리 메커니즘에 관심을 갖고, 인간과 비슷한 고양이의 뇌를 연구.
     
     고양이 두뇌 뒤, 일차 시각 피질이 있는 곳에 전극을 연결.
     
     Edges가 움직이면 반응하는 세포들이 있었고, 단순한 시각 처리 구조가 점점 복잡해진다.

* * *

  2. "Block world" Larry Roberts, 1963.
  
     ![image](https://user-images.githubusercontent.com/66259854/95469862-10c7dc80-09bb-11eb-9c59-412991ead64d.png)
     
     Computer Vision의 첫 박사 논문.
     
     눈에 보이는 사물을 기하학적 모양으로 단순화시켰다.

* * *

  3. “The Summer Vision Project” MIT, 1966.
  
     ![image](https://user-images.githubusercontent.com/66259854/95469869-132a3680-09bb-11eb-8e21-df259699c237.png)
     
     "시각 시스템의 전반을 구현하기 위해 프로젝트 참가자를 효율적으로 이용한다."

* * *

  4. “Vision” David Marr, 1970s.
     
     ![image](https://user-images.githubusercontent.com/66259854/95469881-145b6380-09bb-11eb-8d47-0d6429529268.png)

     우리가 눈으로 받아들인 Image를 최종적인 full 3D 표현으로 만들려면 몇 가지 과정을 거쳐야 한다.

     1)	Primal Sketch.
     
        Edges, Bars, Ends, Virtual Lines, Curves, Boundaries가 표현되는 과정.
        
        Hubel & Wiesel이 얘기한 Edges와 초기의 단순한 구조와 관련된다.
        
     2) 2.5-D Sketch.
     
        시각적 장면을 구성하는 표면, 깊이, Layer, 불연속성 등을 합치는 것.
        
        Surface and Volumetric Primitives 형태의 최종적인 3D 모델을 구현.
        
        이상적이고 직관적인 방식으로 해당 방식은 수십 년 지속되었다.

* * *

  5. “Generalized Cylinder & Pictorial Structure”

     ![image](https://user-images.githubusercontent.com/66259854/95469886-16bdbd80-09bb-11eb-90cb-db234f886669.png)
     
     모든 객체는 단순한 기하학적 형태로 표현할 수 있다.
     
     원통 모양을 조합해서 표현. / 주요 부위와 관절로 표현.
 
 * * *
 
  6. David Lowe, 1987.
 
     ![image](https://user-images.githubusercontent.com/66259854/95469895-19b8ae00-09bb-11eb-80d1-60b1a5c24390.png)
     
     단순한 구조로 실제 세계를 재구성, 인식하기 위한 노력이었다.
     
     면도기를 인식하기 위해 면도기를 선, 경계, 직선의 조합으로 구성했다.

* * *

  7. 이런 시도는 모두 대담하지만, Toy Example에 불과했다.

* * *

  8. “Normalized Cut” Shi & Malik, 1997. //Berkley University
  
     ![image](https://user-images.githubusercontent.com/66259854/95469899-1c1b0800-09bb-11eb-980c-71df5dec6eaa.png)

     객체 인식이 어렵다면 객체 분할(Segmentation)을 우선으로 함.
     
     객체 분할: 이미지의 각 픽셀을 유의미한 방향으로 군집화 하는 방법.
     
     배경인 픽셀과 사람이 속한 픽셀을 구분할 수 있었다. → Image Segment

* * *

  9. (Statistical) Machine Learning, 1999~2000s.
     1) SVM
     2)	Boosting
     3)	Graphical Models
     4)	First wave of Neural Networks

* * *

  10. “Face Detection” Viola & Jones, 2001. //Using Adaboost

      ![image](https://user-images.githubusercontent.com/66259854/95469916-1f15f880-09bb-11eb-9b62-4c42019c8573.png)

      2006년, Fujifilm의 실시간 얼굴인식 디지털카메라.
 
 * * *
 
  11. “SIFT Feature” David Lowe, 1999.
 
      ![image](https://user-images.githubusercontent.com/66259854/95469923-21785280-09bb-11eb-8d6d-cf549cf9fadc.png)

      1990~2010s, Feature Based Object Recognition.
      
      객체의 특징 중 일부는 변화에 강하고 불변하다.
      
      따라서 이런 특징을 가지고 다른 객체에 매칭한다.

* * *

  12. “Spatial Pyramid Matching” Lazebnik, Schmid & Ponce, 2006.
 
      ![image](https://user-images.githubusercontent.com/66259854/95469931-23daac80-09bb-11eb-95e1-2e2d214062b2.png)
      
      우리가 특징을 잘 뽑아낼 수 있다면, 특징은 단서를 제공한다.
      
      이미지의 여러 부분, 해상도에서 추출한 특징을 하나의 특징으로 표현하고, SVM을 사용.

* * *

  13. “Histogram of Gradients & Deformable Part Models”
 
      ![image](https://user-images.githubusercontent.com/66259854/95469940-263d0680-09bb-11eb-8152-a7461f5aaf9e.png)
      
      사람의 몸을 현실적으로 모델링 할 수 있는지에 대한 연구가 진행되었다.

* * *

  14. “PASCAL Visual Object Challenge” Everingham et al. 2006~2012.
      
      ![image](https://user-images.githubusercontent.com/66259854/95469945-289f6080-09bb-11eb-8d54-a1c9c631c516.png)

      Benchmark Dataset을 모으기 시작.
      
      Princeton와 Stanford 그룹의 질문. “우리는 세상의 모든 객체를 인식할 준비가 되었는가?”
      
      대부분 ML 알고리즘을 사용 → Overfitting & So Complex!
      
      복잡한 고차원 데이터로 인해 많은 parameter가 필요하였고,
      
      학습 데이터가 모자라면 Overfitting이 빠르게 발생하여 일반화가 떨어짐.
      
      두 가지의 Motivation: 세상의 모든 것을 인식, Overfitting의 해결.

* * *

  15. “ImageNet”
      
      ![image](https://user-images.githubusercontent.com/66259854/95469956-2b01ba80-09bb-11eb-9a3c-15a93ba35936.png)
      
      ![image](https://user-images.githubusercontent.com/66259854/95469964-2d641480-09bb-11eb-8411-4c16a928831a.png)

      수십억 장의 이미지를 다운받고, WordNet Dictionary로 정리.
      
      수천 개의 객체 클래스에 Clever Crowd Engineering Trick을 도입.
      
      Clever Crowd Engineering Trick: Amazon Mechanical Turk에서 사용하는 플랫폼.
      
      성능 측정을 위해 2009년 ILSVRC를 개최하였다.

      ![image](https://user-images.githubusercontent.com/66259854/95469994-39e86d00-09bb-11eb-9660-bb8b8e97abc9.png)

      2012년 CNN이 등장하면서 오류가 급격하게 낮아졌다.
      
      2012년에는 AlexNet, 2015년에는 ResNet이 등장.
 
## CS231n Overview.
  1. Image Classification에 대해 중점적으로 다룬다.
     
     Object Detection과 Image Captioning도 포함.

  2. ILSVRC의 우승자들.
     
     ![image](https://user-images.githubusercontent.com/66259854/95470015-3e148a80-09bb-11eb-9428-1b1badaebacc.png)

  3. 사실 CNN은 1998년부터 숫자 인식을 위한 목적으로 나왔던 모델이다.
     
     CPU, GPU 등 컴퓨터 성능의 발전과 Dataset의 양과 질 향상으로 주목받게 됨.
     
     ![image](https://user-images.githubusercontent.com/66259854/95470021-3fde4e00-09bb-11eb-9a25-f48ed5282814.png)

## 링크.
[Lecture 1 | Introduction to Convolutional Neural Networks for Visual Recognition |](https://www.youtube.com/watch?v=vT1JzLTH4G4&t=2s)

[CS231n: Convolutional Neural Networks for Visual Recognition](http://cs231n.stanford.edu/2017/)

[CS231n 2017 Lceture1 PDF](http://cs231n.stanford.edu/slides/2017/cs231n_2017_lecture1.pdf)

[soline013/CS231N_17_KOR_SUB](https://github.com/visionNoob/CS231N_17_KOR_SUB/blob/master/kor/Lecture%201%20%20%20Introduction%20to%20Convolutional%20Neural%20Networks%20for%20Visual%20Recognition.ko.srt)
