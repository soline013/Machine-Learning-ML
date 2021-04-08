# YOLO9000: Better, Faster, Stronger.
![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/032457de-83d4-4ebf-a8fa-76dfb6e1434b/Computer_Vision_Timeline.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/032457de-83d4-4ebf-a8fa-76dfb6e1434b/Computer_Vision_Timeline.png)

## Abstract.

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

1. Batch Normalization.
2. High Resolution Classifier.
3. Convolutional With Anchor Boxes.
4. Dimension Clusters.
5. Direct Location Prediction.
6. Fine-Grained Features.
7. Multi-Scale Training.
8. Further Experiments.

## Faster.

1. Darknet-19.
2. Training for Classification.
3. Training for Detection.

## Stronger.

1. Hierarchical  Classification.
2. Dataset Combination With WordTree.
3. Joint Classification and Detection.

## Conclusion.

## Link.

[YOLO: Real-Time Object Detection](https://pjreddie.com/yolo9000/)