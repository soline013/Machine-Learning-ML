# Toy Project 2. - YOLOv1 구현하기.

YOLO v1을 구현하면서 생긴 에러를 정리하고, 신세 한탄하기.

2가지 코드를 시도했는데, 당연하게 발생한 에러로 피곤해져서 YOLO v1은 잠시 멈추고 다른 프로젝트 먼저 진행하기로 했다.

다음 프로젝트는 Swin Transformer를 구현해보려고 한다.

## #1

```shell
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
<ipython-input-30-2580e2106c98> in <module>()
----> 1 train_ds = VOCDataset(train_list, transforms=get_train_transforms())
      2 valid_ds = VOCDataset(valid_list, transforms=get_test_transforms())
      3 test_ds = VOCDataset(test_list, mode='test', transforms=get_test_transforms())
      4 
      5 #torch tensor를 batch size만큼 묶어줌

1 frames
/usr/local/lib/python3.7/dist-packages/albumentations/pytorch/transforms.py in __init__(self, num_classes, sigmoid, normalize)
     52     def __init__(self, num_classes=1, sigmoid=True, normalize=None):
     53         raise RuntimeError(
---> 54             "`ToTensor` is obsolete and it was removed from Albumentations. Please use `ToTensorV2` instead - "
     55             "https://albumentations.ai/docs/api_reference/pytorch/transforms/"
     56             "#albumentations.pytorch.transforms.ToTensorV2. "

RuntimeError: `ToTensor` is obsolete and it was removed from Albumentations. Please use `ToTensorV2` instead - https://albumentations.ai/docs/api_reference/pytorch/transforms/#albumentations.pytorch.transforms.ToTensorV2. 

If you need `ToTensor` downgrade Albumentations to version 0.5.2.
```

Albumentations 버전이 안 맞아서 생긴 에러.

버전을 0.5.2로 Downgrade 해도 되고, ToTensor 대신에 ToTensorV2를 사용해도 해결할 수 있다.

매번 이런 에러만 발생했으면 좋겠다 :/

---

```shell
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
<ipython-input-19-ce8b85296538> in <module>()
      6     t_loss = 0.0
      7     breaking=False
----> 8     for step, (image, target) in tk0:
      9         image, target = image.to(device), target.to(device)
     10         update_lr(optimizer, epoch, float(step) / float(bl - 1))

8 frames
/usr/local/lib/python3.7/dist-packages/albumentations/augmentations/bbox_utils.py in ensure_data_valid(self, data)
     35         if self.params.label_fields:
     36             if not all(i in data.keys() for i in self.params.label_fields):
---> 37                 raise ValueError("Your 'label_fields' are not valid - them must have same names as params in dict")
     38 
     39     def filter(self, data, rows, cols):

ValueError: Your 'label_fields' are not valid - them must have same names as params in dict
```

조금 더 뒤적이면 알 수 있을 것 같은데, 여전히 모르는 것 투성이인 말하는 감자 같다.

코드를 글자 단위로 분해해버릴까.

나도 웰노운 하고 싶다.

## #2
```shell
Traceback (most recent call last):
  File "train.py", line 156, in <module>
    main()
  File "train.py", line 135, in main
    train_loader, model, iou_threshold=0.5, threshold=0.4
  File "/content/drive/MyDrive/Colab Notebooks/utils.py", line 254, in get_bboxes
    for batch_idx, (x, labels) in enumerate(loader):
  File "/usr/local/lib/python3.7/dist-packages/torch/utils/data/dataloader.py", line 521, in __next__
    data = self._next_data()
  File "/usr/local/lib/python3.7/dist-packages/torch/utils/data/dataloader.py", line 1203, in _next_data
    return self._process_data(data)
  File "/usr/local/lib/python3.7/dist-packages/torch/utils/data/dataloader.py", line 1229, in _process_data
    data.reraise()
  File "/usr/local/lib/python3.7/dist-packages/torch/_utils.py", line 425, in reraise
    raise self.exc_type(msg)
NotImplementedError: Caught NotImplementedError in DataLoader worker process 0.
Original Traceback (most recent call last):
  File "/usr/local/lib/python3.7/dist-packages/torch/utils/data/_utils/worker.py", line 287, in _worker_loop
    data = fetcher.fetch(index)
  File "/usr/local/lib/python3.7/dist-packages/torch/utils/data/_utils/fetch.py", line 44, in fetch
    data = [self.dataset[idx] for idx in possibly_batched_index]
  File "/usr/local/lib/python3.7/dist-packages/torch/utils/data/_utils/fetch.py", line 44, in <listcomp>
    data = [self.dataset[idx] for idx in possibly_batched_index]
  File "/usr/local/lib/python3.7/dist-packages/torch/utils/data/dataset.py", line 34, in __getitem__
    raise NotImplementedError
NotImplementedError
```

이것 저것 건드리다가 데이터가 이상해져서 다른 오류까지 발생했다.

나아질 기미가 안 보인다..
