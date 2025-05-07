# Primera fase del proyecto

---

- [Datasets](#Datasets)
- [LeerMascaras](#LeerMascaras)
- [GenerarDatasets](#GenerarDatasets)
- [ExtraerCaracterísticas](#ExtraerCaracterísticas)
- [FeatureEngineering](#FeatureEngineering)
- [AlgoritmosTípicos](#AlgoritmosTípicos)
- [GenerarGráficas](#GenerarGráficas)

---

#### Datasets

Datasets abiertos:

- [coco.py](https://github.com/cocodataset/cocoapi/tree/master/PythonAPI/pycocotools)
- [Inferring Broiler Chicken Weight Dataset](https://www.kaggle.com/datasets/lucasheilbuthh/inferring-broiler-chicken-weight).

#### LeerMascaras

Se leen las máscaras de segmentación que están dentro de la Dataset anterior y se almacenan en una lista (M). 

```console
from coco import COCO
import numpy as np
import matplotlib.pyplot as plt
ann_file='traincoco.json'
coco=COCO(ann_file)
```
```console
img_ids=coco.getImgIds()

coco.createIndex()

ann_id=coco.getAnnIds()

anns=coco.loadAnns(ann_id)

img_info=coco.loadImgs([img_ids[0]])
```

```console
an3s= anns
for an3 in an3s:
a=int(an3['image_id'])
an3['image_id']=a
print(an3)
```
```console
an4s= an3s
for an4 in an4s:
seg=[]
seg.append(an4['segmentation'])
an4['segmentation']=seg
print(an4)
```
```console
M={}
for i in range(549):
 M[i]=np.zeros((img_info[0]['height'], img_info[0]['width']))
print(M)
```
```console
an5s=an4s
for an5 in an5s:
 mask = coco.annToMask(an5)
 M[an5['image_id']]+=mask
for i in range(548):
plt.imshow(M[i])
plt.show()
```
