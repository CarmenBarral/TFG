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

#### GenerarDatasets

Se generan datasets en csv con las características del centroide, area y perímetro de cada imagen. Se generan dos datasets con y sin ids. FALTA AÑADIR LOS PESOS

```console
import numpy as np
import cv2 as cv
import csv
import matplotlib.pyplot as plt
```
```console
from coco import COCO
import numpy as np
import matplotlib.pyplot as plt
ann_file='/content/drive/MyDrive/TFG/Datasets/Json_Files/traincoco.json'
coco=COCO(ann_file)
```

```console
img_ids=coco.getImgIds()
img_info=coco.loadImgs(img_ids)
Dataset={}
```
```console
nombrefiles=[]
for i in range(len(img_info)):
  nombrefiles.append(img_info[i]['file_name'])
```
```console
ListaNombres=[]
for i in range(len(nombrefiles)):
  ListaNombres.append('/content/drive/MyDrive/TFG/Datasets/Pictures/train/'+nombrefiles[i])
```
```console
for i in range(len(ListaNombres)):
  img = cv.imread(ListaNombres[i], cv.IMREAD_GRAYSCALE) # Consider changing ListImages[0] to ListaNombres[i] to iterate through all images
  assert img is not None, "file could not be read, check with os.path.exists()"
  ret,thresh = cv.threshold(img,127,255,0)
  contours,hierarchy = cv.findContours(thresh, 1, 2)

  cnt = contours[0]
  M = cv.moments(cnt)
  if M['m00'] == 0:
    cx=0
    cy=0
  else:
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])
  area = cv.contourArea(cnt)
  perimeter = cv.arcLength(cnt,False)

  # Create a nested dictionary for the current image if it doesn't exist
  if nombrefiles[i] not in Dataset:
    Dataset[nombrefiles[i]] = {}

  Dataset[nombrefiles[i]]['cx'] = cx
  Dataset[nombrefiles[i]]['cy'] = cy
  Dataset[nombrefiles[i]]['area'] = area
  Dataset[nombrefiles[i]]['perimeter'] = perimeter
```
```console
columnasNOIDNOPeso=['cx', 'cy', 'area', 'perimeter']
columnasIDNOPeso=['id', 'cx', 'cy', 'area', 'perimeter']
```
```console
ids=[]
for id in nombrefiles:
  ids.append(id[8:12])
```
```console
coursesNOIDNOPeso=[]
for i in range(len(Dataset)):
  dic={}
  dic['cx']=Dataset[nombrefiles[i]]['cx']
  dic['cy']=Dataset[nombrefiles[i]]['cy']
  dic['area']=Dataset[nombrefiles[i]]['area']
  dic['perimeter']=Dataset[nombrefiles[i]]['perimeter']
  coursesNOIDNOPeso.append(dic)
```
```console
coursesIDNOPeso=[]
for i  in range(len(Dataset)):
  dic={}
  dic['id']=ids[i]
  dic['cx']=Dataset[nombrefiles[i]]['cx']
  dic['cy']=Dataset[nombrefiles[i]]['cy']
  dic['area']=Dataset[nombrefiles[i]]['area']
  dic['perimeter']=Dataset[nombrefiles[i]]['perimeter']
  coursesIDNOPeso.append(dic)
```
```console
with open('DatasetNOIDNOPeso.csv', 'w', encoding='UTF8', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=columnasNOIDNOPeso)
    writer.writeheader()
    for courseNOIDNOPeso in coursesNOIDNOPeso:
        writer.writerow(courseNOIDNOPeso)
```
```console
with open('DatasetIDNOPeso.csv', 'w', encoding='UTF8', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=columnasIDNOPeso)
    writer.writeheader()
    for courseIDNOPeso in coursesIDNOPeso:
        writer.writerow(courseIDNOPeso)
```


#### ExtraerCaracterísticas

Se extraen con opencv las carácterísticas de cada imagen

```console
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
```

```console
from coco import COCO
import numpy as np
import matplotlib.pyplot as plt
ann_file='/content/drive/MyDrive/TFG/Datasets/Json_Files/traincoco.json'
coco=COCO(ann_file)
```
```console
img_ids=coco.getImgIds()
img_info=coco.loadImgs(img_ids)
```
```console
nombrefiles=[]
for i in range(len(img_info)):
  nombrefiles.append(img_info[i]['file_name'])
```
```console
ListaNombres=[]
for i in range(len(nombrefiles)):
  ListaNombres.append('/content/drive/MyDrive/TFG/Datasets/Pictures/train/'+nombrefiles[i])
```
```console
Caracteristicas={}
```
```console
for i in range(len(ListaNombres)):
  img = cv.imread(ListaNombres[i], cv.IMREAD_GRAYSCALE) # Consider changing ListImages[0] to ListaNombres[i] to iterate through all images
  assert img is not None, "file could not be read, check with os.path.exists()"
  ret,thresh = cv.threshold(img,127,255,0)
  contours,hierarchy = cv.findContours(thresh, 1, 2)

  cnt = contours[0]
  M = cv.moments(cnt)
  if M['m00'] == 0:
    cx=0
    cy=0
  else:
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])
  area = cv.contourArea(cnt)
  perimeter = cv.arcLength(cnt,False)

  # Create a nested dictionary for the current image if it doesn't exist
  if nombrefiles[i] not in Caracteristicas:
    Caracteristicas[nombrefiles[i]] = {}

  Caracteristicas[nombrefiles[i]]['cx'] = cx
  Caracteristicas[nombrefiles[i]]['cy'] = cy
  Caracteristicas[nombrefiles[i]]['area'] = area
  Caracteristicas[nombrefiles[i]]['perimeter'] = perimeter
```
```console
print(Caracteristicas)
```

#### FeatureEngineering
Se vuelven a generar dos datasets en csv con y sin id combinando las caracaterísticas entre si: cx*cy, area/perimetro, cx*area, cy*area


```console
import numpy as np
import cv2 as cv
import csv
import matplotlib.pyplot as plt
```
```console
from coco import COCO
import numpy as np
import matplotlib.pyplot as plt
ann_file='/content/drive/MyDrive/TFG/Datasets/Json_Files/traincoco.json'
coco=COCO(ann_file)
```

```console
img_ids=coco.getImgIds()
img_info=coco.loadImgs(img_ids)
DatasetFE={}
```
```console
nombrefiles=[]
for i in range(len(img_info)):
  nombrefiles.append(img_info[i]['file_name'])
```
```console
ListaNombres=[]
for i in range(len(nombrefiles)):
  ListaNombres.append('/content/drive/MyDrive/TFG/Datasets/Pictures/train/'+nombrefiles[i])
```
```console
for i in range(len(ListaNombres)):
  img = cv.imread(ListaNombres[i], cv.IMREAD_GRAYSCALE) # Consider changing ListImages[0] to ListaNombres[i] to iterate through all images
  assert img is not None, "file could not be read, check with os.path.exists()"
  ret,thresh = cv.threshold(img,127,255,0)
  contours,hierarchy = cv.findContours(thresh, 1, 2)

  cnt = contours[0]
  M = cv.moments(cnt)
  if M['m00'] == 0:
    cx=0
    cy=0
  else:
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])
  area = cv.contourArea(cnt)
  perimeter = cv.arcLength(cnt,False)

  # Create a nested dictionary for the current image if it doesn't exist
  if nombrefiles[i] not in DatasetFE:
    DatasetFE[nombrefiles[i]] = {}

  DatasetFE[nombrefiles[i]]['cx'] = cx
  DatasetFE[nombrefiles[i]]['cy'] = cy
  DatasetFE[nombrefiles[i]]['area'] = area
  DatasetFE[nombrefiles[i]]['perimeter'] = perimeter
```
```console
columnasFENOIDNOPeso=['cx', 'cy', 'area', 'perimeter','cxcy','area/perimetro','cxarea','cyarea']
columnasFEIDNOPeso=['id', 'cx', 'cy', 'area', 'perimeter','cxcy','area/perimetro','cxarea','cyarea']
```
```console
ids=[]
for id in nombrefiles:
  ids.append(id[8:12])
```
```console
coursesFENOIDNOPeso=[]
for i in range(len(DatasetFE)):
  dic={}
  dic['cx']=DatasetFE[nombrefiles[i]]['cx']
  dic['cy']=DatasetFE[nombrefiles[i]]['cy']
  dic['area']=DatasetFE[nombrefiles[i]]['area']
  dic['perimeter']=DatasetFE[nombrefiles[i]]['perimeter']
  dic['cxcy'] = DatasetFE[nombrefiles[i]]['cx']*DatasetFE[nombrefiles[i]]['cy']
  if DatasetFE[nombrefiles[i]]['perimeter'] == 0:
    dic['area/perimetro'] = 0
  else:
    dic['area/perimetro'] = DatasetFE[nombrefiles[i]]['area']/DatasetFE[nombrefiles[i]]['perimeter']
  dic['cxarea'] = DatasetFE[nombrefiles[i]]['cx']*DatasetFE[nombrefiles[i]]['area']
  dic['cyarea'] = DatasetFE[nombrefiles[i]]['cy']*DatasetFE[nombrefiles[i]]['area']
  coursesFENOIDNOPeso.append(dic)
```
```console
coursesFEIDNOPeso=[]
for i  in range(len(DatasetFE)):
  dic={}
  dic['id']=ids[i]
  dic['cx']=DatasetFE[nombrefiles[i]]['cx']
  dic['cy']=DatasetFE[nombrefiles[i]]['cy']
  dic['area']=DatasetFE[nombrefiles[i]]['area']
  dic['perimeter']=DatasetFE[nombrefiles[i]]['perimeter']
  dic['cxcy'] = DatasetFE[nombrefiles[i]]['cx']*DatasetFE[nombrefiles[i]]['cy']
  if DatasetFE[nombrefiles[i]]['perimeter'] == 0:
    dic['area/perimetro'] = 0
  else:
    dic['area/perimetro'] = DatasetFE[nombrefiles[i]]['area']/DatasetFE[nombrefiles[i]]['perimeter']
  dic['cxarea'] = DatasetFE[nombrefiles[i]]['cx']*DatasetFE[nombrefiles[i]]['area']
  dic['cyarea'] = DatasetFE[nombrefiles[i]]['cy']*DatasetFE[nombrefiles[i]]['area']
  coursesFEIDNOPeso.append(dic)
```
```console
with open('DatasetFENOIDNOPeso.csv', 'w', encoding='UTF8', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=columnasFENOIDNOPeso)
    writer.writeheader()
    for courseFENOIDNOPeso in coursesFENOIDNOPeso:
        writer.writerow(courseFENOIDNOPeso)
```
```console
with open('DatasetFEIDNOPeso.csv', 'w', encoding='UTF8', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=columnasFEIDNOPeso)
    writer.writeheader()
    for courseFEIDNOPeso in coursesFEIDNOPeso:
        writer.writerow(courseFEIDNOPeso)
```

#### AlgoritmosTípicos

#### GenerarGráficas
