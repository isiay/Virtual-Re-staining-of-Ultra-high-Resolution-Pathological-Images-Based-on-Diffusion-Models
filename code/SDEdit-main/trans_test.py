path="/SDEdit/epoch2_3_20"
target_path ="/SDEdit/epoch2_3_20_y"

import os
import cv2
names = os.listdir(path)
print(len(names))
i=0
j=0
for name in names:
    if name.split('.')[0][-1] == 'x':
        i+=1
        continue
    new_name = name
    new_path = os.path.join(target_path,new_name)

    iamge = cv2.imread(os.path.join(path, name))
    cv2.imwrite(new_path,iamge)
    j+=1
print(i)
print(j)
