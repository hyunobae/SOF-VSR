import cv2
import os

pth = 'D:/decodedData/TVD/low/4to3'

directory = os.listdir(pth)

# print(directory)

for i in directory:
    subdir = os.listdir(pth + '/' + i)

    lrdir = os.listdir(pth + '/' + i + '/' + subdir[0])
    # print(lrdir)

    for j in range(len(lrdir)):
        img = pth + '/' + i + '/' + subdir[0] + '/' + lrdir[j]
        print(img)
        src = cv2.imread(img, cv2.IMREAD_COLOR)
        dst = cv2.resize(src, dsize=(0, 0), fx=0.25, fy=0.25, interpolation=cv2.INTER_CUBIC)

        fname = lrdir[j][:-4]
        fname = fname[2:]
        print(fname)
        name = pth + '/' + i + '/' + subdir[1] + '/lr' + fname + '.png'
        print(name)
        cv2.imwrite(name, dst)
