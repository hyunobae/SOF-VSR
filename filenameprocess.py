import os
import shutil

pth = 'D:/VSRdataset/frame'

directory = os.listdir(pth)

print(directory)

for i in directory:  # hr 바꾸기

    subdir = os.listdir(pth + '/' + i)
    print(subdir)
    lrdir = os.listdir(pth + '/' + i + '/' + subdir[1])
    print(pth + '/' + i + '/' + subdir[1])
    print(lrdir)

    for j in range(len(lrdir)):  # hr directory
        origname = lrdir[j][:-4]
        num = origname[1:]

        print(origname)
        # fname = origname[:-4]
        # print(fname)

        os.rename(pth + '/' + i + '/' + subdir[1] + '/' + origname + '.png',
                    pth + '/' + i + '/' + subdir[1] + '/' + 'lr' + str(int(num)-1) + '.png')
        print(pth + '/' + i + '/' + subdir[1] + '/' + origname + '.png' + ' -> ' + pth + '/' + i + '/' + subdir[
            1] + '/' + 'lr' + str(num) + '.png')
