import os
import shutil

pth = 'C:/Users/배재현/Desktop/originalSOF/TIP/data/test/tvd'

directory = os.listdir(pth)

print(directory)

for i in directory:  # hr 바꾸기

    subdir = os.listdir(pth + '/' + i)
    print(subdir)
    lrdir = os.listdir(pth + '/' + i + '/' + subdir[0])
    print(pth + '/' + i + '/' + subdir[0])
    print(lrdir)

    for j in range(len(lrdir)):  # hr directory
        origname = lrdir[j][:-4]
        #num = origname[5:]
        #imagexx.png
        print(origname)
        #print(num)
        fname = origname[5:]
        print(fname)

        os.rename(pth + '/' + i + '/' + subdir[0] + '/' + origname + '.png',
                    pth + '/' + i + '/' + subdir[0] + '/' + 'hr' + str(int(fname)-1) + '.png')
        print(pth + '/' + i + '/' + subdir[0] + '/' + origname + '.png' + ' -> ' + pth + '/' + i + '/' + subdir[
            0] + '/' + 'hr' + str(int(fname)-1) + '.png')
