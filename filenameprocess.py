import os
import shutil

pth = 'D:/CDVL/yuv/hevc/frame'

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
        print(origname)
        # fname = origname[:-4]
        # print(fname)

        os.rename(pth + '/' + i + '/' + subdir[0] + '/' + origname + '.png',
                    pth + '/' + i + '/' + subdir[0] + '/' + 'hr' + str(int(origname)-1) + '.png')
        print(pth + '/' + i + '/' + subdir[0] + '/' + origname + '.png' + ' -> ' + pth + '/' + i + '/' + subdir[
            0] + '/' + 'hr' + str(origname) + '.png')
