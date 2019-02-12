import os
import time
from subprocess import Popen

def return_dirs(folder):
    return [dname for dname in os.listdir(folder)
        if os.path.isdir(os.path.join(folder, dname)) and not os.path.exists(os.path.join(savedir, dname))]

def chunkify(lst,n):
    return [lst[i::n] for i in range(n)]

#savedir = '/mv_users/peiguo/snapshots/dogs-fcn/'
#datadir = '/mv_users/peiguo/dataset/cud-keypoint/fcn/'
savedir = '/mv_users/peiguo/snapshots/nab-fcn/'
datadir = '/mv_users/peiguo/dataset/nab-keypoint/fcn/'
dnames = return_dirs(datadir)
ngpu = 2
dchunks = chunkify(dnames, ngpu)

for i in range(ngpu):
    cmdstr = ''
    dchunck = dchunks[i]
    print(i, dchunck)
    #continue
    for dch in dchunck:
        data_src = os.path.join(datadir, dch)
        save_dst = os.path.join(savedir, dch)
        cmdstr += 'CUDA_VISIBLE_DEVICES=%d th main.lua -retrain ../models/resnet-50.t7 -data %s -nClasses 555 -batchSize 16 -dispIter 64 -save %s; ' % (4+i, data_src, save_dst)
    #print(cmdstr)
    #print(dchunck)
    Popen(cmdstr, shell=True)
    time.sleep(5)
