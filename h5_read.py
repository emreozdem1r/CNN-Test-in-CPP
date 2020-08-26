import numpy as np
import h5py


def AAgetWeightsForLayer(layerName, fileName):
    with h5py.File(fileName, mode='r') as f:
        for key in f:
            print(key, f[key])
            o = f[key]
            for key1 in o:
                print(key1, o[key1])
                r = o[key1]
                for key2 in r:
                    print(key2, r[key2])


def isGroup(obj):
    if isinstance(obj, h5py.Group):
        return True
    return False


def isDateset(obj):
    if isinstance(obj, h5py.Dateset):
        return True
    return False


def getDatasetFromGroup(datasets, obj):
    if isGroup(obj):
        for key in obj:
            x = obj[key]
            getDatasetFromGroup(datasets, x)
    else:
        datasets.append(obj)


def getWeightsForLayer(layerName, fileName):
    weights = []
    with h5py.File(fileName, mode='r') as f:
        for key in f:
            if layerName in key:
                obj = f[key]
                datasets = []
                getDatasetFromGroup(datasets, obj)

                for dataset in datasets:
                    w = np.array(dataset)
                    weights.append(w)
    return weights

weights = getWeightsForLayer("conv2d", 'weights.hdf5')
#if you want to take dense weights, you should write dense instead conv2d
#if you want to take conv3d weights, you should write conv3d instead conv2d
#also you should change function parameter bottom of the code

#Yapay sinir ağlarının ağırlıklarını almak için için conv3d yerine dense yazılmalıdır
print(np.shape(weights))

dosya = open('conv2d_model.txt','w')
for i in weights:
    if len(i.shape) == 1:
        for a in i:
            print(a)
            dosya.write(str(a))
            dosya.write('\n')
    if len(i.shape) == 2:
        for a in i:
            for b in a:
                print(b)
                dosya.write(str(b))
                dosya.write('\n')
    if len(i.shape) == 3:
        for a in i:
            for b in a:
                for c in b:
                    print(c)
                    dosya.write(str(c))
                    dosya.write('\n')
    if len(i.shape) == 4:
        for a in i:
            for b in a:
                for c in b:
                    for d in c:
                        print(str(d))
                        dosya.write(str(d))
                        dosya.write('\n')
    if len(i.shape) == 5:
        for a in i:
            for b in a:
                for c in b:
                    for d in c:
                        for e in d:
                            print(e)
                            dosya.write(str(e))
                            dosya.write('\n')

dosya.close()

#Yapay sinir ağlarının ağırlıklarını almak için için conv3d yerine dense yazılmalıdır
AAgetWeightsForLayer("conv2d", 'weights.hdf5')