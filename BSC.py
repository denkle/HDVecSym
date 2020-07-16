import numpy as np

def rotate(HDvectors, rotateby=1):
    return np.roll(HDvectors, rotateby, axis=0)


def bind(HDvectors1, HDvectors2):
    if HDvectors1.shape == HDvectors2.shape:
        return np.logical_xor(HDvectors1, HDvectors2).astype(int)
    elif len(HDvectors1.shape) == 1 or len(HDvectors2.shape) == 1 or \
            HDvectors1.shape[1] == 1 or HDvectors2.shape[1] == 1:
        if len(HDvectors1.shape) == 1 or HDvectors1.shape[1] == 1:
            return np.logical_xor(HDvectors2, np.tile(HDvectors1, (1, HDvectors2.shape[1]))).astype(int)
        elif len(HDvectors2.shape) == 1 or HDvectors2.shape[1] == 1:
            return np.logical_xor(HDvectors1, np.tile(HDvectors2, (1, HDvectors1.shape[1]))).astype(int)
    else:
        raise Exception("Dimensions of arrays must agree or one of them should be a vector")

def unbind(HDvectors1, HDvectors2):
    if HDvectors1.shape == HDvectors2.shape:
        return np.logical_xor(HDvectors1, HDvectors2).astype(int)
    else:
        raise Exception("Dimensions of arrays must agree or one of them should be a vector")

def bundle(HDvectors, optype="majority", kappa=3, **kwargs):
    if len(HDvectors[0].shape) == 1:
        base = HDvectors[0] - 0.5
    else:
        base = np.sum(HDvectors[0], axis=1) - (HDvectors[0].shape[1] / 2)


    # Majority rule
    if optype == "majority":
        mask = base==0
        base[base < 0] = 0
        base[base > 0] = 1
        base[mask] = np.random.randint(0, 2, np.count_nonzero(mask))
        base = np.expand_dims(base, axis=1)  # to keep (N,1) shape
        return base
    else:
        raise Exception("The type is not defined")

def item(concepts, N=1000):
    memory = np.random.randint(low=0, high=2, size=(N, len(concepts)))
    return [memory, concepts]


def similarity(HDvectors1, HDvectors2, stype="Hamming"):
    dp = np.dot(np.transpose(HDvectors1), HDvectors2)
    if stype == "dot":
        return dp
    elif stype == "cosine":
        norms = (np.dot(np.expand_dims(np.linalg.norm(HDvectors1, axis=0), axis=1),
                        np.expand_dims(np.linalg.norm(HDvectors2, axis=0), axis=0)))
        dp.astype(float)
        return np.divide(dp, norms)
    elif stype == "Hamming":    
        Ham=np.zeros((HDvectors1.shape[1],HDvectors2.shape[1]),dtype='float64')
        for i in range(HDvectors2.shape[1]):
            hamming=np.logical_xor(HDvectors1, np.tile(np.expand_dims(HDvectors2[:,i], axis=1), (1, HDvectors1.shape[1]))).astype(int)        
            hamming_h=np.sum(hamming, axis=0)/hamming.shape[0]
            Ham[:,i]=hamming_h  
        return Ham
    else:
        raise Exception("The type is not defined")


def getitems(itemmem, concepts):
    HDvectors = np.zeros((itemmem[0].shape[0], len(concepts)), dtype=int)
    for i in range(len(concepts)):
        con = concepts[i]
        ind = itemmem[1].index(con)
        if ind != -1:
            HDvectors[:, i] = itemmem[0][:, ind]
        else:
            raise Exception("The concept is not present in the item memory")
    return [HDvectors, concepts]


def probe(itemmem, query, searchtype="nearest", simtype="Hamming"):
    scores = similarity(itemmem[0], query, stype=simtype)

    if searchtype == "nearest":
        ind = np.argmin(scores)
        return [np.expand_dims(itemmem[0][:, ind], axis=1), itemmem[1][ind]]
    else:
        raise Exception("The specified search type is not defined")