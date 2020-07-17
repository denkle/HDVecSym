import numpy as np
import math
from scipy.fftpack import fftn, ifftn

def power(HDvector, power):
    return ifftn(fftn(HDvector, shape=HDvector.shape)**power).real

def rotate(HDvectors, rotateby=1):
    return np.roll(HDvectors, rotateby, axis=0)

def involution(HDvector):
    if len(HDvector.shape) == 1:
        return np.append(HDvector[0], HDvector[1:][::-1])
    else:
        return np.concatenate(([HDvector[0]], HDvector[1:][::-1]), axis=0)

def fftconvolve(HDvectors1, HDvectors2):
    return ifftn(fftn(HDvectors1, shape=HDvectors1.shape) * fftn(HDvectors2, shape=HDvectors2.shape)).real

def fftcorrelate(HDvectors1, HDvectors2):
    return fftconvolve(involution(HDvectors1), HDvectors2)

def bind(HDvectors1, HDvectors2):
    if HDvectors1.shape == HDvectors2.shape:
        return fftconvolve(HDvectors1, HDvectors2)
    elif len(HDvectors1.shape) == 1:
        return fftconvolve(HDvectors1[np.newaxis].reshape(HDvectors2.shape), HDvectors2)
    elif len(HDvectors2.shape) == 1:
        return fftconvolve(HDvectors1, HDvectors2[np.newaxis].reshape(HDvectors1.shape))
    else:
        raise Exception("Dimensions of arrays must agree or one of them should be a vector")

def unbind(HDvectors1, HDvectors2):
    if HDvectors1.shape == HDvectors2.shape:
        return fftcorrelate(HDvectors1, HDvectors2)
    elif len(HDvectors1.shape) == 1:
        return fftcorrelate(HDvectors1[np.newaxis].reshape(HDvectors2.shape), HDvectors2)
    elif len(HDvectors2.shape) == 1:
        return fftcorrelate(HDvectors1, HDvectors2[np.newaxis].reshape(HDvectors1.shape))
    else:
        raise Exception("Dimensions of arrays must agree or one of them should be a vector")

def bundle(HDvectors, norm="frobenius", optype="unrestricted", kappa=3, **kwargs):
    if len(HDvectors[0].shape) == 1:
        base = HDvectors[0]
    else:
        base = np.sum(HDvectors[0], axis=1)
        if norm == "frobenius":
            base = np.divide(np.expand_dims(base, axis=1), np.linalg.norm(base, axis=0))
        elif norm == "scaling":
            base = np.expand_dims(base, axis=1) / math.sqrt(HDvectors[0].shape[1])
        else:
            raise Exception("The normalization method is not defined")

    # Regular addition
    if optype == "unrestricted":
        return base
    # Clipping function
    elif optype == "clipping":
        base[base > kappa] = kappa
        base[base < -kappa] = -kappa
        return base
    else:
        raise Exception("The type is not defined")


def item(concepts, N=1000):
    memory = np.random.normal(loc = 0.0, scale = math.sqrt(1/N), size =(N, len(concepts)))
    return [memory,concepts]


def similarity(HDvectors1, HDvectors2, stype="dot"):
    dp = np.dot(np.transpose(HDvectors1), HDvectors2)
    if stype == "dot":
        return dp
    elif stype == "cosine":
        norms = (np.dot(np.expand_dims(np.linalg.norm(HDvectors1, axis=0), axis=1),
                        np.expand_dims(np.linalg.norm(HDvectors2, axis=0), axis=0)))
        dp.astype(float)
        return np.divide(dp, norms)
    else:
        raise Exception("The type is not defined")


def getitems(itemmem, concepts):
    HDvectors = np.zeros((itemmem[0].shape[0], len(concepts)), dtype=float)
    for i in range(len(concepts)):
        con = concepts[i]
        ind = itemmem[1].index(con)
        if ind != -1:
            HDvectors[:, i] = itemmem[0][:, ind]
        else:
            raise Exception("The concept is not present in the item memory")
    return [HDvectors, concepts]


def probe(itemmem, query, searchtype="nearest", simtype="dot"):
    scores = similarity(itemmem[0], query, stype=simtype)

    if searchtype == "nearest":
        ind = np.argmax(scores)
        return [np.expand_dims(itemmem[0][:, ind], axis=1), itemmem[1][ind]]
    else:
        raise Exception("The specified search type is not defined")
