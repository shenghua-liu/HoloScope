import sys,math
sys.path.append('../')
import numpy as np
import scipy.sparse.linalg as slin
from scipy.sparse import coo_matrix, csr_matrix, csc_matrix
from os.path import expanduser
home = expanduser("~")


def loadtensor2matricization(tensorfile, sumout=[], mtype=coo_matrix,
                             weighted=True, dtype=int):
    'sumout: marginized (sumout) the given ways'
    matcols={}
    rindexcols={}
    xs, ys, data = [], [], []
    with open(tensorfile, 'rb') as f:
        for line in f:
            elems = line.strip().split(',')
            elems = np.array(elems)
            u = int(elems[0])
            colidx = range(1,len(elems)-1) #remove sumout
            colidx = set(colidx) - set(list(sumout))
            colidx = sorted(list(colidx))
            col=' '.join(elems[colidx])
            if col not in matcols:
                idx = len(matcols)
                matcols[col] = idx
                rindexcols[idx]=col
            cid = matcols[col]
            w = dtype(elems[-1])
            xs.append(u)
            ys.append(cid)
            data.append(w)
        nrow, ncol = max(xs)+1, max(ys)+1
        sm = mtype( (data, (xs, ys)), shape=(nrow, ncol), dtype=dtype )
        if weighted is False:
            sm.data[0:] = dtype(1)
        f.close()

    return sm, rindexcols

