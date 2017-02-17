import sys
sys.path.append('../')
import numpy as np
import math
import random
import numpy.random as nr
import matplotlib.pyplot as plt
import scipy.linalg as sla
from mytools.ioutil import *
from scipy.sparse import coo_matrix, lil_matrix, csc_matrix
from os.path import expanduser
home = expanduser("~")

def genEvenDenseBlock(A, B, p):
    m = []
    for i in xrange(A):
        a = np.random.binomial(1, p, B)
        m.append(a)
    return np.array(m)

def genPowerLawBlock(A, B, alpha, c=1):
    m = []
    m.append(np.ones(B,int))
    for i in xrange(A-1, 0, -1):
        n1 = max(1, np.int64( c * i**alpha ))
        l = np.ones(n1,int)
        r = np.zeros(B-n1, int)
        m.append( np.append(l,r) )
    return np.array(m)

def genTriDenseBlock(A1, B1, A2, B2, A3, B3, p1=0.9, alpha2=0.1, alpha3=0.8):
    c2 = np.float64(B2)/(A2**alpha2)
    c3 = np.float64(B3)/(A3**alpha3)
    M1 = genEvenDenseBlock(A1, B1, p1)
    M2 = genPowerLawBlock(A2, B2,  alpha2, c2)
    M3 = genPowerLawBlock(A3, B3, alpha3, c3)
    M = sla.block_diag(M1, M2, np.transpose(M3))
    return M

def genHyperbolaDenseBlock(A, B, alpha, c):
    '''this is a kind of power low which we use alpha as negative
    and the number of rows and cols are restricted'''
    m = []
    for i in xrange(A):
        fi = c*((i+1)**alpha)
        n1 = min(B, int( math.ceil(fi) ) )
        l=np.ones(n1,int)
        n0 = max(0, B-n1)
        r=np.zeros(n0)
        m.append(np.append(l,r))

    return np.array(m)

def genHyperbolaDenseBlock2(A, B, alpha, tau):
    'this is from hyperbolic paper: i^\alpha * j^\alpha > \tau'
    m = np.empty([A, B], dtype=int)
    for i in xrange(A):
        for j in xrange(B):
            if (i+1)**alpha * (j+1)**alpha > tau:
                m[i,j] = 1
            else:
                m[i,j] = 0
    return m

def genDiDenseBlock(A, B, A2, B2, alpha=-1, p=1):
    alpha = -alpha if alpha>0 else alpha
    c = B*A**(-alpha)
    assert(A2>A and B2>B)
    M1 = genEvenDenseBlock(A,B,p)
    M2 = genHyperbolaDenseBlock(A2, B2, alpha, c)
    M = sla.block_diag(M1,M2)
    return M

def genDiHyperRectBlocks(A1, B1, A2, B2, alpha=-0.5, tau=None, p=1):
    if tau is None:
        tau = A1**alpha * B1**alpha
    m1 = genEvenDenseBlock(A1, B1, p=p)
    m2 = genHyperbolaDenseBlock2(A2, B2, alpha, tau)
    M = sla.block_diag(m1, m2)
    return M

def addnosie(M, A, B, p, black=True, A0=0, B0=0):
    v = 1 if black else 0
    for i in xrange(A-A0):
        a = np.random.binomial(1, p, B-B0)
        for j in a.nonzero()[0]:
            M[A0+i,B0+j]=v
    return M


def genTriRectBlocks(A, B, p1, p2, p3, p=0.001):
    m = genEvenDenseBlock(A, B, p)
    m = addnosie(m, A/3, B/2, p1, black=True)
    m = addnosie(m, A0=A/3, A=2*A/3, B0=B/3, B=2*B/3, p=p2, black=True)
    m = addnosie(m, A0=2*A/3, A=A, B0=B/2, B=B, p=p3, black=True)
    return m


def injectCliqueCamo(M, m0, n0, p, testIdx):
    (m,n) = M.shape
    M2 = M.copy().tolil()

    colSum = np.squeeze(M2.sum(axis = 0).A)
    colSumPart = colSum[n0:n]
    colSumPartPro = np.int_(colSumPart)
    colIdx = np.arange(n0, n, 1)
    population = np.repeat(colIdx, colSumPartPro, axis = 0)

    for i in range(m0):
        # inject clique
        for j in range(n0):
            if random.random() < p:
                M2[i,j] = 1
        # inject camo
        if testIdx == 1:
            thres = p * n0 / (n - n0)
            for j in range(n0, n):
                if random.random() < thres:
                    M2[i,j] = 1
        if testIdx == 2:
            thres = 2 * p * n0 / (n - n0)
            for j in range(n0, n):
                if random.random() < thres:
                    M2[i,j] = 1
        # biased camo           
        if testIdx == 3:
            colRplmt = random.sample(population, int(n0 * p))
            M2[i,colRplmt] = 1

    return M2.tocsc()

def injectCliqueCamo2Unpopular(M, acnt, bcnt, p, popbd, testIdx):
    '''popbd is the popular bound'''
    (m,n) = M.shape
    M2 = M.copy().tolil()

    colSum = M2.sum(0).getA1()
    colids = np.arange(n, dtype=int)
    targetcands = np.argwhere(colSum < popbd).flatten()
    targets = random.sample(targetcands, bcnt)
    camocands = np.setdiff1d(colids, targets, assume_unique=True)
    camoprobs = colSum[camocands]/float(colSum[camocands].sum())
    #population = np.repeat(camocands, colSum[camocands].astype(int), axis=0)
    fraudcands = np.arange(acnt,dtype=int) #users can be hacked
    fraudsters = random.sample(fraudcands, acnt)

    for i in fraudcands:
        #
        for j in targets:
            if random.random() < p:
                M2[i,j] = 1
        # inject camo
        if testIdx == 1:
            thres = p * bcnt / (n - bcnt)
            for j in targets:
                if random.random() < thres:
                    M2[i,j] = 1
        if testIdx == 2:
            thres = 2 * p * bcnt / (n - bcnt)
            for j in targets:
                if random.random() < thres:
                    M2[i,j] = 1
        # biased camo           
        if testIdx == 3:
            colRplmt = np.random.choice(camocands, size=int(bcnt*p),
                                        p=camoprobs)
            M2[i,colRplmt] = 1

    return M2, (fraudsters, targets)

def generateProps(rates, times, k, s, t0, tsdiffcands, tsp):

    if len(rates) > 0:
        rs = np.random.choice([4, 4.5], size=s)
        if k in rates:
            for r in rs:
                rates[k].append(r)
        else:
            rates[k] = list(rs)
    if len(times) > 0:
        ts = np.random.choice(tsdiffcands, size=s, p=tsp) + t0
        if k in times:
            for t in ts:
                times[k].append(t)
        else:
            times[k] = list(ts)
    return

def injectFraud2PropGraph(freqfile, ratefile, tsfile, acnt, bcnt, goal, popbd,
                          testIdx = 3, idstartzero=True, re=False, suffix=None,
                         weighted=True, output=True):
    if not idstartzero:
        print 'we do not handle id start 1 yet for ts and rate'
        ratefile, tsfile = None, None

    M = loadedge2sm(freqfile, coo_matrix, weighted=weighted,
                     idstartzero=idstartzero)
    'smax: the max # of multiedge'
    smax = M.data.max() #max freqency
    if acnt == 0 and re:
        return M, ([], [])
    M2 = M.tolil()
    (m, n) = M2.shape
    rates, times, tsdiffs, t0 = [], [], [], 0
    t0, tsdiffcands,tsp = 0, [], []
    if ratefile is not None:
        rates = loadDictListData(ratefile, ktype=str, vtype=float)
    if tsfile is not None:
        times = loadDictListData(tsfile, ktype=str, vtype=int)
        tsmin, tsmax = sys.maxint,0
        tsdiffs = np.array([])
        prodts={i:[] for i in xrange(n)}
        for k,v in times.iteritems():
            k = k.split('-')
            pid = int(k[1])
            prodts[pid] += v
        for pv in prodts.values():
            pv = sorted(pv)
            minv, maxv = pv[0], pv[-1]
            if tsmin > minv:
                tsmin = minv
            if tsmax < maxv:
                tsmax = maxv
            if len(pv)<=2:
                continue
            vdiff = np.diff(pv)
            'concatenate with [] will change value to float'
            tsdiffs = np.concatenate((tsdiffs, vdiff[vdiff>0]))
        tsdiffs.sort()
        tsdiffs = tsdiffs.astype(int)
	tsdiffcands = np.unique(tsdiffs)[:20] #another choice is bincount
	tsp = np.arange(20,dtype=float)+1
	tsp = 1.0/tsp
	tsp = tsp/tsp.sum()
    t0 = np.random.randint(tsmin, tsmax,dtype=int)

    colSum = M2.sum(0).getA1()
    colids = np.arange(n, dtype=int)
    targetcands = np.argwhere(colSum < popbd).flatten()
    targets = random.sample(targetcands, bcnt)
    camocands = np.setdiff1d(colids, targets, assume_unique=True)
    camoprobs = colSum[camocands]/float(colSum[camocands].sum())
    #population = np.repeat(camocands, colSum[camocands].astype(int), axis=0)
    fraudcands = np.arange(m,dtype=int) #users can be hacked
    fraudsters = random.sample(fraudcands, acnt)
    'rating times for one user to one product, multiedge'
    scands = np.arange(1,smax+1,dtype=int)
    sprobs = 1.0/scands
    sprobs = sprobs/sprobs.sum()
    # inject near clique
    '''
    for i in fraudsters:
        for j in targets:
            s = np.random.choice(scands, size=1, p=sprobs)
            if random.random() < p:
                M2[i,j] += s
                k = '{}-{}'.format(i,j)
                generateProps(rates, times, k, s, t0, tsdiffcands,tsp)
    '''
    for j in targets:
        exeusers = random.sample(fraudsters, goal)
        for i in exeusers:
            s = np.random.choice(scands, size=1, p=sprobs)[0] if weighted else 1
            if (not weighted) and M2[i,j] > 0:
                continue
            M2[i,j] += s
            k = '{}-{}'.format(i,j)
            generateProps(rates, times, k, s, t0, tsdiffcands,tsp)

    # inject camo
    p = goal/float(acnt)
    for i in fraudsters:
        if testIdx == 1:
            thres = p * bcnt / (n - bcnt)
            for j in targets:
                s = np.random.choice(scands, size=1, p=sprobs) if weighted else 1
                if (not weighted) and M2[i,j] > 0:
                    continue
                if random.random() < thres:
                    M2[i,j] += s
                    k = '{}-{}'.format(i,j)
                    generateProps(rates, times, k, s, t0, tsdiffcands, tsp)
        if testIdx == 2:
            thres = 2 * p * bcnt / (n - bcnt)
            for j in targets:
                s = np.random.choice(scands, size=1, p=sprobs) if weighted else 1
                if (not weighted) and M2[i,j] > 0:
                    continue
                if random.random() < thres:
                    M2[i,j] += s
                    k = '{}-{}'.format(i,j)
                    generateProps(rates, times, k, s, t0, tsdiffcands, tsp)
        # biased camo           
        if testIdx == 3:
            colRplmt = np.random.choice(camocands, size=int(bcnt*p),
                                        p=camoprobs)
            #M2[i,colRplmt] = 1
            s = np.random.choice(scands, size=1, p=sprobs) if weighted else 1
            for j in colRplmt:
                if (not weighted) and M2[i,j] > 0:
                    continue
                M2[i,j] += s
                k = '{}-{}'.format(i,j)
                generateProps(rates, times, k, s, t0, tsdiffcands, tsp)

    if suffix is not None:
        suffix = str(suffix)
    else:
        suffix =''
    if ratefile is not None and output is True:
        saveDictListData(rates, ratefile+'.inject'+suffix)
    if tsfile is not None and output is True:
        saveDictListData(times, tsfile+'.inject'+suffix)
    M2 = M2.tocoo()
    if not weighted:
        M2.data[0:] =1
    if output is True:
        savesm2edgelist(M2.astype(int), freqfile+'.inject'+suffix, idstartzero=idstartzero)
        saveSimpleListData(fraudsters, freqfile+'.trueA'+suffix)
        saveSimpleListData(targets, freqfile+'.trueB'+suffix)
    if re:
        return M2, (fraudsters, targets)
    else:
        return

if __name__=="__main__":
    #datapath=home+'/Data/'
    datapath='./testdata/'
    dataname='example1.txt'
    data=datapath+dataname
    #m = genTriDenseBlock(1000, 1000, 1000, 500, 1000,1000, p1=0.8, alpha2=3,
    #                     alpha3=9.0)
    #m = genDiDenseBlock(200,200, 1000, 1000, alpha=-1.5, p=0.8)
    #A1,B1,A2,B2=200,200,1500,1500 #500,500, 1500, 1500
    #m = genDiDenseBlock(A1,B1,A2, B2, alpha=-1)
    #m=addnosie(m, A1+A2, B1, 0.005, black=True, A0=A1, B0=0)
    #m=addnosie(m, A1, B1+B2, 0.005, black=True, A0=0, B0=B1)
    #m=addnosie(m, A1+A2, B1+B2, 0.4, black=False)

    #m = genTriRectBlocks(1500,1500,0.6,0.6,0.6)

    #m = genDiHyperRectBlocks(50, 50, 2500, 2500, alpha=-0.5)
    #m=addnosie(m, 2550, 50, 0.005, black=True, A0=50, B0=0)
    #m=addnosie(m, 50, 2550, 0.005, black=True, A0=0, B0=50)
    #m = genDiHyperRectBlocks(200, 200, 2500, 2500, alpha=-0.5, tau=0.02)
    #m = genDiHyperRectBlocks(50, 50, 2500, 2500, alpha=-0.5, tau=0.002)
    A1,B1,A2,B2= 500,500, 2500, 2500
    m = genDiHyperRectBlocks(A1, B1, A2, B2, alpha=-0.5, tau=0.002)
    m = addnosie(m, A1+A2, B1+B2, 0.005, black=True, A0=0, B0=0)
    m = addnosie(m, A1+A2, B1+B2, 0.99, black=False, A0=0, B0=0)
    #m[0:500][:,500:3000]=m[500:1000][:,500:3000]

    #m = genTriRectBlocks(3000,3000,0.6,0.6,0.6)
    sm = coo_matrix(m)
    sm = injectCliqueCamo(sm, 500, 500, p=0.005, testIdx=3)
    #sm, susp = injectCliqueCamo2Unpopular(sm, 500, 500, p=0.1, popbd=200, testIdx=3)
    fig=plt.figure()
    plt.spy(sm, marker=',', markersize = 1)
    fig.show()

