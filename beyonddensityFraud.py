import sys, os, time
sys.path.append('../')
import numpy as np
import scipy as sci
import numpy.random as nr
import scipy.sparse.linalg as slin
import copy
from collections import OrderedDict
from mytools.MinTree import MinTree
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix, lil_matrix
from mytools.ioutil import loadedge2sm, saveSimpleListData
import matplotlib.pyplot as plt
from gendenseblock import *
from matricizationSVD import *
from edgepropertyAnalysis import *
from os.path import expanduser
from scipy.stats import rankdata
import math
home = expanduser("~")

class Ptype(object):
    freq =0
    ts = 1
    rate=2
    @staticmethod
    def ptype2str(p):
        if p == Ptype.freq:
            return 'freq'
        if p == Ptype.ts:
            return 'ts'
        if p == Ptype.rate:
            return 'rate'
    @staticmethod
    def ptypes2str(ptypes):
        strs=[]
        if Ptype.freq in ptypes:
            strs.append(Ptype.ptype2str(Ptype.freq))
        if Ptype.ts in ptypes:
            strs.append(Ptype.ptype2str(Ptype.ts))
        if Ptype.rate in ptypes:
            strs.append(Ptype.ptype2str(Ptype.rate))
        pstr = '-'.join(strs)
        return pstr

class BeyondDensityOpt:
    def __init__(self, graphmat, qfun='exp', b=64, suspbd= 0.0,
                 aggmethod='joint', sdrop=True, mbd=0.5, sdropscale='linear',
                 tsfile=None, tunit='s', ratefile=None, fileidstartzero=True,
                 sigma=0.10, coe=0, qchopbd=10 ):
        'how many times of a user rates costumers if he get the cost balance'
        self.coe = coe
        #self.eog = 0.1
        self.sigma = sigma #number of rating that a customer requires
        'the larger expbase can give a heavy penalty to the power-law curve'
        self.expbase = b #200 #64 #2**5, to make 0.8 susp ratio has 0.5 chance been abnormal
        self.scale = qfun #lin, const, pl, exp
        self.b = b
        self.aggmethod=aggmethod
        self.suspbd = suspbd #susp < suspbd will assign to zero
        self.qchopbd = qchopbd #qchopbd effect both chopping users at start() and prod susp 
        self.priordropslop=sdrop

        self.graph=graphmat
        self.graphr = graphmat.tocsr()
        self.graphc = graphmat.tocsc()
        self.matricizetenor=None
        self.nU, self.nV=graphmat.shape
        self.indegrees = graphmat.sum(0).getA1()
        print 'matrix size: {} x {}\t#edges: {}'.format(self.nU, self.nV,
                                                          self.indegrees.sum())

        self.tsfile, self.ratefile, self.tunit = tsfile, ratefile, tunit
        self.tspim, self.ratepim = None, None
        'field for multiple property graph'
        if tsfile is not None or ratefile is not None:
            if self.priordropslop:
                self.orggraph = self.graphr.copy()
            else:
                self.orggraph = self.graphr
        if tsfile is not None:
            self.mbd = mbd #multiburst bound
            self.tspim = MultiEedgePropBiGraph(self.orggraph)
            self.tspim.load_from_edgeproperty(tsfile,
                                             idstartzero=fileidstartzero,
                                             mtype=csr_matrix, dtype=np.int64)
            self.tspim.setup_ts4all_sinks(tunit)
            if self.priordropslop:
                'slops weighted with max burst value'
                self.weightWithDropslop(weighted=True, scale=sdropscale)
        if ratefile is not None:
            self.ratepim = MultiEedgePropBiGraph(self.orggraph)
            self.ratepim.load_from_edgeproperty(ratefile,
                                                idstartzero=fileidstartzero,
                                                mtype=csr_matrix, dtype=float)
            self.ratepim.setup_rate4all_sinks()

        'weighed with idf prior from Fraudar'
        #self.weightWithIDFprior()
        'if weighted the matrix the windegrees is not equal to indegrees'
        self.windegrees = self.graphc.sum(0).getA1()
        self.woutdegrees = self.graphr.sum(1).getA1()

        'No Need: optimize the fbs, bsusps as the sparse'
        self.A = np.array([]) #binary array
        self.fbs = np.zeros(graphmat.shape[1], dtype=np.int) #frequency of bs in B
        '\frac_{ f_A{(bi)} }{ f_U{(bi)}}'
        self.bsusps = np.array([]) # the suspicious scores of products given A
        self.vx = 0 # current objective value
        self.vxs = [] #record all the vxs of optimizing iterations
        self.Y= np.array([])
        self.yfbs = np.array([])
        self.ybsusps = np.array([])
        'current is the best'
        self.bestvx = self.vx
        self.bestA = np.array([])
        self.bestfbs = np.array([])
        self.bestbsusps = np.array([])

        self.qchop = False # if do the quick chop at the begining
        self.qchopusers = np.array([])
        if self.qchop:
            self.excludeprods=np.argwhere( self.indegrees < self.qchopbd
                                         ).flatten()
        else:
            self.excludeprods=np.array([])
        '''
        with quick chop, not all the users are suitable for adding
        we do not want to add quick chopping users back
        '''
        self.wholeaddcands = np.arange(self.nU, dtype=int)
        '''the number of proper users to be considered. if quick chopping (qchop
        is True), npropA will be the number of nonzeros'''
        self.npropA = self.nU

    def weightWithDropslop(self, weighted, scale='log1p'):
        'weight the adjacency matrix with the sudden drop of ts for each col'
        if weighted:
            colWeights = np.multiply(self.tspim.dropslops, self.tspim.dropfalls)
        else:
            colWeights = self.tspim.dropslops
        if scale == 'logistic':
            from scipy.stats import logistic
            from sklearn import preprocessing
            'zero mean scale'
            colWeights = preprocessing.scale(colWeights)
            colWeights = logistic.cdf(colWeights)
        elif scale == 'linear':
            from sklearn import preprocessing
            #add a base of suspecious for each edge
            colWeights = preprocessing.minmax_scale(colWeights) +1
        elif scale == 'plusone':
            colWeights += 1
        elif scale == 'log1p':
            colWeights = np.log1p(colWeights) + 1
        else:
            print '[Warning] no scale for the prior weight'

        n = self.nV
        colDiag = lil_matrix((n, n))
        colDiag.setdiag(colWeights)
        self.graphr = self.graphr * colDiag.tocsr()
        self.graph = self.graphr
        self.graphc = self.graph.tocsc(copy=False)
        print "finished computing weight matrix"

    def weightWithIDFprior(self):
        print 'weightd with IDF prior'
        colWeights = 1.0/np.log(self.indegrees + 5)
        n = self.nV
        colDiag = lil_matrix((n, n))
        colDiag.setdiag(colWeights)
        self.graphr = self.graphr * colDiag.tocsr()
        self.graph = self.graphr
        self.graphc = self.graph.tocsc(copy=False)
        return

    'No use. Notice: the algorithm use default usesigma'
    def maxobjfunc_sigma(self, A, fbs, bsusps=None, usesigma=True):
        'todo: use lil_matrix for fbs, and use dot for sum'
        '''
            fbs=lil_matrix(fbs)
            bsusps=lil_matrix(fbs)
            fbs.dot(bsusps) #check
        '''
        nu = 0.0
        de = 0.0
        numA = np.sum(A)
        #de = np.sum(fbs)
        'opt2 camouflag resistence'
        de = numA + bsusps.sum() #math.sqrt(numA*bsusps.sum())#similar
        if de == 0 and numA == 0:
            return 0

        if usesigma is False:
            fbs2 = fbs**2/float(numA)
            sigma = 1e-3
        else:
            fbs2 = fbs
            sigma = opt.sigma

        #suspidx = np.where(fbs>=sigma * numA)[0] #note: use !fbs here
        suspidx = np.array([True]*len(fbs))
        if bsusps is None:
            nu = np.sum(fbs2[suspidx])
        else:
            nu = np.dot(fbs2[suspidx], bsusps[suspidx])

        res = nu/np.float64( de + self.coe * numA )
        return res

    'new objective with no f_A(v)/|A|, and no sigma'
    def maxobjfunc(self, A, fbs, bsusps=None):
        nu = 0.0
        de = 0.0
        numA = np.sum(A)
        de = numA + bsusps.sum() #math.sqrt(numA*bsusps.sum())#similar
        if numA == 0:
            return 0
        #sigma = 1e-3
        #suspidx = np.where(fbs>=sigma * numA)[0]
        #nu = np.dot(fbs[suspidx], bsusps[suspidx])
        if bsusps is not None:
            nu = np.dot(fbs, bsusps)
        else:
            nu = fbs.sum()
        res = nu/np.float64( de + self.coe * numA )
        return res

    def aggregationMultiProp(self, mbs, method='sum'):
        rankmethod = 'average'
        k=60 #for rank fusion
        if len(mbs) == 1:
            val = mbs.values()[0]
            if method == 'rank':
                rb = rankdata(-np.array(val), method=rankmethod)
                return np.reciprocal(rb+k) * k
            else:
                return val
        if method == 'joint':
            'use joint probability'
            bsusps = mbs.values()[0]
            for v in mbs.values()[1:]:
                bsusps = np.multiply(bsusps, v)
        elif method == 'sum':
            bsusps = mbs.values()[0]
            for v in mbs.values()[1:]:
                bsusps += v
        elif method == 'rank':
            'rank fusion'
            arrbsusps = []
            for val in mbs.values():
                rb = rankdata(-np.array(val), method=rankmethod)
                arrbsusps.append(np.reciprocal(rb+k))
            bsusps = np.array(arrbsusps).sum(0) * k
        else:
            print '[Error] Invalid method {}\n'.format(method)
        return bsusps

    #@profile
    def evalsusp4ts(self, suspusers, multiburstbd = 0.5, weighted=True):
        'the id of suspusers consistently starts from 0 no matter the source'
        #self.tspim.setupsuspects(suspusers)
        incnt, inratio = self.tspim.suspburstinvolv(multiburstbd, weighted,
                                                    delta=True)
        suspts=inratio
        return suspts

    #@profile
    def evalsusp4rate(self, suspusers, neutral=False, scale='max'):
        #self.ratepim.setupsuspects(suspusers), scaling=False
        susprates = self.ratepim.suspratedivergence(neutral, delta=True)
        if scale == 'max':
            assert(self.ratepim.maxratediv > 0)
            nsusprates = susprates/self.ratepim.maxratediv
        elif scale=='minmax':
            #need a copy, and do not change susprates' value for delta
            nsusprates = preprocessing.minmax_scale(susprates, copy=True)
        else:
            #no scale 
            nsusprates = susprates
        return nsusprates

    'sink suspicious with qfunc, no f_A(v)/|A|'
    def prodsuspicious(self, fbs, A=None, scale='exp', ptype=[Ptype.freq]):
        multibsusps={}
        if Ptype.freq in ptype:
            posids = self.windegrees>0
            bs = np.zeros(self.nV)
            bs[posids] = np.divide(fbs[posids], self.windegrees[posids].astype(np.float64))
            multibsusps[Ptype.freq] = bs
            if scale == 'log' or scale == 'logratio':
                'log, logratio only works for freq'
                multibsusps[Ptype.freq] =\
                        self.qfunc(bs, sfbs = sfbs.toarray()[0], scale=scale)
                scale='pl'
                self.b, self.suspbd = 1, 0.0
        if Ptype.ts in ptype:
            suspusers = A.nonzero()[0]
            bs = self.evalsusp4ts(suspusers, multiburstbd=self.mbd)
            multibsusps[Ptype.ts] = bs
        if Ptype.rate in ptype:
            suspusers = A.nonzero()[0]
            bs = self.evalsusp4rate(suspusers)
            multibsusps[Ptype.rate] = bs
        bsusps = self.aggregationMultiProp(multibsusps, self.aggmethod)
        #lessbdidx = np.argwhere(fbs < self.sigma * sizeA).flatten()
        if self.qchop:
            bsusps[self.excludeprods] = 0.0
        bsusps = self.qfunc(bsusps, fbs=fbs, scale=scale)
        return bsusps
    'has a problem to keep large indegree of v'
    def prodsuspicious_no(self, fbs, sizeA=None, scale='exp'):
        posids = self.windegrees>0
        bs = np.zeros(self.nV)
        bs[posids] = np.divide(fbs[posids], self.windegrees[posids].astype(np.float64))
        if sizeA > 0:
            bsusps = np.multiply(bsusps, fbs/float(sizeA))
        else:
            bsusps *= 0
        return bsusps

    def quickchopUsers(self):
        possibleB = np.argwhere(self.indegrees >= self.qchopbd).flatten()
        'no any contribution to possibleB'
        cusers = np.argwhere( self.graphc[:,possibleB].getnnz(1) <= 0 )[:,0]
        return cusers

    def initpimsuspects(self, suspusers, ptype):
        if Ptype.ts in ptype:
            self.tspim.setupsuspects(suspusers)
            temp1, temp2 = self.tspim.suspburstinvolv(multiburstbd=0.5, weighted=True,
                                       delta=False)
        if Ptype.rate in ptype:
            self.ratepim.setupsuspects(suspusers)
            tmp = self.ratepim.suspratedivergence(neutral=False,
                                            delta=False)
        return

    def start(self, A0, ptype=[Ptype.ts]):
        if self.qchop is True:
            self.qchopusers = self.quickchopUsers()
            for i in self.qchopusers:
                A0[i] = 0
            self.wholeaddcands = np.delete(np.arange(self.nU, dtype=int),
                                           self.qchopusers)
        self.A = A0
        self.npropA = np.sum(A0)
        users = A0.nonzero()[0]
        self.ptype=ptype # the property type that the postiorer uses
        self.fbs = self.graphr[users].sum(0).getA1()
        self.fbs = self.fbs.astype(np.float64, copy=False)
        'initially set up currrent suspects'
        self.initpimsuspects(users, ptype=ptype)
        self.bsusps = self.prodsuspicious(self.fbs, self.A, ptype=ptype)
        self.vx = self.maxobjfunc(self.A, self.fbs, self.bsusps)
        self.vxs.append(self.vx)
        "current is the best"
        self.bestA = np.array(self.A)
        self.bestvx = self.vx
        self.bestfbs = np.array(self.fbs)
        self.bestbsusps = np.array(self.bsusps)

    def candidatefbs(self, z):
        'increase or decrease'
        coef = 1 if self.A[z] == 0 else -1
        '''
        for  b in self.graphr.getrow(z).nonzero()[0]:
            candfbs[b] += coef * self.graph[z,b]
            #assert(candfbs[b].all()>=0)
        '''
        bz = self.graphr[z]
        candfbs = (coef*bz + self.fbs).getA1()
        return candfbs

    #@profile
    def greedyshaving(self, sampleB=False):
        '''greedy algorithm'''
        maxint = np.iinfo(np.int64).max/2
        delscores = np.array([maxint]*self.nU)
        #abproducts = np.argwhere( self.fbs >= self.sigma )[:,0]
        delcands = self.A.nonzero()[0]
        #deluserCredit = self.graphr[delcands,:].tocsc()[:,abproducts].sum(1).getA1()
        deluserCredit = self.graphr[delcands,:].dot(self.bsusps)
        delscores[delcands] = deluserCredit
        print 'set up the greedy min tree'
        MT = MinTree(delscores)
        i=0
        sizeA = np.sum(self.A)
        sizeA0 = sizeA
        setA = set(self.A.nonzero()[0])
        while len(setA) > 0:
            z, nextdelta = MT.getMin()
            setY = setA - {z}
            Y = copy.copy(self.A) # A is X
            Y[z] = 1-Y[z]
            self.Y=Y
            self.yfbs = self.candidatefbs(z)
            Ylist = Y.nonzero()[0]
            self.setdeltapimsusp(z, Ylist, add=False)
            #self.yfbs = self.yfbs.astype(np.float64, copy=False )
            self.ybsusps = self.prodsuspicious(self.yfbs, self.Y,
                                               ptype=self.ptype)
            vy = self.maxobjfunc(self.Y, self.yfbs, self.ybsusps)
            'chose next if next if the best'
            if vy > self.bestvx:
                self.bestA = np.array(self.Y)
                self.bestfbs = self.yfbs
                self.bestbsusps = self.ybsusps
                self.bestvx = vy
            MT.changeVal(z, maxint) #make the min to the largest
            '''changed abnormal products since of user deletion.
            To be optimized for efficency
            The following excludeProds is not assuming the weight is 1 or
            weighted
            '''
            '''"opt 1"
            remainusers = self.Y.nonzero()[0]
            remusermat = self.graphr[remainusers,:]
            #the remaining users should change their weight in MinT
            remuserdelta = remusermat.dot(self.ybsusps - self.bsusps)
            for u, cc in zip(remainusers, remuserdelta):
                if cc == 0:
                    continue
                else:
                    MT.changeVal(u,cc)
            '''
            '''"opt 2"
            effectusers = self.u2ugraph[z].nonzero()[1] #may be effected users
            effectusermat = self.graphr[effectusers]
            effectuserdelta = effectusermat.dot(self.ybsusps - self.bsusps)
            for u,cc in zip(effectusers, effectuserdelta):
                if cc == 0:
                    'no effect on user u by deleting z'
                    continue
                else:
                    MT.changeVal(u,cc)
            '''
            '''"opt 3"
            prodchange = self.ybsusps - self.bsusps
            effectprod = prodchange.nonzero()[0]
            userdelta = np.zeros(self.nU,dtype=int)
            for p in effectprod:
                for u, freq in zip(self.graphT.rows[p], self.graphT.data[p]):
                    if self.Y[u] == 1:
                        userdelta[u] += freq * prodchange[p]
            for u in userdelta.nonzero()[0]:
                MT.changeVal(u, userdelta[u])
            '''
            if not sampleB:
                'opt 4: current best'
                prodchange = self.ybsusps - self.bsusps
                effectprod = prodchange.nonzero()[0]
                if len(effectprod)>0:
                    #this is delta for all users
                    userdelta = self.graphc[:,effectprod].dot(prodchange[effectprod])
                    yuserdelta = userdelta[Ylist]
                    for u in yuserdelta.nonzero()[0]:
                        uidx = Ylist[u]
                        MT.changeVal(uidx,yuserdelta[u])
            else:
                'random sample opt5 based on opt4'
                effectprod = self.graphr[z].nonzero()[1] #ids
                l = len(effectprod)
                if l > 0:
                    beta = min(1.0, 500.0/l)
                    #beta = max(0.1, beta)
                    samp = np.random.binomial(1, beta, l)
                    seffectprod = effectprod[samp.nonzero()[0]]
                    bsuspchange=self.ybsusps[seffectprod] - self.bsusps[seffectprod]
                    userdelta = self.graphc[:,seffectprod].dot(bsuspchange)
                    userdelta = userdelta[list(setY)]
                    #userdelta = np.multiply(userdelta, self.Y) #slow
                    userdelta = userdelta/float(beta) #expectation
                    for u in userdelta.nonzero()[0]:
                        MT.changeVal(u,userdelta[u])

            'delete next user, make current to next'
            self.A = self.Y
            sizeA -= 1
            setA = setY
            self.fbs = self.yfbs
            self.bsusps = self.ybsusps
            self.vx = vy
            self.vxs.append(self.vx)
            if i % (sizeA0/100 + 1) == 0:
                sys.stdout.write('.')
                sys.stdout.flush()
            i+=1
            '''
            if i==3:#for performance test
                break
            '''
        '''
        if np.sum(self.A) == 0:
            print '\nNo abnormal area under conditions. sigma:{}, coe:{}'.format(
                self.sigma, self.coe)
        '''
        print ''
        return np.sum(self.A)

    '''
    Since the deleting greedy starts from a whole set U, it has a complexity
    with |E||E|/|V|.
    We use adding greedy starting from one row
    default: ptype=[Ptype.freq], numSing=10, rbd='avg'
    '''
    def initfastgreedy(self, ptype, numSing, rbd='avg'):
        self.ptype=ptype
        self.numSing=numSing #number of singular vectors we consider
        self.avgexponents=[]
        if len(ptype)==1:
            self.initfastgreedy2D(numSing, rbd)
        elif len(ptype) > 1:
            self.initfastgreedyMD(numSing, rbd)

        self.bestvx = -1
        self.qchop=False
        #reciprocal of indegrees
        self.sindegreciprocal = csr_matrix(self.windegrees).astype(np.float64)
        data = self.sindegreciprocal.data
        nozidx = data.nonzero()[0]
        self.sindegreciprocal.data[nozidx] = data[nozidx]**(-1)
        '''
        self.sA = set({cutrows[0]}) #a set of A
        self.sfbs = csr_matrix(self.graphr[list(self.sA)].sum(0)) #sparse of fbs
        sbsusp = self.sProdsuspicious(self.sfbs, scale=self.scale,
        sizeA=len(self.sA))
        'obj vale'
        self.bestA = set(self.sA)
        self.bestvx = self.vx
        self.vx = self.sMaxobjfunc(len(self.sA), self.sfbs, sbsusp)
        self.vxs = [self.vx]
        '''
        return

    def tenormatricization(self, tspim, ratepim, tbindic, rbins,
                           mtype=coo_matrix, dropweight=True, logdegree=False):
        'matricize the pim of ts and rates into matrix'
        if tspim is None and ratepim is None:
            return self.graph, range(self.nV)
        tscm, rtcm, dl = None, None,0
        if Ptype.ts in self.ptype and tspim is not None:
            tscm = tspim.edgeidxm.tocoo()
            dl = len(tscm.data)
        if Ptype.rate in self.ptype and ratepim is not None:
            rtcm = ratepim.edgeidxm.tocoo()
            dl = len(rtcm.data)
        if dropweight is True and tspim is not None:
            w = np.multiply(tspim.dropfalls, tspim.dropslops)
            #if self.tunit == 's':
            #    w *= 3600
            w = np.log1p(w) + 1
        else:
            w = np.ones(self.nV)
        xs, ys, data, colWeights = [],[],[],[] # for matricized tenor
        matcols, rindexcols={},{}
        for i in xrange(dl):
            if tscm is not None and rtcm is not None:
                assert(tscm.row[i] == rtcm.row[i] and tscm.col[i] == rtcm.col[i])
                u = tscm.row[i]
                v = tscm.col[i]
                for t1, r1 in zip(tspim.eprop[tscm.data[i]],
                                ratepim.eprop[rtcm.data[i]]):
                    t = t1/int(tbindic[self.tunit])
                    r = rbins(r1)
                    strcol = ' '.join(map(str,[v,t,r]))
                    if strcol not in matcols:
                        idx = len(matcols)
                        matcols[strcol] = idx
                        rindexcols[idx]=strcol
                    xs.append(u)
                    ys.append(matcols[strcol])
                    data.append(1.0)
            elif tscm is not None:
                u = tscm.row[i]
                v = tscm.col[i]
                for t1 in tspim.eprop[tscm.data[i]]:
                    t = t1/int(tbindic[self.tunit])
                    strcol = ' '.join(map(str,[v,t]))
                    if strcol not in matcols:
                        idx = len(matcols)
                        matcols[strcol] = idx
                        rindexcols[idx]=strcol
                    xs.append(u)
                    ys.append(matcols[strcol])
                    data.append(1.0)
            elif rtcm is not None:
                u = rtcm.row[i]
                v = rtcm.col[i]
                for r1 in ratepim.eprop[rtcm.data[i]]:
                    r = rbins(r1)
                    strcol = ' '.join(map(str,[v,r]))
                    if strcol not in matcols:
                        idx = len(matcols)
                        matcols[strcol] = idx
                        rindexcols[idx]=strcol
                    xs.append(u)
                    ys.append(matcols[strcol])
                    data.append(1.0)
            else:
                print 'Warning: no ts and rate for matricization'
                return self.graph, range(self.nV)

        nrow, ncol = max(xs)+1, max(ys)+1
        sm = mtype( (data, (xs, ys)), shape=(nrow, ncol), dtype=np.float64 )
        if logdegree:
            print 'using log degree'
            sm.data[0:] = np.log1p(sm.data)
        if dropweight:
            m1, n1 = sm.shape
            for i in xrange(n1):
                pos = rindexcols[i].find(' ')
                v = int(rindexcols[i][:pos])
                colWeights.append(w[v])
            colDiag = lil_matrix((n1, n1))
            colDiag.setdiag(colWeights)
            sm = sm * colDiag.tocsr()
        return sm, rindexcols

    def initfastgreedyMD(self, numSing, rbd):
        '''
            use matricizationSVD instead of freq matrix svd
        '''
        afile = self.tsfile if self.tsfile is not None else self.ratefile
        ipath =  os.path.dirname(os.path.abspath(afile))
        if 'wbdata' not in ipath:
            tbindic={'s':24*3600, 'd':30}
        else:
            print 'init for wbdata'
            tbindic={'s':3600, 'd':30}
        'edgepropertyAnalysis has already digitized the ratings'
        rbins = lambda x: int(x) #lambda x: 0 if x<2.5 else 1 if x<=3.5 else 2
        tunit = self.tunit
        print 'generate tensorfile with tunit:{}, tbins:{}'.format(tunit,
                                                                   tbindic[tunit])
        if self.matricizetenor is None:
            matricize_start = time.time()
            sm, rindexcol = self.tenormatricization(self.tspim, self.ratepim,
                    tbindic, rbins, mtype=coo_matrix,
                    dropweight=self.priordropslop,
                    logdegree=False)
            self.matricizetenor = sm
            print '::::matricize time cost: ', time.time() - matricize_start
        sm = self.matricizetenor
        print "matricize {}x{} and svd dense... ..."\
                .format(sm.shape[0], sm.shape[1])
        u, s, vt = slin.svds(sm, k=numSing, which='LM')
        u = np.fliplr(u)
        s = s[::-1]
        CU, CV = [],[]
        for i in xrange(self.numSing):
            ui = u[:, i]
            si = s[i]
            if abs(max(ui)) < abs(min(ui)):
                ui = -1*ui
            if type(rbd) is float:
                sqrtSi = math.sqrt(si)
                ui *= sqrtSi
                rbdrow= rbd
            elif rbd == 'avg':
                rbdrow = 1.0/math.sqrt(self.nU)
            else:
                print 'unkown rbd {}'.format(rbd)
            rows = np.argsort(-ui, axis=None, kind='quicksort')
            for jr in xrange(len(rows)):
                r = rows[jr]
                if ui[r] <= rbdrow:
                    break
            self.avgexponents.append(math.log(jr, self.nU))
            'consider the # limit'
            if self.nU > 1e6:
                en = math.log(self.graph.sum(), self.nU)
                e0 = 1.6 # math.sqrt(3) # make sure 3/e0 < 2
                ep = max(e0, en) # claim ep < 2
                nn = sm.shape[0] + sm.shape[1]
                #nn = sm.shape[0]
                nlimit = int(math.ceil(nn**(1/ep)))
                cutrows = rows[:min(jr,nlimit)]
            else:
                cutrows = rows[:jr]

            CU.append(cutrows)

        self.CU = np.array(CU)
        self.CV = np.array(CV)
        return

    def initfastgreedy2D(self, numSing, rbd):
        'rbd threshold that cut the singular vecotors, default is avg'
        'parameters for fastgreedy'
        u, s, vt = slin.svds(self.graphr.astype(np.float64), k=numSing, which='LM')
        #revert to make the largest singular values and vectors in the front
        u = np.fliplr(u)
        vt = np.flipud(vt)
        s = s[::-1]
        self.U = []
        self.V = []
        self.CU = []
        self.CV = []
        for i in xrange(self.numSing):
            ui = u[:, i]
            vi = vt[i, :]
            si = s[i]
            if abs(max(ui)) < abs(min(ui)):
                ui = -1*ui
            if abs(max(vi)) < abs(min(vi)):
                vi = -1*vi
            if type(rbd) is float:
                sqrtSi = math.sqrt(si)
                ui *= sqrtSi
                vi *= sqrtSi
                rbdrow, rbdcol = rbd, rbd
            elif rbd == 'avg':
                rbdrow = 1.0/math.sqrt(self.nU)
                rbdcol = 1.0/math.sqrt(self.nV)
            else:
                print 'unkown rbd {}'.format(rbd)
            rows = np.argsort(-ui, axis=None, kind='quicksort')
            cols = np.argsort(-vi, axis=None, kind='quicksort')
            for jr in xrange(len(rows)):
                r = rows[jr]
                if ui[r] <= rbdrow:
                    break
            self.avgexponents.append(math.log(jr, self.nU))
            if self.nU > 1e6:
                en = math.log(self.graph.sum(), self.nU)
                e0 = 1.6 # math.sqrt(3) # make sure 3/e0 < 2
                ep = max(e0, en) # claim ep < 2
                nn = self.nU + self.nV
                nlimit = int(math.ceil(nn**(1.0/ep)))
                cutrows = rows[:min(jr,nlimit)]
            else:
                cutrows = rows[:jr]
            for jc in xrange(len(cols)):
                c = cols[jc]
                if vi[c] <= rbdcol:
                    break
            cutcols = cols[:jc]
            'begin debug'
            self.U.append(ui)
            self.V.append(vi)
            'end debug'
            self.CU.append(cutrows)
            self.CV.append(cutrows)

        self.CU = np.array(self.CU)
        self.CV = np.array(self.CV)
        return

    def qfunc(self, ratios, fbs=None, scale='pl'):
        if self.aggmethod == 'rank':
            'do not use qfun if it is rank aggregation'
            return ratios
        if self.suspbd <= 0.0:
            #lessbdidx = ratios <= 0.0
            greatbdidx = ratios > 0.0
        else:
            greatbdidx = ratios >= self.suspbd
            lessbdidx = ratios < self.suspbd
            'picewise q funciton if < suspbd, i.e. epsilon'
            ratios[lessbdidx] = 0.0
        'picewise q funciton if >= suspbd, i.e. epsilon'
        if scale == 'exp':
            ratios[greatbdidx] = self.expbase**(ratios[greatbdidx]-1)
        elif scale == 'pl':
            ratios[greatbdidx] = ratios[greatbdidx]**self.b
        elif scale == 'lin':
            ratios[greatbdidx] = np.fmax(self.b*(ratios[greatbdidx]-1)+1, 0)
        elif scale == 'const':
            ratios[greatbdidx] = 1.0
        elif scale == 'log' and fbs is not None:
            'tf-idf like weight, only works for Ptype.freq'
            ratios[greatbdidx] = 1/np.log(self.windegrees[greatbdidx] -
                                        fbs[greatbdidx]+math.e)
        elif scale == 'logratio' and fbs is not None:
            'only works for Ptype.freq'
            ratios[greatbdidx] = np.log(self.windegrees[greatbdidx]+1)/\
                    np.log(fbs[greatbdidx]+1)
        elif scale == 'arcsin':
            ratios[greatbdidx] = 2*np.arcsin(ratios[greatbdidx])/(math.pi)
        else:
            print 'unrecognized scale: ' + scale
            sys.exit(1)
        return ratios

    'P(B|A) for all b \in B, no need of f_A(v)/|A|'
    #@profile
    def sProdsuspicious(self, sfbs, suspusers, ptype=[Ptype.freq],
                        scale='exp'):
        suspusers=np.array(suspusers)
        multibsusps={}
        if Ptype.freq in ptype:
            sbs = sfbs.multiply(self.sindegreciprocal)
            bs = sbs.toarray()[0]
            multibsusps[Ptype.freq] = bs
            assert(np.all(bs<=1+1e-6))
            if scale == 'log' or scale == 'logratio':
                'log, logratio only works for freq'
                multibsusps[Ptype.freq] =\
                        self.qfunc(bs, fbs = sfbs.toarray()[0], scale=scale)
                scale='pl'
                self.b, self.suspbd = 1, 0.0
        if Ptype.ts in ptype:
            bs = self.evalsusp4ts(suspusers, multiburstbd=self.mbd)
            multibsusps[Ptype.ts] = bs
            assert(np.all(bs<=1+1e-6))
        if Ptype.rate in ptype:
            bs = self.evalsusp4rate(suspusers)
            multibsusps[Ptype.rate] = bs
        bsusps = self.aggregationMultiProp(multibsusps, self.aggmethod)
        #sbsusps = csc_matrix(bsusps)
        bsusps = self.qfunc(bsusps, fbs = sfbs.toarray()[0], scale=scale)
        sbsusps = csc_matrix(bsusps, dtype=np.float64)
        #sbsusps.eliminate_zeros()
        return sbsusps

    'no use, this has a problem to keep high indegree sink'
    def sProdsuspicious_no(self, sfbs, sizeA, scale='exp'):
        sbsusps = sfbs.multiply(self.sindegreciprocal)
        sbsusps.data = self.qfunc(sbsusps.data, fbs = sfb.toarray()[0], scale=scale)
        sbsusps.eliminate_zeros()
        sbsusps=sbsusps.multiply(sfbs/float(sizeA))
        return sbsusps

    'no need for f_A(v)/|A| in objective'
    def sMaxobjfunc_sigma(self, sizeA, sfbs, sbsusps=None, usesigma=True):
        #de = sfbs.sum()
        'opt 2 camouflage resistance'
        de = sizeA + sbsusps.sum() if sbsusps is not None else \
                sizeA + self.nV
        ''' #opt 1
        lessfbsidx = sfbs.data < opt.sigma * sizeA
        nsfbs = sfbs.copy()
        nsfbs.data[lessfbsidx] = 0
        nu = sbsusps.multiply(nsfbs).sum()
        '''
        if de==0 and sizeA==0:
            return 0
        #opt2 for performance
        nsfbs=sfbs.toarray()[0]
        if usesigma is False:
            nsfbs = nsfbs**2/float(sizeA)
            sigma = 1e-3
        else:
            sigma = opt.sigma
        nsfbs[ sfbs.toarray()[0] < sigma * sizeA ] = 0
        if sbsusps is not None:
            nu = sbsusps.dot(nsfbs)[0]
        else:
            nu = nsfbs.sum()
        obj = nu/np.float64( de + self.coe * sizeA )
        return obj

    'effective: no need for f_A(v)/|A|, and sigma any more'
    def sMaxobjfunc(self, sizeA, sfbs, sbsusps):
        de = sizeA + sbsusps.sum()
        if de==0 and sizeA==0:
            return 0
        nsfbs=sfbs.toarray()[0]
        #sigma=1e-3
        #nsfbs[ nsfbs < sigma * sizeA ] = 0
        nu = sbsusps.dot(nsfbs)[0]
        obj = nu/np.float64( de + self.coe * sizeA )
        return obj

    def setdeltapimsusp(self, z, ysuspusers, add):
        if Ptype.ts in self.ptype:
            self.tspim.deltasuspects(z, ysuspusers, add)
        if Ptype.rate in self.ptype:
            self.ratepim.deltasuspects(z, ysuspusers, add)
        return

    def removecurrentblock(self, rows):
        '''it is for find second block, remove rows from
           self.graph, self.matricizetenor
        '''
        print 'removing {} rows from graph'.format(len(rows))
        lilm = self.graph.tolil()
        lilm[rows,:]=0
        self.graph=lilm.tocsr()
        self.graphc= lilm.tocsc()
        self.graphr = self.graph

        if self.matricizetenor is not None:
            print 'removing {} rows from tensor'.format(len(rows))
            lilmm = self.matricizetenor.tolil()
            lilmm[rows,:] = 0
            self.matricizetenor = lilmm.tocoo()
        return

    #@profile
    def greedyInflating(self, Q, ptype):
        'greedy inflating the ordered candidates'
        lenQ = len(Q)
        if lenQ == 0:
            return
        self.sA = set()
        self.sfbs = csr_matrix((1, self.nV), dtype=np.int64)
        self.initpimsuspects([], ptype=ptype)
        itr = 0
        for r in Q:
            sfbsdelta = self.graphr[r]
            syfbs = self.sfbs + sfbsdelta
            suspusers = list(self.sA)
            suspusers.append(r)
            self.setdeltapimsusp(r, suspusers, add=True)
            sybsusp = self.sProdsuspicious(syfbs,
                                           suspusers=suspusers,
                                           scale=self.scale, ptype=ptype)
            vy = self.sMaxobjfunc(len(suspusers), syfbs, sybsusp)
            'update current x with y'
            self.sA.add(r)
            self.sfbs = syfbs
            self.vx = vy
            self.vxs.append(self.vx)

            ''''stop once vx decrease'
            if self.bestvx > self.vx:
                break
            '''

            if self.bestvx < self.vx or (self.bestvx == self.vx and
            len(self.bestA)<len(self.sA) ):
                #copy a set of sA, since it will be added next iteration
                self.bestA = set(self.sA)
                self.bestvx = self.vx
                self.bestfbs = self.sfbs
                self.bestbsusps = sybsusp

            if itr % (lenQ/10 + 1) == 0:
                sys.stdout.write('.')
                sys.stdout.flush()
            itr+=1
        print ''
        return

    def greedysuddendrop(self, sbd=1):
        '''greedy algo for sudden drop messages
           sort users according to the invlovment of weighted sudden drop
        '''

    #@profile
    def fastgreedy(self, inflating=False, shaving=True):
        'adding and deleting greed algorithm'
        'No Need: user order for r with obj fuct'
        self.fastlocalbest = []
        self.fastbestvx = 0
        self.fastbestA, self.fastbestfbs, self.fastbestbsusps = \
                np.zeros(self.nU), np.zeros(self.nV), np.zeros(self.nV)
        for k in xrange(self.numSing):
            lenCU = len(self.CU[k])
            if lenCU == 0:
                continue
            print 'process {}-th singular vector, size: {}'.format(
                k+1, lenCU)

            if inflating:
                print '-- inflating ...'
                self.greedyInflating(self.CU[k], ptype=self.ptype)
                print '-- current opt size: {}'.format(len(self.bestA))
                print '-- current opt value: {}'.format(self.bestvx)
                A = np.zeros(self.nU, dtype=int)
                A[list(self.bestA)] = 1
                self.bestA = A
                self.bestfbs = self.bestfbs.toarray()[0]
                self.bestbsusps = self.bestbsusps.toarray()[0]
                if self.fastbestvx < self.bestvx:
                    self.fastbestvx = self.bestvx
                    self.fastbestA = self.bestA
                    self.fastbestfbs = self.bestfbs
                    self.fastbestbsusps = self.bestbsusps
            if shaving:
                print '*** *** shaving ...'
                A0 = np.zeros(self.nU, dtype=int)
                if inflating:
                    A0[list(self.bestA)] = 1 #shaving from current best
                else:
                    A0[self.CU[k]]=1 #shaving from sub singluar space
                self.start(A0, ptype=self.ptype)
                self.greedyshaving(sampleB=False)
                print '*** *** shaving opt size: {}'.format(sum(self.bestA))
                print '*** *** shaving opt value: {}'.format(self.bestvx)
                if self.fastbestvx < self.bestvx:
                    self.fastbestvx = self.bestvx
                    self.fastbestA = np.array(self.bestA)
                    self.fastbestfbs = np.array(self.bestfbs)
                    self.fastbestbsusps = np.array(self.bestbsusps)
                    print '=== === improved opt size: {}'.format(sum(self.fastbestA))
                    print '=== === improved opt value: {}'.format(self.fastbestvx)

            brankscores = np.multiply(self.bestbsusps, self.bestfbs)
            A = self.bestA.nonzero()[0]
            self.fastlocalbest.append((self.bestvx, (A, brankscores)))
            'clear inflating or shaving best'
            self.bestvx = 0

        self.bestvx, self.bestA, self.bestfbs, self.bestbsusps = \
                    self.fastbestvx, self.fastbestA, \
                    self.fastbestfbs, self.fastbestbsusps
        return

    def kcoreshavingGreedy(self, ptype):
        'kcore degeneracy'
        import networkx as nx
        import networkx.algorithms.bipartite.matrix as nxbim
        from networkx.algorithms import bipartite
        self.nxg = nxbim.from_biadjacency_matrix(self.graph,
              create_using=nx.DiGraph(), edge_attribute='frequency')
        self.kcoredecomp = nx.core_number(self.nxg)
        degeneracy = max(self.kcoredecomp.values())
        #m, n = self.graph.shape
        rows, cols = bipartite.sets(self.nxg)
        m,n = len(rows), len(cols)
        self.corenumrows = np.zeros(m,dtype=int)
        self.corenumcols = np.zeros(n,dtype=int)
        for k, v in self.kcoredecomp.iteritems():
            if self.nxg.node[k]['bipartite'] == 0:
                self.corenumrows[k]=v
            else:
                self.corenumcols[k-m]=v
        self.kshellshave(ptype=ptype)
        #self.coreAdmp(ptype=ptype)
        return

    def coreAdmp(self, ptype):
        'Deviation from MIRROR PATTERN (dmp)'
        coreorder = np.argsort(self.corenumrows)[::-1]
        corerank = coreorder.argsort()+1
        outdegorder = np.argsort(self.woutdegrees)[::-1]
        outdegrank =  outdegorder.argsort()+1
        dmp = np.absolute( np.log2(outdegrank) - np.log2(corerank) )
        dmpbd = 0# math.log(1.1,2)
        A0=np.zeros(self.nU, dtype=int)
        A0[dmp>dmpbd]=1
        print '*** *** current size: {}'.format(sum(A0))
        self.qchop=False
        self.start(A0, ptype)
        self.greedyshaving(sampleB=False)
        return

    'shave the top-20 largest shell'
    def kshellshave(self, ptype):
        kshellcnt=np.bincount(self.corenumrows)
        topargkshellcnt = np.argsort(-kshellcnt)[0:20]
        kshecands = np.sort(topargkshellcnt)[::-1]
        #gradcorenrows = self.corenumrows
        self.kcorebestvx = 0
        maxstay = 20
        stay = 0
        for k in kshecands: #xrange(degeneracy, kcoremin, -1):
            A0idx = np.argwhere(self.corenumrows==k).flatten()
            if len(A0idx) == 0:
                continue
            print '*** *** processing {}-th shell, size: {} ...'.format(k,
                                                                        len(A0idx))
            #print '*** *** processing {} core, size: {} ...'.format(k,
            #                                                            len(A0idx))
            A0=np.zeros(self.nU, dtype=int)
            A0[A0idx]=1
            self.qchop=False
            self.start(A0, ptype=ptype)
            self.greedyshaving(sampleB=False)
            print ''
            print '*** *** current opt size: {}'.format(sum(self.bestA))
            print '*** *** current opt value: {}'.format(self.bestvx)
            if self.kcorebestvx < self.bestvx:
                self.kcorebestvx = self.bestvx #have improvement
                self.kcorebestA  = np.array(self.bestA)
                self.kcorebestfbs = np.array(self.bestfbs)
                self.kcorebestbsusps = np.array(self.bestbsusps)
                print '=== === improved opt size: {}'.format(
                    sum(self.kcorebestA))
                print '=== === improved opt value: {}'.format(
                    self.kcorebestvx)
            """
            elif self.kcorebestvx > self.bestvx:
                print '''*** *** stop greedy since obj decreses from {} to
                    {}'''.format(self.kcorebestvx, self.bestvx)
                break #stop loop when vx decressing
            else:
                stay += 1
                if stay>=maxstay:
                    print '*** *** stop since stay too long'
                    break
            """
            'clear shaving best'
            #self.bestvx, self.bestA[0:] = 0, 0
            self.bestvx = 0

        self.bestvx, self.bestA, self.bestfbs, self.bestbsusps = \
                self.kcorebestvx, self.kcorebestA, \
                self.kcorebestfbs, self.kcorebestbsusps

        return

    def drawObjectiveCurve(self, outfig):
        fig = plt.figure()
        plt.plot(self.vxs, '-')
        plt.title('The convergence curve of simulated anealing.')
        plt.xlabel('# of iterations')
        plt.ylabel('objective value')
        if outfig is not None:
            fig.savefig(outfig)
        return fig

    def drawAccRejcnts(self, outfig):
        fig = plt.figure()
        plt.plot(self.acccnt, )
        plt.plot(self.acccnt, 'k--', label='accept times')
        plt.plot(-1 * np.array(self.rejcnt), 'k:', label='Data length')
        legend = plt.legend(loc='upper center', shadow=True, fontsize='x-large')
        # Put a nicer background color on the legend.
        legend.get_frame().set_facecolor('#00FFCC')
        if outfig is not None:
            fig.savefig(outfig)
        return fig

def paramGridBruteforce(sigmas, coes):
    pg=[]
    for s in sigmas:
        for c in coes:
            pg.append((s,c))
    return pg

def evalTriDense(opt):
    A1=np.append(np.ones(1000,dtype=int), np.zeros(2000,dtype=int))
    A2=np.append(np.zeros(1000,dtype=int), np.ones(1000,dtype=int))
    A2=np.append(A2,np.zeros(1000,dtype=int))
    A3=np.append(np.zeros(2000,dtype=int), np.ones(1000,dtype=int))
    fbs1=opt.graphr[A1.nonzero()].sum(0).getA1()
    fbs2=opt.graphr[A2.nonzero()].sum(0).getA1()
    fbs3=opt.graphr[A3.nonzero()].sum(0).getA1()
    o1=opt.maxobjfunc(A1, fbs1)
    o2=opt.maxobjfunc(A2, fbs2)
    o3=opt.maxobjfunc(A3, fbs3)
    o=opt.maxobjfunc(np.ones(3000,dtype=int), opt.indegrees)
    bo = opt.maxobjfunc(opt.bestA, opt.bestfbs)
    print 'o1: {}'.format(o1)
    print 'o2: {}'.format(o2)
    print 'o3: {}'.format(o3)
    print 'o all: {}'.format(o)
    print 'best o: {}'.format(bo)
    return o1,o2,o3,o,bo

def beyonddensity(wmat, alg, ptype, qfun, b, epsilon, aggmethod,
                  sdrop=True, mbd=0.5, tsfile=None, tunit='s',
                  ratefile=None, fileidstartzero=True,
                  numSing=10, rbd='avg', shaving=True, inflating=False,
                  nblock=1):
    print 'initial...'
    if sci.sparse.issparse(wmat) is False and os.path.isfile(wmat):
        sm = loadedge2sm(wmat, csr_matrix, weighted=True,
                         idstartzero=fileidstartzero)
    else:
        sm = wmat.tocsr()
    inprop = 'Considering '
    if Ptype.freq in ptype:
        inprop += '+[involvement] '
    if Ptype.ts in ptype:
        assert(os.path.isfile(tsfile))
        inprop += '+[time stamps] '
    elif sdrop:
        inprop += '+[sudden drop]'
    else:
        tsfile=None
    if Ptype.rate in ptype:
        assert(os.path.isfile(ratefile))
        inprop += '+[rating i.e. # of stars] '
    else:
        ratefile = None
    print inprop

    opt = BeyondDensityOpt(sm, qfun=qfun, b=b, suspbd=epsilon,
                           aggmethod = aggmethod, sdrop=sdrop,mbd=mbd,
                           tsfile=tsfile, tunit=tunit, ratefile=ratefile,
                           fileidstartzero=fileidstartzero)
    opt.nbests=[]
    opt.nlocalbests=[] #mainly used for fastgreedy
    gsrows,gbscores,gbestvx = 0,0,0
    for k in xrange(nblock):
        start_time = time.time()
        if alg == 'greedy':
            n1, n2 = sm.shape
            if n1 + n2 > 1e4:
                print '[Warning] alg {} is slow for size {}x{}'\
                        .format(alg, n1, n2)
            A = np.ones(opt.nU,dtype=int)
            print 'initial start'
            opt.start(A, ptype=ptype)
            print 'greedy shaving algorithm ...'
            opt.greedyshaving(sampleB=False)
        elif alg == 'fastgreedy':
            print """alg: {}\n\t+ # of singlular vectors: {}\n\
            + truncated bd: {}\n\t+ Shaving:{} + inflating:{}\n"""\
                    .format(alg, numSing, rbd, shaving, inflating)
            print 'initial start'
            opt.initfastgreedy(ptype, numSing, rbd)
            print "::::Finish Init @ ", time.time() - start_time
            print 'fast greedy algorithm ...'
            opt.fastgreedy(inflating=inflating, shaving=shaving)
        elif alg == 'kcoregreedy':
            print 'alg: {}\n'.format(alg)
            opt.kcoreshavingGreedy(ptype=ptype)
        else:
            print 'No such algorithm: '+alg
            sys.exit(1)

        print "::::Finish Algorithm @ ", time.time() - start_time

        srows = opt.bestA.nonzero()[0]
        bscores = np.multiply(opt.bestfbs, opt.bestbsusps)
        #Brank = (-bscores).argsort()
        opt.nbests.append((opt.bestvx, (srows, bscores)))
        opt.nlocalbests.append(opt.fastlocalbest)
        gsrows, gbscores, gbestvx = (srows,bscores,opt.bestvx) \
                if gbestvx < opt.bestvx  else (gsrows, gbscores, gbestvx)
        if k < nblock-1:
            opt.removecurrentblock(srows)
        print 'global best size ', len(gsrows)
        print 'global best value ', gbestvx
    return (gsrows, gbscores), gbestvx, opt

if __name__=="__main__":
    print 'argv ', sys.argv
    if len(sys.argv)>1:
        testid = int(sys.argv[1])
    else:
        testid = 0
    datapath=home+'/Data/'
    testdatapath='./testdata/'
    respath='./testout/'

    print 'loading data ... ...'
    #test1
    #edgefnm= testdatapath+'bigraph.test'
    #umtsfnm = testdatapath+'usermsgts.test'
    #fileidstartzero = False
    if testid == 1:
        #test2
        './testdata/test_appuprate.dict'
        edgefnm= testdatapath+'test_userbeer.edgelist'#'test_appup.edgelist'
        umtsfnm = testdatapath+'test_userbeerts.dict'#'test_appupts.dict'
        tunit='s'
        umratefnm = testdatapath+'test_userbeerrate.dict' #'test_appuprate.dict'
        fileidstartzero = True
        sm = loadedge2sm(edgefnm, csr_matrix, weighted=True,
                         idstartzero=fileidstartzero)
        sm = sm.tocsr()
        suspbd = 0.0
        numSing = 10
        rbd = 'avg'
        qfuns=['exp']
        bs=[32]
        ptype=[Ptype.freq, Ptype.ts, Ptype.rate]
        sdrop=True
        nblock=2

    if testid == 0:
        '''
        M = genTriDenseBlock(1000, 1000, 1000, 500, 1000,1000, p1=0.8, alpha2=3,
                             alpha3=9.0)
        sm=csr_matrix(M)
        '''
        #A1,B1,A2,B2=500,500, 1500, 1500 #100,100,1500,1500 #
        #m = genDiDenseBlock(A1,B1,A2, B2, alpha=-1)
        #m=addnosie(m, A1+A2, B1, 0.005, black=True, A0=A1, B0=0)
        #m=addnosie(m, A1, B1+B2, 0.005, black=True, A0=0, B0=B1)
        #m=addnosie(m, A1+A2, B1+B2, 0.4, black=False)

        #m = genTriRectBlocks(3000,3000,0.6,0.6,0.6)
        #m = genDiHyperRectBlocks(50, 50, 2500, 2500, alpha=-0.5, tau=0.02)
        #m=addnosie(m, 2550, 50, 0.005, black=True, A0=50, B0=0)
        #m=addnosie(m, 50, 2550, 0.005, black=True, A0=0, B0=50)
        #m=addnosie(m, 2550, 2550, 0.005, black=True, A0=0, B0=0)
        A1,B1,A2,B2= 500,500, 2500, 2500 #100,100, 2500, 2500
        m = genDiHyperRectBlocks(A1, B1, A2, B2, alpha=-0.5, tau=0.002)
        m = addnosie(m, A1+A2, B1+B2, 0.01, black=True, A0=0, B0=0)
        m = addnosie(m, A1+A2, B1+B2, 0.4, black=False, A0=0, B0=0)
        #m[0:500][:,500:3000]=m[500:1000][:,500:3000] #camouflage
        #m = addnosie(m, 500, 3000, 0.99, black=False, A0=0, B0=500)
        sm = coo_matrix(m)
        sm = injectCliqueCamo(sm, 500, 500, p=0.6, testIdx=3)
        numSing = 10
        suspbd=0.0
        rbd = 'avg'
        qfuns=['exp']
        bs=[128]#[128]
        ptype=[Ptype.freq]
        sdrop=False
        umtsfnm, umratefnm, tunit, fileidstartzero = None, None, True, 's'
        nblock =2

    if testid == 3:
        edgefnm = datapath+'wbdata/usermsg.edgelist'
        umtsfnm = None
        umratefnm = None
        fileidstartzero=False
        sm = loadedge2sm(edgefnm, csc_matrix, weighted=True, idstartzero=fileidstartzero)
        sm = sm.tocsr()
        suspbd = 0.0
        numSing = 2
        qfuns=['exp']
        bs=[128]
        rbd = 4e-1
        ptype=[Ptype.freq]

    if testid == 4:
        edgefnm= datapath+'BeerAdvocate/userbeer.edgelist'
        umtsfnm = datapath+'BeerAdvocate/userbeerts.dict'
        umratefnm = datapath+'BeerAdvocate/userbeerrate.dict'
        fileidstartzero = True
        sm = loadedge2sm(edgefnm, csr_matrix, weighted=True,
                         idstartzero=fileidstartzero)
        sm = sm.tocsr()
        suspbd = 0.0
        numSing = 2
        rbd = 1e-4
        qfuns=['exp']
        bs=[128]
        ptype=[Ptype.freq, Ptype.rate, Ptype.ts]
        sdrop = True
        tunit='s'
        nblock=3

    '''
    [0,1e1, 5e1, 1e2, 5e2, 800.0,1000]
    [1e1, 5e1, 1e2, 2e2, 4e2, 6e2, 8e2, 1e3]
    np.array([1e1, 5e1, 1e2, 2e2])
    [0, 1e1, 5e1, 1e2, 2e2, 4e2 ]
    [0, 1e2, 2e2]
    '''
    #coes = [0] #[1, 1e1, 5e1, 1e2]#np.array([1e1, 5e1, 1e2])
    #paragrid=paramGridBruteforce(sigmas, coes)
    paragrid=paramGridBruteforce(qfuns, bs)
    figs =OrderedDict({})
    vxss =OrderedDict({})
    algs = ['greedy', 'fastgreedy', 'kcoregreedy']
    alg=algs[1]
    for qfun, b in  paragrid:
        bdres = beyonddensity(sm, alg, ptype, qfun, b, epsilon=suspbd,
                             aggmethod='joint', sdrop=sdrop,
                             tsfile=umtsfnm, tunit=tunit, ratefile=umratefnm,
                             fileidstartzero=fileidstartzero,
                             numSing=numSing, rbd=rbd, shaving=True,
                              inflating = False, nblock=nblock)
        opt = bdres[-1]
        T='alg{}q{}_b{}'.format(alg, opt.scale, opt.b)
        vxss[T]=opt.vxs
        figs[T] = opt.drawObjectiveCurve(respath+'curve'+T+'.eps')
        print '\n\toptimized A size: {}'.format(len(bdres[0][0]))
        print '\toptimized objec: {}'.format(bdres[1])

        continue #skip the following

        if alg == 'greedy':
            print """--processing no sigma, no coe, no qchopb,
                    qfun:{}, b:{}, suspbd:{} """.format(opt.scale, opt.b, opt.suspbd)
            A = np.ones(opt.nU,dtype=int)
            print 'initial start'
            opt.start(A, ptype=ptype)
            #opt.simulateAnealing()
            print 'greedy algorithm ...'
            opt.greedyshaving(sampleB=False)
        elif alg == 'fastgreedy':
            print """--processing no sigma, no coe, qfun:{}, b: {}, suspbd: {},
                roundbd: {}""".format(opt.scale, opt.b, opt.suspbd, rbd)
            print 'initial start'
            opt.initfastgreedy(ptype, numSing, rbd)
            print 'fast greedy algorithm ...'
            opt.fastgreedy(shaving=True)
        elif alg == 'kcoregreedy':
            opt.kcoreshavingGreedy(ptype=ptype)

        T='alg{}q{}_b{}'.format(alg, opt.scale, opt.b)
        vxss[T]=opt.vxs
        figs[T] = opt.drawObjectiveCurve(
            respath+'curvealg{}_b{}.eps'.format(alg, opt.scale, opt.b))
        '''
        fig = opt.drawObjectiveCurve(
            respath+'convergecurve_s{}c{}.png'.format(sigma, coe))
        #fig.show()
        fig2 = opt.drawAccRejcnts(
            respath+'acceptrejectcnts_s{}c{}.png'.format(sigma, coe))
        #fig2.show()
        '''
        print '\n\toptimized A size: {}'.format(np.sum(opt.bestA))
        print '\toptimized objec: {}'.format(opt.bestvx)

