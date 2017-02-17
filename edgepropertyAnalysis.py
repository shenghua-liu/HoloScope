import sys
sys.path.append('../')
import numpy as np
import scipy as sci
#import networkx as nx
#import scipy.sparse.linalg as slin
from collections import OrderedDict
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix, lil_matrix
from mytools.ioutil import loadedge2sm, saveSimpleListData, loadSimpleList
from mytools.drawutil import drawHexbin, drawTimeseries
import matplotlib.pyplot as plt
from gendenseblock import *
from os.path import expanduser
import math
from userdeltat import usermsgtsdic
from scipy import stats
from sklearn import preprocessing
import datetime
home = expanduser("~")

class MultiEedgePropBiGraph:
    def __init__(self, wadjm):
        self.wadjm = wadjm.tocsr().astype(np.float64) #weighted adjacent matrix
        self.nU , self.nV = wadjm.shape
        self.indegrees = self.wadjm.sum(0).getA1()
        self.inbd=2 # the objects that has at least 2 edges are considered
        'self.spv = {} #suspect users property vale '
        'self.apv = {} #property values for all sinks'
        'self.eprop =[] # the edge property list of PIM'
        """
        since the data is cut by the end of time, so we need to leave a twait
        to see if it is a sudden drop or cut by the end of time
        """
        self.twaits = {'s':12*3600, 'h':24, 'd':30, None:0}

    #@profile
    def load_from_edgeproperty(self, profnm, idstartzero, mtype=coo_matrix,
                               dtype=int):
        'load the graph edge property, time stamps, ratings, or text vector'
        self.idstartzero = idstartzero #record for output recovery
        offset = -1 if idstartzero is False else 0
        'sparse matrix has special meaning of 0, so property index start from 1'
        self.eprop = [np.array([])] #make the idx start from 1 in sparse matrix
        with open(profnm, 'rb') as fin:
            idx=1
            x,y,data=[],[],[]
            for line in fin:
                um, prop = line.strip().split(':')
                u, m = um.split('-')
                u = int(u)+offset
                m = int(m)+offset
                x.append(u)
                y.append(m)
                data.append(idx) #data store the index of edge properties
                prop = np.array(prop.strip().split()).astype(dtype)
                self.eprop.append(prop)
                idx += 1
            fin.close()
            self.edgeidxm = mtype((data, (x,y)), shape=(max(x)+1, max(y)+1))
            self.edgeidxmr = self.edgeidxm.tocsr()
            self.edgeidxmc = self.edgeidxm.tocsc()
            self.edgeidxml = self.edgeidxm.tolil()
            self.edgeidxmlt = self.edgeidxm.transpose().tolil()
            self.eprop = np.array(self.eprop)
        return

    def ratebins(self, propvals, btype):
        if not( max(propvals) == 5 and min(propvals)>=1 ):
            print 'Warning: preprocess to let rating between [1 5]'
            btype = 'org' #use original values as hist
        'bins: [v1, v2), [v2, v3), ..., [v_{n-1}, v_n]'
        bins=[]
        if btype == 'orig':
            bins=np.concatenate([propvals, [max(propvals)+1]])
        elif btype == 'int':
            s = math.floor(min(propvals))
            e = math.ceil(max(propvals)) + 1
            bins=np.arange(s,e).astype(int)
        elif btype == 'polar':
            '''for 1 to 5 score space, 1,1.5,2; 2.5,3,3.5; 4,4.5,5'''
            bins=[1,2.5,4,5.5]
        else:
            print '[Error] no ratebins type'
        return bins

    #@profile
    def setup_rate4all_sinks(self):
        '''set up the rating property for all sinks'''
        propvals=set() #vacabulary size or score space
        for vs in self.eprop:
            for v in set(vs):
                propvals.add(v)
        self.propvals = np.array(sorted(list(propvals)))
        'assume score is [1,5], and  arounding real scores into 3 catagories, '
        if not (max(propvals)==5 and min(propvals)>=1):
            print 'Warning: rating scores are not in [1,5]. They are [{}]'.\
                    format(', '.join(map(str, propvals)))

        '(1, 1.5, 2), (2.5, 3, 3.5), (4, 4.5, 5)'
        for i in xrange(len(self.eprop)):
            if min(propvals)<1:
                self.eprop[i] = np.digitize(self.eprop[i], bins=[0,2.5,4,5.01])-1
            else:
                self.eprop[i] = np.digitize(self.eprop[i], bins=[1,2.5,4,5.01])-1
        allmlt = self.edgeidxmlt #all susp msg matrix
        'effect sinks'
        cols = np.argwhere(self.indegrees>=self.inbd).flatten()
        self.inbdcolset = set(cols)
        apv = {}  #all property values
        ahists={} #all histograms of sinks
        amean, avar = np.zeros(self.nV, dtype=np.float64), \
                np.zeros(self.nV, dtype=np.float64)
        for i in cols:
            aidx = allmlt.data[i]
            apvi = np.concatenate(self.eprop[aidx]) #no np.sort
            apv[i]= apvi
            amean[i] = apvi.mean()
            avar[i] = apvi.var()
            ahists[i] = np.bincount(apvi, minlength=3)
        self.amean, self.avar = amean, avar
        self.ahists = ahists
        self.apv = apv
        return

    #@profile
    def setup_ts4all_sinks(self, tunit, bins='auto'):
        'calculate the one-time values for every sink, like bursting, dying, drop'
        maxts = [np.max(t) for t in self.eprop[1:]]
        self.endt = max(maxts)
        self.twait = self.twaits[tunit]
        allmlt = self.edgeidxmlt #all susp msg matrix
        'effect sinks'
        cols = np.argwhere(self.indegrees>=self.inbd).flatten()
        self.inbdcolset = set(cols)
        apv = {}
        awakburstpt, burstvals, burstslops, ainbursts={},{},{},{}
        dyingpt = {}
        dropslops, dropfalls=np.zeros(self.nV, dtype=np.float64), \
                np.zeros(self.nV, dtype=np.float64)
        amean, avar = np.zeros(self.nV, dtype=np.float64), \
                np.zeros(self.nV, dtype=np.float64)
        for i in cols:
            aidx = allmlt.data[i]
            aumts = np.concatenate(self.eprop[aidx]) #no sort
            apv[i]= aumts
            amean[i] = aumts.mean()
            avar[i] = aumts.var()
            'awaking bursting points and values, debugpt for debug'
            #abpts, bv = self.awakeburstpoints(aumts, bins=bins)
            abpts, bvs, slops, debugpt = awakburstpoints_recur(aumts, bins=bins)
            awakburstpt[i], burstvals[i], burstslops[i] =abpts, bvs, slops
            cnts=[]
            for abpt in abpts:
                '#of edges involve in bursting'
                left, right = abpt
                cnt = ((aumts>=left) & (aumts<=right)).sum()
                cnts.append(cnt)
            ainbursts[i]=np.array(cnts)
            dropfall, dropt, slop = \
                burstmaxdying_recur(aumts, endt=self.endt, twait=self.twait, bins=bins)
                #burstdyingpoints(aumts, endt=self.endt, twait=self.twait, bins=bins)
            dyingpt[i] = dropt
            dropslops[i], dropfalls[i] = slop, dropfall

        self.amean, self.avar = amean, avar
        self.apv = apv
        self.awakeburstpt, self.burstvals, self.burstslops, self.ainbursts = \
                awakburstpt, burstvals, burstslops, ainbursts
        self.dyingpt, self.dropslops, self.dropfalls = \
                dyingpt, dropslops, dropfalls
        return

    def load_from_userobjrates(self, uoratefn, idstartzero,
                               mtype=csc_matrix, dtype=float):
        'if inject load and setup need to be called separately'
        self.load_from_edgeproperty(uoratefn, idstartzero,
                                   mtype, dtype)
        self.setup_rate4all_sinks()
        return

    def load_from_usermsgtimes(self, umtsfn, tunit, idstartzero,
                               mtype=csc_matrix, dtype=int):
        self.load_from_edgeproperty(umtsfn, idstartzero=idstartzero,
                                   mtype=mtype, dtype=int)
        self.setup_ts4all_sinks(tunit=tunit)
        return

    'this is only called once, always put into the init/start func'
    #@profile
    def setupsuspects(self, users):
        self.suspuser = np.array(users)
        self.deltacols, self.delcols = [], set()
        if len(self.suspuser) ==0:
            self.spv = {}
            return
        #suspmlt = self.edgeidxmr[self.suspuser].transpose().tolil()
        suspmlt = self.edgeidxml[self.suspuser].transpose()
        colwsum = self.wadjm[self.suspuser].sum(0).getA1()
        cols = np.where(colwsum>= self.inbd)[0]
        cols = set(cols) & self.inbdcolset
        spv = {}
        for col in cols:
            spids = suspmlt.data[col]
            #property indices of suspect sink
            #only consider those objects have more than inbd edges with suspusers
            sumts = np.concatenate(self.eprop[spids])
            spv[col]=sumts
        self.spv = spv
        return

    'must be effecient, shared among rating, ts, text'
    #@profile
    def deltasuspects(self, z, yusers, add=True):
        self.suspuser = yusers
        zmat = self.edgeidxmr[z]
        cols = zmat.nonzero()[1]
        deltacols, delcols = [], set()
        i = -1
        for col in cols:
            i += 1
            if col not in self.inbdcolset:
                continue
            spid = zmat.data[i]
            if add:
                self.spv[col]= np.concatenate((self.spv[col],self.eprop[spid]))\
                        if col in self.spv else self.eprop[spid]
                deltacols.append(col)
            else:
                if col not in self.spv:
                    continue #donot added in the initial
                #minus
                self.spv[col] = list(self.spv[col])
                for e in self.eprop[spid]:
                    self.spv[col].remove(e)
                if len(self.spv[col])==0:
                    self.spv.pop(col, None)
                    delcols.add(col)
                else:
                    deltacols.append(col)

        self.deltacols, self.delcols = deltacols, delcols
        return

    def setupsuspects_opt1(self, users):
        'todo: incremental, need to change sumidx2cols structure'
        self.suspuser = np.array(users)
        suspmlt = self.edgeidxmr[self.suspuser].transpose().tolil() #susp matrix
        spv = {}
        col=-1
        for spids in suspmlt.data:
            col+=1
            #property indices of suspect sink
            if len(spids)>self.inbd:
                #only consider those objects have more than inbd edges with suspusers
                sumts = np.concatenate(self.eprop[spids])
                spv[col]=sumts
        self.spv = spv
        return

    def awakeburstpoints(self, ts, bins='auto'):
        '''
            Calculates the awaking and bursting points of a time series
            There may be multiple pair of awaking and bursting points,
            considering the multiple bursting cases.
            bins: is the same as numpy.histogram
        '''
        abpts=[]
        bvals=[]#record the bursting point value
        abptidxs=[]
        hts = np.histogram(ts, bins=bins)
        ys = np.append([0], hts[0]) #add zero, so 0 is allocated to lowest left bound
        ys = ys.astype(np.float64)
        xs = hts[1]
        import Queue
        maxidxs = Queue.Queue()
        maxidxs.put(np.argmax(ys))#initial#np.argwhere(ys==max(ys)).flatten()
        startidx = 0 #start idx of the line
        while not maxidxs.empty():
            maxidx = maxidxs.get()
            x0,y0,xm,ym=xs[startidx], ys[startidx], xs[maxidx], ys[maxidx]
            sqco = math.sqrt((ym-y0)**2 + (xm-x0)**2) #sqrt of coefficient
            '''
            dta = 0
            ta=t0dix #waking pt at the beginning
            for i in xrange(startidx, maxidx):
                x,y = xs[i],ys[i]
                dt = ((ym-y0)*x - (xm-x0)*y - ym*x0 + xm*y0)/sqco
            '''
            xvec, yvec=xs[startidx:maxidx], ys[startidx:maxidx]
            'if pt (x,y) at the above of the line, the distance is negative'
            dts = ((ym-y0)*xvec - (xm-x0)*yvec + (xm*y0 - ym*x0))/sqco
            xaidx = np.argmax(dts) + startidx #recover to global idx
            xa = xs[max(xaidx-1, 0)] #since we add 0 to ys, find the left bound
            abpts.append((xa, xm)) #save results
            bvals.append(ys[maxidx])
            abptidxs.append((xaidx, maxidx))
            'locate the next bursting and starting earlier than awaking'
            diffyincrese = np.argwhere(np.diff(ys[maxidx:]) >0)
            if len(diffyincrese) > 0:
                turningptidx = diffyincrese[0,0]+maxidx
                ntmaxidx = np.argmax(ys[turningptidx:]) + turningptidx #global idx
                '''using the minimum point between turning pt and next max pt
                   warning: this may skip a burst peak between turningpt and
                   local minimum pt
                '''
                startidx = np.argmin(ys[turningptidx:ntmaxidx]) + turningptidx
                maxidxs.put(ntmaxidx)
            else:
                break
        return np.array(abpts), np.array(bvals)

    #@profile
    def suspratedivergence(self, neutral=False, delta=False):
        '''calculate the diverse of ratings betwee A and U\A
           scaling=False
        '''
        if delta:
            cols, delcols = self.deltacols, self.delcols
            ratediv = self.ratediv
            if len(self.spv) < 1:
                self.ratediv[0:]=0.0
                return self.ratediv
        else:
            cols, delcols = self.spv.keys(), set()
            ratediv =np.zeros(self.nV, dtype=float)
            self.maxratediv = 0

        #bal = np.zeros(self.nV) 
        for col in cols:
            if col in delcols:
                assert(col not in self.spv)
                ratediv[col] = 0
                continue
            rs = self.spv[col] #stophere
            shis=np.bincount(rs, minlength=3)
            ahis = self.ahists[col]
            ohis=ahis-shis
            shis, ohis = shis+1, ohis+1 #a kind of multinomial prior
            if neutral is False:
                'remove netrual 2.5, 3, 3.5'
                shis[1], ohis[1] = 0, 0
            #cal KL-divergence
            kl = stats.entropy(shis, ohis)
            lenrs = len(rs)
            lenars = len(self.apv[col])
            ssum, osum = float(lenrs)+1, float(lenars-lenrs)+1
            #bal[col]=(min(ssum/osum, osum/ssum))
            bal = (min(ssum/osum, osum/ssum))
            ratediv[col]=kl*bal #optimal tune density:0.1 ==> 0.91 F-measure
            self.maxratediv = kl if self.maxratediv < kl else self.maxratediv
        #if scaling:
        #    ratediv = preprocessing.minmax_scale(ratediv, copy=False)
        #self.ratediv = np.multiply(bal, ratediv)
        self.ratediv = ratediv
        return self.ratediv

    'concrete center distance'
    def suspuserccdist(self):
        ccdist=[]
        mccdist=[] #average, mean
        smean, svar = [],[] #all mean/var
        for col, sumts in self.spv.iteritems():
            smean.append(sumts.mean())
            svar.append(sumts.var())
            cc = self.amean[col] #concrete center
            ccd = 0
            for ts in sumts:
                ccd += abs(ts-cc)
            ccdist.append(ccd)
            mccdist.append(ccd/float(len(sumts)))

        self.ccdist, self.mccdist = \
                np.array(ccdist), np.array(mccdist)
        self.smean, self.svar = \
                np.array(smean), np.array(svar)
        return

    def setupsuspmsg(self, msg):
        if msg is not None:
            self.suspmsg = np.array(msg)
        else:
            'heuristically choose suspect msg'
            suspinvolv = self.wadjm[self.suspuser].sum(0).getA1()
            suspmsgbd=50
            self.suspmsg = np.argwhere(suspinvolv>=suspmsgbd).flatten()
        return

    'for testing'
    def run_tsccdistscore_eval(self, msg):
        suspwm = self.wadjm[self.suspuser].tocsc()
        colsum = suspwm.sum(0).getA1()
        cols = self.spv.keys() #effective cols for current susp users
        self.involvratios = np.divide(colsum[cols], self.indegrees[cols])
        self.involvcnts = colsum[cols]
        self.setupsuspmsg(msg, idstartzero)
        self.suspuserccdist()
        #record the idx in mx/y vx/y that belongs to suspect msgs
        self.suspmsgidx =np.in1d(cols, self.suspmsg).nonzero()[0]
        return

    #@profile
    def suspburstinvolv(self, multiburstbd=0.5, weighted=True, delta=False):
        '''calc how many points allocated in awake and burst period, over total
           number of U who involv in the burst
        '''
        if delta:
            cols, delcols = self.deltacols, self.delcols
            inburstcnt, inburstratio = self.inburstcnt, self.inburstratio
        else:
            inburstcnt, inburstratio = \
                    np.zeros(self.nV, dtype=int), np.zeros(self.nV, dtype=float)
            cols, delcols = self.spv.keys(), set()

        for col in cols:
            if col in delcols:
                assert(col not in self.spv)
                inburstcnt[col], inburstratio[col] = 0, 0.0
                continue
            st = self.spv[col]
            abpts, bvs, slops, ainburst = self.awakeburstpt[col], \
                    self.burstvals[col], self.burstslops[col], self.ainbursts[col]
            'get the satisfied multiburst points'
            burstids = bvs/float(bvs[0]) >=  multiburstbd
            abpts, slops, bvs, ainburst = abpts[burstids], slops[burstids],\
                    bvs[burstids], ainburst[burstids]
            scnt, wscnt, wallcnt =0, 0, 0
            #for (left, right), sp, acnt in \
            #             zip(abpts[burstids], slops[burstids], ainburst[burstids]):
            for i in xrange(len(abpts)):
                (left, right), sp, bv, acnt = abpts[i],slops[i], bvs[i], ainburst[i]
                '#susp users in burst'
                cnt1 = ((st >= left) & (st <= right)).sum()
                scnt += cnt1
                '#all users in burst'
                assert(acnt>=cnt1)
                if weighted is not False:
                    wscnt +=  cnt1 * sp * bv
                    wallcnt += acnt * sp * bv
                else:
                    wscnt += cnt1
                    wallcnt += acnt
            inburstcnt[col]=scnt
            inburstratio[col] = wscnt/float(wallcnt)

        self.inburstcnt = inburstcnt
        self.inburstratio =inburstratio
        return self.inburstcnt, self.inburstratio

    def suspburstinvolv_self(self, multiburstbd=0.5, weighted=True):
        '''calc how many points allocated in awake and burst period, over total
           number of A
        '''
        inburstcnt=[]
        inburstratio=[]
        maxslop = 0.0
        for col, st in self.spv.iteritems():
            abpts, bvs, slops = self.awakeburstpt[col], \
                    self.burstvals[col], self.burstslops[col]
            'get the satisfied multiburst points'
            burstids = bvs/float(bvs[0]) >=  multiburstbd
            incnt, wincnt =0, 0
            for (left, right), sp in zip(abpts[burstids], slops[burstids]):
                cnt = np.multiply(st >= left, st <= right).sum()
                incnt += cnt
                wincnt += cnt * sp
                maxslop = max(sp, maxslop)
            inburstcnt.append(incnt)
            if weighted is not False:
                inburstratio.append(wincnt/float(len(st)))
            else:
                inburstratio.append(incnt/float(len(st)))

        if weighted is True or weighted == 'max':
            inburstratio = np.array(inburstratio, dtype=float)/maxslop #normalized by max slop
        elif weighted == 'minmax':
            inburstratio = preprocessing.minmax_scale(inburstratio, copy=False)
        self.inburstcnt = np.array(inburstcnt, dtype=float)
        self.inburstratio =inburstratio

        return self.inburstcnt, self.inburstratio

    def showInvolvCCdist(self, outfignm=None, involvcnt=True,
                         meandist=True, gridsize=100):
        '''
            The heatmap of suspect involving ratios (may times involve cnt)
            vs edge time distance from concrete center as paper said
        '''
        involv = np.multiply(self.involvratios, self.involvcnts) if involvcnt \
                else self.involvratios

        ccdist= self.mccdist if meandist else self.ccdist
        suspmsgidx = self.suspmsgidx  #record the idx in mx/y vx/y that belongs to suspect msgs
        fig = drawHexbin(involv,ccdist, outfig=None, xscale='log', yscale='log',
                          colorscale=True, suptitle='cc distance vs suspect involvment',
                          gridsize=gridsize)
        plt.loglog(involv[suspmsgidx], ccdist[suspmsgidx], 'ks', markersize=10,
                   markerfacecolor='none', markeredgecolor='k')
        if involvcnt:
            plt.xlabel('involvlement of suspect users')
        else:
            plt.xlabel('involving ratios of suspect users')
        plt.ylabel('distance to concrete center')
        if outfignm is not None:
            fig.savefig(outfignm)
        return fig, [involv, ccdist]

    def showsuspMeanVar(self, allmsg=True, outfigmean=None,
                        outfigvar=None, gridsize=100):
        suspm = self.edgeidxmr[self.suspuser].tocsc() #susp matrix
        allm = self.edgeidxmc #all susp msg matrix
        if allmsg is False:
            suspm = suspm[:,self.suspmsg]
            allm = self.edgeidxmc[:,self.suspmsg]
        mx,my, vx, vy=[],[],[],[]
        if self.inbd < 2:
            print 'warning: the bound of involving by suspect users must >= 2'
            vxyscale = 'linear'
        else:
            vxyscale = 'log'
        suspwm = self.wadjm[self.suspuser].tocsc()
        colsum = suspwm.sum(0).getA1()
        cols = np.argwhere(colsum>=self.inbd).flatten() #suspsuers involved in the msgs
        suspmsgidx, idx = [], 0 #record the idx in mx/y vx/y that belongs to suspect msgs
        for i in cols:
            xidx = suspm[:,i].data
            yidx = allm[:,i].data
            xdata = np.concatenate(self.eprop[xidx])
            ydata = np.concatenate(self.eprop[yidx])
            mx.append(xdata.mean())
            my.append(ydata.mean())
            vx.append(xdata.var())
            vy.append(ydata.var())
            if i in self.suspmsg:
                suspmsgidx.append(idx)
            idx += 1
        mx, my, vx, vy = np.array(mx), np.array(my),np.array(vx),np.array(vy)
        fig1 = drawHexbin(mx,my, outfig=None, xscale='log', yscale='log',
                          colorscale=True, suptitle='suspect msg means',
                          gridsize=gridsize)
        if allmsg:
            plt.loglog(mx[suspmsgidx], my[suspmsgidx], 'ks', markersize=10,
                       markerfacecolor='none', markeredgecolor='k')
        plt.xlabel('suspect users')
        plt.ylabel('all users')
        if outfigmean is not None:
            fig1.savefig(outfigmean)

        vxg0=vx>0
        fig2 = drawHexbin(vx[vxg0],vy[vxg0], outfig=None, xscale=vxyscale, yscale=vxyscale,
                          colorscale=True, suptitle='suspect msg variences',
                         gridsize=gridsize)
        if allmsg:
            suspvxg0 = vx[suspmsgidx]>0
            plt.loglog(vx[suspmsgidx][suspvxg0], vy[suspmsgidx][suspvxg0], 'ks', markersize=10,
                       markerfacecolor='none', markeredgecolor='k')
        plt.xlabel('suspect users')
        plt.ylabel('all users')
        if outfigvar is not None:
            fig2.savefig(outfigvar)

        '''
        fig = plt.figure(1)
        plt.subplot(121)
        plt.scatter(mx, my, marker='.')
        plt.xlabel('suspect users')
        plt.ylabel('all users')
        plt.title('suspect msgs means')
        plt.subplot(122)
        plt.scatter(vx, vy, marker='.')
        plt.xlabel('suspect users')
        plt.ylabel('all users')
        plt.title('suspect msgs variance')
        '''
        return [fig1, fig2],[mx,my,vx,vy, cols[vx==0]]

    def showslopvsindegrees(self, sloptype='burst', weighted=True, showsuspmsg=True,
                                 outfnm=None):
        cols = self.indegrees>=self.inbd
        indegrees = self.indegrees[cols]
        slops, ws, wslops, suspmsglocalids = [],[],[],[]
        if sloptype == 'burst':
            pstr = 'max bursting'
            for s in self.burstslops.values():
                slops.append(s[0])
            slops = np.array(slops)
            for val in self.burstvals.values():
                ws.append(val[0])
            ws = np.array(ws)
        elif sloptype == 'drop':
            pstr = 'dropping'
            slops = self.dropslops[cols]
            ws = self.dropfalls[cols]
        if showsuspmsg:
            msgbinary  = np.zeros(self.nV)
            msgbinary[self.suspmsg]=1
            suspmsglocalids = msgbinary[cols].nonzero()[0]
        if weighted:
            wslops = np.multiply(ws, slops)
            fig = showgeneralheatmap(indegrees, wslops, suspmsglocalids)
            plt.ylabel('weighted {} slop'.format(pstr))
            plt.title('Weighted {} slops v.s. indegrees of messages'.format(pstr))
        else:
            fig = showgeneralheatmap(indegrees, slops, suspmsglocalids)
            plt.ylabel('{} slop'.format(pstr))
            plt.title('{} slops v.s. indegrees of messages'.format(pstr))
        plt.xlabel('indegree')
        if outfnm is not None:
            fig.savefig(outfnm)
        gwslops = np.zeros(self.nV)
        gwslops[cols] = wslops
        return fig, (cols.nonzero()[0], suspmsglocalids, gwslops)

#@profile
def awakburstpoints_recur(ts, bins='auto'):
    'recursive version'
    hts = np.histogram(ts, bins=bins)
    ys = np.append([0], hts[0]) #add zero, so 0 is allocated to lowest left bound
    ys = ys.astype(np.float64)
    xs = hts[1]
    abptidxs = []
    startidx = 0
    'recursively get the idx for awake and burst pts'
    recurFindAwakePt(xs, ys, start=startidx, abptidxs=abptidxs)
    if len(abptidxs)==0:
        return [], [0], [0], None
    'extend left bound by -1, since we added zero to histogram'
    abptextidxs, bvsrt, slops = sort_extendLeftbd(abptidxs, xs, ys)
    'convert abptext idx to bd value in xs'
    abpts = np.array([(xs[l], xs[r]) for l, r in abptextidxs])
    return abpts, bvsrt, slops, [abptidxs, abptextidxs]

#@profile
def sort_extendLeftbd(abptidxs, xs, ys):
    'sort bds by burst val, and extend the left bound of sorted awakeburst pts'
    bv=[ ys[r]-ys[l] for l, r in abptidxs] #use abdiff as bv
    abptys = sorted(zip(abptidxs, bv), key=lambda x:x[1], reverse=True)
    abptsrt, bvsrt = zip(*abptys)
    abptsrt = np.array(abptsrt)
    bvsrt = np.array(bvsrt)
    'calculate slop of bursting before extending'
    slops, diffs = [], []
    for l, r in abptsrt:
        slop = (ys[r]-ys[l])/float(xs[r]-xs[l])
        slops.append(slop)
        #diffs.append(ys[r]-ys[l])
    slops = np.array(slops)
    #diffs = np.array(slops)
    'extend left, if overlep keep that of higher burst val'
    for i in xrange(len(abptsrt)):
        nl, nr = max(abptsrt[i][0]-1,0), abptsrt[i][1]
        for j in xrange(i):
            pl, pr = abptsrt[j][0], abptsrt[j][1]
            if nr >= pr and nl < pr:
                nl = pr
            if nl <= pl and nr > pl:
                print '[Warning] extended a impossible bound'
                nr = pl #impossible case, recurFindAwakePt guarantees that
        abptsrt[i][0], abptsrt[i][1]=nl,nr #extend
    return abptsrt, bvsrt, slops

#@profile
def recurFindAwakePt(xs, ys, start=0, abptidxs=[]):
    if len(ys)<=1 or len(xs)<=1:
        return
    maxidx = np.argmax(ys)
    x0,y0,xm,ym = xs[0], ys[0], xs[maxidx], ys[maxidx]
    sqco = math.sqrt((ym-y0)**2 + (xm-x0)**2) #sqrt of coefficient
    xvec, yvec = xs[:maxidx], ys[:maxidx]
    dts = ((ym-y0)*xvec - (xm-x0)*yvec + (xm*y0 - ym*x0))/sqco
    xaidx = np.argmax(dts)
    abptidxs.append((xaidx+start, maxidx+start))
    'left'
    recurFindAwakePt(xs[:xaidx], ys[:xaidx], start=start, abptidxs=abptidxs)
    'right'
    diffyincrese = np.argwhere(np.diff(ys[maxidx:]) >0)
    if len(diffyincrese) > 0:
        turningptidx = diffyincrese[0,0]+maxidx
        recurFindAwakePt(xs[turningptidx:], ys[turningptidx:],
                         start = turningptidx + start,
                         abptidxs=abptidxs)
    ''' #opt2
    l = len(ys)-maxidx
    if l >= 3:
        for idx in xrange(1,l-1):
            if ys[maxidx+idx] < ys[maxidx+idx+1]:
                'current turning pt starts from maxid'
                turningptidx = idx+maxidx
                #ntmaxidx = np.argmax(ys[turningptidx:]) + turningptidx #global idx
                #'using the minimum point between turning pt and next max pt'
                #startidx = np.argmin(ys[turningptidx:ntmaxidx]) + turningptidx
                recurFindAwakePt(xs[turningptidx:], ys[turningptidx:],
                                 start = turningptidx + start,
                                 abptidxs=abptidxs)
                break
    '''
    return

#@profile
def burstdyingpoints(ts, endt, twait=12*3600, bins='auto'):
    'endt is used to judge if the dying is caused by observation window'
    hts = np.histogram(ts, bins=bins)
    xs = hts[1]
    ys = hts[0].astype(np.float64)
    lenys = len(ys)
    if  lenys < 2:
        return 0, xs[0], 0
    maxts = max(ts)
    if maxts < endt - twait:
        ys = np.concatenate((ys, [0.0]))
    else:
        #hadd = stats.mode(ys)[0][0]
        hadd = (ys[-1]+ys[-2])/2.0
        ys = np.concatenate((ys, [hadd]))
    lenys += 1
    #xs = np.append(xs, 2*xs[-1]-xs[-2])#move one time bins
    #burstidx = np.argwhere(ys==max(ys))[0,-1] #this is slow the last maximum
    burstidx = lenys - np.argmax(ys[::-1]) -1 #the last max occurrence
    #burstidx = np.argmax(ys)
    if burstidx == lenys-1: #bursting at the end
        dyingidx = lenys-1
        slop = (ys[burstidx] - 0)/float(xs[-1]-xs[-2])
        fall = ys[burstidx]
    else:
        xm, ym, xe, ye = xs[burstidx], ys[burstidx], xs[-1], ys[-1]
        sqco = math.sqrt((ym-ye)**2 + (xm-xe)**2) #sqrt of coefficient
        xvec, yvec = xs[burstidx+1:], ys[burstidx+1:]
        dts = -((ym-ye)*xvec - (xm-xe)*yvec + (xm*ye - ym*xe))/sqco
        dts = np.absolute(dts) #use abs value
        dyingidx = len(dts)-np.argmax(dts[::-1])-1 + burstidx+1
        slop = (ym - ys[dyingidx])/float(xs[dyingidx] - xm)#dyingidx alwasy >0
        if dyingidx == lenys -1:
            fall = ym # assume continue to fall to 0, keeping current slop
        else:
            #fall = ym-ys[dyingidx]
            fall = ym - ys[-1] #approximation of dying fall
    return fall, xs[dyingidx], slop #, (burstidx,dyingidx)

def burstmaxdying_recur(ts, endt, twait=12*3600, bins='auto'):
    'endt is used to judge if the dying is caused by observation window'
    hts = np.histogram(ts, bins=bins)
    xs = hts[1]
    ys = hts[0].astype(np.float64)
    if  len(ys) < 2:
        return 0, xs[0], 0
    maxts = max(ts)
    if maxts < endt - twait:
        ys = np.concatenate((ys, [0.0]))
    else:
        #hadd = stats.mode(ys)[0][0]
        hadd = (ys[-1]+ys[-2])/2.0
        ys = np.concatenate((ys, [hadd]))

    maxdying=[0.0, 0.0, 0.0]
    recurFindMaxFallDying(xs, ys, maxdying)
    return maxdying

def recurFindMaxFallDying(xs, ys, maxdying):
    lenys = len(ys)
    if lenys < 2:
        return
    #xs = np.append(xs, 2*xs[-1]-xs[-2])#move one time bins
    #burstidx = np.argwhere(ys==max(ys))[0,-1] #this is slow the last maximum
    burstidx = lenys - np.argmax(ys[::-1]) -1 #the last max occurrence
    if ys[burstidx]-min(ys) < maxdying[0]:
        return
    #burstidx = np.argmax(ys)
    if burstidx == lenys-1: #bursting at the end
        dyingidx = lenys-1
        slop = (ys[burstidx] - 0)/float(xs[-1]-xs[-2])
        fall = ys[burstidx]
    else:
        xm, ym, xe, ye = xs[burstidx], ys[burstidx], xs[-1], ys[-1]
        sqco = math.sqrt((ym-ye)**2 + (xm-xe)**2) #sqrt of coefficient
        xvec, yvec = xs[burstidx+1:], ys[burstidx+1:]
        dts = -((ym-ye)*xvec - (xm-xe)*yvec + (xm*ye - ym*xe))/sqco
        dts = np.absolute(dts)
        dyingidx = len(dts)-np.argmax(dts[::-1])-1 + burstidx+1
        slop = (ym - ys[dyingidx])/float(xs[dyingidx] - xm)#dyingidx alwasy >0
        if dyingidx == lenys -1:
            fall = ym # assume continue to fall to 0, keeping current slop
        else:
            fall = ym-ys[dyingidx]
    if fall > maxdying[0]:
        maxdying[0:3] = [fall, xs[dyingidx], slop]

    if dyingidx < lenys-1:
        'move to the right'
        subburstidx = np.argmax(ys[dyingidx:]) + dyingidx
        recurFindMaxFallDying(xs[subburstidx:], ys[subburstidx:], maxdying)
    if burstidx > 1:
        'move to the left'
        subdyingidx = np.argmin(ys[:burstidx])
        recurFindMaxFallDying(xs[:subdyingidx], ys[:subdyingidx], maxdying)
    return

def pim2tensorformat(tsfile, ratefile, tensorfile, tunit='s', tbins='h'):
    'convert the pim files: tsfile, ratefile into tensor file, i.e. tuples'
    rbins = lambda x: 0 if x<2.5 else 1 if x<=3.5 else 2 #lambda x: x 
    propdict = {}
    with open(tsfile, 'rb') as fts, open(ratefile, 'rb') as frt,\
            open(tensorfile, 'wb') as fte:
        for line in fts:
            k,v = line.strip().split(':')
            propdict[k]=[v]
        for line in frt:
            k,v=line.strip().split(':')
            propdict[k].append(v)
        for k, vs in propdict.iteritems():
            u, b = k.strip().split('-')
            tss = vs[0].strip().split(' ')
            tss = map(int, tss)
            if tunit == 's':
                'time unit is second'
                if tbins == 'h':
                    'time bin size is hour'
                    tss = np.array(tss, dtype=int)/3600
                elif tbins == 'd':
                    'time bin size is day'
                    tss = np.array(tss, dtype=int)/(3600*24)
            'no matter what the tunit is'
            if type(tbins) is int:
                tss = np.array(tss, dtype=int)/tbins
            tss = map(str, tss)
            'process ts'
            rts = vs[1].strip().split(' ')
            rts = map(float, rts)
            digrs = []
            for r1 in rts:
                r = rbins(r1)
                digrs.append(r)
            digrs = map(int, digrs)
            digrs = map(str, digrs)
            for i in xrange(len(tss)):
                fte.write(','.join((u, b, tss[i], digrs[i], '1')))
                fte.write('\n')
        fts.close()
        frt.close()
        fte.close()
    return

def tspim2tensorformat(tsfile, tensorfile, tunit='s', tbins='h',
                       idstartzero=False):
    offset = 0 if idstartzero else -1
    propdict = {}
    with open(tsfile, 'rb') as fts, open(tensorfile, 'wb') as fte:
        for line in fts:
            k,v = line.strip().split(':')
            propdict[k]=[v]
        for k, vs in propdict.iteritems():
            u, b = k.strip().split('-')
            if idstartzero is False:
                u = str(int(u)+offset)
                b = str(int(b)+offset)
            tss = vs[0].strip().split(' ')
            tss = map(int, tss)
            if tunit == 's':
                'time unit is second'
                if tbins == 'h':
                    'time bin size is hour'
                    tss = np.array(tss, dtype=int)/3600
                elif tbins == 'd':
                    'time bin size is day'
                    tss = np.array(tss, dtype=int)/(3600*24)
            if type(tbins) is int:
                tss = np.array(tss, dtype=int)/tbins
            tss = map(str, tss)
            for i in xrange(len(tss)):
                fte.write(','.join((u, b, tss[i], '1')))
                fte.write('\n')
        fts.close()
        fte.close()
    return

def showgeneralheatmap(xs, ys, idx, outfignm=None, gridsize=100, xscale='log',
                      yscale='log'):
    fig = drawHexbin(xs, ys, outfig=None, xscale=xscale, yscale=yscale,
                      colorscale=True, suptitle='general heatmap for discovery',
                      gridsize=gridsize)
    if xscale == 'log' and yscale == 'log':
        plt.loglog(xs[idx], ys[idx], 'ks', markersize=10,
                   markerfacecolor='none', markeredgecolor='k')
    elif xscale == 'linear' and yscale == 'linear':
        plt.plot(xs[idx], ys[idx], 'ks', markersize=10,
                   markerfacecolor='none', markeredgecolor='k')
    plt.xlabel('general x')
    plt.ylabel('general y')
    if outfignm is not None:
        fig.savefig(outfignm)
    return fig

if __name__=="__main__":
    datapath=home+'/Data/'

    indata = datapath+'wbdata/usermsg.edgelist'
    usermsgts = datapath+'wbdata/usermsgtsdict.dat'
    tunit='s'
    idstartzero = False
    respath = datapath+'wbdata/results/'
    #case fraudar
    suspuserfn = datapath+'wbdata/fraudardenseblocklog0.rows' #'wbdata/fraudardenseblockeven0.rows'
    suspmsgfn = datapath+ 'wbdata/fraudardenseblocklog0.cols' #'wbdata/fraudardenseblockeven0.cols'
    suspidstartzero = True
    '''
    datapath=datapath+'BeerAdvocate/'
    indata = datapath+'userbeer.edgelist.inject0'
    usermsgts = datapath+'userbeerts.dict.inject0'
    idstartzero=True
    respath=datapath+'results/'
    suspuserfn = datapath + 'userbeer.edgelist.trueA0'
    suspmsgfn = datapath + 'userbeer.edgelist.trueB0'
    suspidstartzero = True
    ''' 

    '''#case short horizontal bar with name patterns
    suspuserfn = datapath+'wbdata/suspicious_users.useridx'
    suspmsgfn = None #datapath+'wbdata/'
    suspidstartzero = False #False #suspicious_users.useridx start from 1
    '''


    '''#case CR, cost-gain
    suspuserfn = datapath+'wbdata/involratiofastgreedy.rows'
    #involratiofastgreedy.cols
    suspmsgfn = datapath+'wbdata/involvfastgreedyge100.cols'
    suspidstartzero = True
    '''

    '''#test cases
    testdatapath='./testdata/'
    indata = './testdata/bigraph.test'
    usermsgts = './testdata/usermsgts.test'
    suspuserfn = './testdata/suspuser.test'
    suspmsgfn = './testdata/suspmsg.test'
    respath='./testout/'
    suspidstartzero = False
    '''

    print 'load weighted ajacent matrix'
    wsm = loadedge2sm(indata, csr_matrix, issquared=False, weighted=True,
                      dtype=int, idstartzero=idstartzero)
    '''#preprocess
    procdata = datapath+'wbdata/wb201311proc.dat'
    numusers= 2748001
    umdic=usermsgtsdic(procdata, numusers, usermsgts, delthreshold=0)
    '''
    #mg = nx.from_scipy_sparse_matrix(ssm,
    #                           create_using=nx.MultiDiGraph(),edge_attribute='freq')
    print 'create and load multi edge property bipartite graph ... ...'
    mpg = MultiEedgePropBiGraph(wadjm = wsm)
    mpg.load_from_usermsgtimes(usermsgts, tunit=tunit, idstartzero=idstartzero)
    startts = 1383192000 #2013-10-31 UTC-4
    #print 'it is ' + datetime.datetime.fromtimestamp(tmin).strftime('%Y-%m-%d %H:%M:%S')

    print 'load suspect users and msgs ... ...'
    suspuser = loadSimpleList(suspuserfn, dtype=int)
    suspuser = np.array(suspuser)
    suspuser = suspuser if suspidstartzero else suspuser-1
    if suspmsgfn is not None:
        suspmsg = loadSimpleList(suspmsgfn, dtype=int)
        suspmsg = np.array(suspmsg)
        suspmsg = suspmsg if suspidstartzero else suspmsg-1
    else:
        suspmsg = None
    'result from fraudar, cr, fastgreedy, the id starts from zero'
    mpg.setupsuspects(suspuser)
    mpg.setupsuspmsg(suspmsg)
    #mpg.run_tsccdistscore_eval(suspmsg)
    '''
    print 'showing suspect mean and variance ... ...'
    figs,points = mpg.showsuspMeanVar(allmsg=True,
                        outfigmean = respath+'suspect_mean_heatmap.png',
                        outfigvar = respath+'suspect_var_heatmap.png')
    figs[0].show()
    figs[1].show()
    '''
    '''
    print 'showing suspect involvement & distance to center'
    fig, points = mpg.showInvolvCCdist( outfignm=respath+'suspect_involv_ccdist_heatmap.png',
                                    involvcnt=True, gridsize=100, meandist=True)
    fig.show()
    '''
    print 'showing bursting slops'
    fig, res = mpg.showslopvsindegrees(sloptype='burst', weighted=True, showsuspmsg=False,
                     outfnm=respath+sloptype+'slopsvsindegree.eps')

