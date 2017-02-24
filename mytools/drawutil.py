import matplotlib.pyplot as plt
import numpy as np


def drawScatterPoints(xs, ys, outfig=None, suptitle="scatter points",
                     xlabel='x', ylabel='y'):
    fig = plt.figure()
    fig.suptitle(suptitle)
    plt.scatter(xs, ys, marker='.')
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    if outfig is not None:
        fig.savefig(outfig)
    return fig

def drawHexbin(xs, ys, outfig=None, xscale = 'log', yscale= 'log',
               gridsize = 200,
               suptitle='Hexagon binning points',
               colorscale=True ):
    '''
        xscale: [ 'linear' | 'log' ]
            Use a linear or log10 scale on the horizontal axis.
	yscale: [ 'linear' | 'log' ]
            Use a linear or log10 scale on the vertical axis.
        gridsize: [ 100 | integer ]
            The number of hexagons in the x-direction, default is 100. The
            corresponding number of hexagons in the y-direction is chosen such that
            the hexagons are approximately regular. Alternatively, gridsize can be
            a tuple with two elements specifying the number of hexagons in the
            x-direction and the y-direction.
    '''
    if xscale == 'log' and min(xs)<=0:
        print '[Warning] logscale with nonpositive values in x coord'
        print '\tremove {} nonpositives'.format(len(np.argwhere(xs<=0)))
        xg0=xs>0
        xs = xs[xg0]
        ys = ys[xg0]
    if yscale == 'log' and min(ys)<=0:
        print '[Warning] logscale with nonpositive values in y coord'
        print '\tremove {} nonpositives'.format(len(np.argwhere(ys<=0)))
        yg0=ys>0
        xs = xs[yg0]
        ys = ys[yg0]

    fig = plt.figure()
    if colorscale:
        plt.hexbin(xs, ys, bins='log', gridsize=gridsize, xscale=xscale,
                   yscale=yscale, mincnt=1, cmap=plt.cm.jet)
        plt.title(suptitle+' with a log color scale')
        cb = plt.colorbar()
        cb.set_label('log10(N)')
    else:
        plt.hexbin(xs, ys, gridsize=gridsize, xscale=xscale, yscale=yscale,
                   mincnt=1, cmap=plt.cm.jet)
        plt.title(suptitle)
        cb = plt.colorbar()
        cb.set_label('counts')
    #plt.axis([xmin, xmax, ymin, ymax])
    if outfig is not None:
        fig.savefig(outfig)
    return fig

def drawTimeseries(T, S, bins='auto', savepath='', savefn=None, dumpfn=None):
    ts = np.histogram(T,bins=bins)
    y = np.append([0],ts[0])
    f=plt.figure()
    plt.plot(ts[1], y, 'r-+')
    if len(S)>0:
        ssts = np.histogram(S, bins=ts[1])
        ssy = np.append([0], ssts[0])
        plt.plot(ssts[1], ssy, 'b-*')
    if savefn is not None:
        f.savefig(savepath+savefn)
    if dumpfn is not None:
        import pickle
        pickle.dump(f, file(savepath+dumpfn,'w'))
    return f

