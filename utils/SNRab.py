"""Utility to check SNR"""

def SNRab(x, x_hat, db=True):
    # better safe than sorry (matrix vs array)
    import numpy as np

    xx = x.flatten()
    yy = x_hat.flatten()
    
    u = xx.sum()
    v = (xx*yy).sum()
    w = (yy**2).sum()
    p = yy.sum()
    q = len(xx)**2
    
    a = (v*q - u*p)/(w*q - p*p)
    b = (w*u - v*p)/(w*q - p*p)
    
    SNRopt = np.sqrt((xx**2).sum() / ((xx - (a*yy + b))**2).sum())
    SNRraw = np.sqrt((xx**2).sum() / ((xx - yy)**2).sum())
    
    if db:
        SNRopt = 20*np.log10(SNRopt)
        SNRraw = 20*np.log10(SNRraw)

    return SNRopt, SNRraw, a, b
