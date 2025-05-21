#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NlayerSGDaging.py
version Tues May20 2025 
Stochastic gradient descent learning
with NL layers of tanh hiddern units
aging calculation, division by noise variance at each step
@authors: Hertz, Tyrcha
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import pickle
import random
 

TRAINM = True
TRAINJ = True
ORTHJ = False   # enforce orthogonality of J matrices
ORTHJ0 = False  # enforce orthogonality of J0 matrix
CONTINUE = True
TRAINEDNET = 'SGDbackupNL2Nh70i20Keta0004mbs300S100.pkl'

 
BACKUP = True
BACKFILEin =  'SGDagingbackupNL2nc80nh70tw20i20Keta0004mbs300S100.pkl'
BACKFILEout = 'SGDagingbackupNL2nc80nh70tw20i30Keta0004mbs300S100.pkl'

RETRIEVE = True
if RETRIEVE == True:
    with open(BACKFILEin, 'rb') as f:
        lossout, Delta, Deltastep, DWold, DKold, DJ0old, DJold, DMold,\
        Wtw, Ktw, J0tw, Jtw, Mtw, DJnew, DMnew, DDJ, DDM,\
        Wgrad, Kgrad, J0grad, Jgrad, Mgrad, noisevar, noises,\
        mbs, eta, plotrange, Svx, Svtarget, begins, ends,\
        NL, Nvx, Nh, nc, Ttrain, gJ, gM, gJ0,\
        W, J0, J, K, M, hbias, h0,\
        mu, mutm1, H, dS, dmu, X, tp, nbegin, totalsteps = pickle.load(f)
           

eta = 0.0004
twinKsteps = 200     # waiting time in thousand of steps
niter = 10000
outinterval = 1000     # interval between writes to terminal
noisecalcinterval = 2000  #  interval between calculations of noise vairance
orthinterval = 200      # interval between orthogonalizations M matrices 
testinterval = 500      # interval between saves of losses to file 
datalength = int(niter/testinterval)
steprange = np.arange(niter)
startT = 5000               # from bpttbachNlayerSGD2.py
mbs = 300
nc = 80
nbegin = 100
 

if CONTINUE == False:
    with open(TRAINEDNET, 'rb') as f:
        plotrange, lossout,testlossout,Svtest,Svtesttarget,testbegins,mutest,mutesttm1,\
               Sv, Svtarget, begins, ends, mSv, stdSv, Svx,\
               Nv, Nvx, sqrtNvx, Ttrain,Ttest, gJ, gJ0, gM,\
               W, J0, J, K, M, J0init, Jinit, Minit, bbias, hbias, h0,\
               mu, mutm1, H, dS, dmu, X, tp, outinterval, et, steprange,\
               totalsteps, nbegin = pickle.load(f)
    
        # loss statistics:        
    lossout = np.zeros(datalength)
    testlossout = np.zeros(datalength)
     
    steprange = np.arange(niter)
    plotrange = np.arange(datalength) 
    tp = []  
    totalsteps = len(steprange)
 

    Nvx = np.shape(Svx)[0]
    Ttrain = np.shape(Svx)[1]
    h0 = np.tile(hbias, Ttrain).reshape(Ttrain, Nvx).T
    NL = np.shape(M)[0]
    Nh = np.shape(M)[1]

    Wtw = np.copy(W)
    Ktw = np.copy(K)
    J0tw = np.copy(J0)
    Jtw = np.copy(J)
    Mtw = np.copy(M)

    Delta = np.zeros(datalength)
    DMnew = np.zeros(NL)
    DDM = np.zeros(NL)
    DJnew = np.zeros(NL-1)
    DDJ = np.zeros(NL-1)

    DWold = 0
    DKold = 0
    DJ0old = 0
    DJold = np.zeros(NL-1)
    DMold = np.zeros(NL)

    Kgrad = np.zeros((Nvx, Nh, Ttrain))
    J0grad = np.zeros((Nh, Nvx, Ttrain))
    Jgrad = np.zeros((NL-1, Nh, Nh, Ttrain))
    Mgrad = np.zeros((NL, Nh, Nh, Ttrain))
    Wgrad = np.zeros((Nvx, Nvx, Ttrain))
    noises = []


if CONTINUE == True:
    dsteprange = np.arange(niter) 
    steprange = dsteprange + totalsteps
    totalsteps += niter
    drange = np.arange(datalength)
    drange += len(plotrange)
    plotrange = np.concatenate([plotrange, drange])

    lossout = np.concatenate([lossout, np.zeros(datalength)])
    Delta = np.concatenate([Delta, np.zeros(datalength)])

dmutp1 = np.roll(dmu, -1, 2)

then = time.time()
start = time.ctime(then)
print('NlayerSGDaging run started: ', start)


for step in steprange:
    t = step*eta
    if step == 0:
        tp = [eta]
    else:
        tp.append(tp[-1] + eta)

 
    t0s = random.sample(range(Ttrain), mbs)
    Svxs = Svx[:, t0s]
    dSs = dS[:, t0s]
    mus = mu[:, :, t0s]
    dmus = dmu[:, :, t0s]
    mutm1s = mutm1[:, :, t0s]

    W += eta*dSs@Svxs.T/mbs
    K += eta*dSs@mus[NL-1].T/mbs
    h0 += eta*np.reshape(np.mean(dSs, axis=1), (Nvx, 1))
    if TRAINJ == True:
        J0 += eta*dmus[0]@Svxs.T/mbs
        for layer in range(NL-1):
            J[layer] += eta*dmus[layer+1]@mus[layer].T/mbs
    if TRAINM == True:
        for layer in range(NL):
            M[layer] += eta*dmus[layer]@mutm1s[layer].T/mbs
            # now we have made all weight changes for all the minibatches for this step
            # now update mu,dmu,dS mutm1:
    mu[0] = np.tanh(J0@Svx + M[0]@mutm1[0])
    for layer in range(1, NL):
        mu[layer] = np.tanh(J[layer-1]@mu[layer-1] + M[layer]@mutm1[layer])
    mutm1 = (mutm1 + np.roll(mu, 1, 2))/2
    mutm1[:, :, begins] = 0
    dmutp1 = (dmutp1 + np.roll(dmu, -1, 2))/2
    dmutp1[:, :, ends] = 0
    H = K@mu[NL-1] + W@Svx + h0
    dS = Svtarget - np.tanh(H)
    X = 1 - mu*mu
    dmu[NL-1] = X[NL-1]*(M[NL-1].T@dmutp1[NL-1] + K.T@dS)
    for layer in reversed(range(NL-1)):
        dmu[layer] = X[layer]*(J[layer].T@dmu[layer+1]
                               + M[layer].T@dmutp1[layer])

            # now do re-orthogonalisations for this step:
    if step % orthinterval == 0:
        if ORTHJ0 == True:
            u, s, vh = np.linalg.svd(J0, full_matrices=False)
            J0 = gJ0*u@vh
        if ORTHJ == True:
            for layer in range(NL-1):
                u, s, vh = np.linalg.svd(J[layer], full_matrices=True)
                J[layer] = gJ*u@vh
        for layer in range(NL):
            u, s, vh = np.linalg.svd(M[layer], full_matrices=True)
            M[layer] = gM*u@vh
    #  learning step done

#  new weight variances, class by class:
    DWnew = np.mean((W - Wtw)**2)
    DKnew = np.mean((K - Ktw)**2)
    DJ0new = np.mean((J0 - J0tw)**2)
#    DJnew = np.mean((J - Jtw)**2)
    for layer in range(NL-1):
        DJnew[layer] = np.mean((J[layer] - Jtw[layer])**2)
    for layer in range(NL):
        DMnew[layer] = np.mean((M[layer] - Mtw[layer])**2)
# new changes (increases) in weight variance:
    DDW = DWnew - DWold
    DDK = DKnew - DKold
    DDJ0 = DJ0new - DJ0old
#    DDJ = DJnew - DJold
    for layer in range(NL-1):
        DDJ[layer] = DJnew[layer] - DJold[layer]
    for layer in range(NL):
        DDM[layer] = DMnew[layer] - DMold[layer]
# and store the current ones to be used as the old ones on the next iteration:
    DWold = np.copy(DWnew)
    DKold = np.copy(DKnew)
    DJ0old = np.copy(DJ0new)
    DJold = np.copy(DJnew)
    DMold = np.copy(DMnew)

# now average over weight classes:
    DD = Nvx*(DDK + DDJ0 + DDW*Nvx/Nh) + Nh*(np.sum(DDJ) + np.sum(DDM))
    DD /= Nvx*Nvx/Nh + 2*Nvx + (2*NL-1)*Nh

# effective noise variance:
    if step % noisecalcinterval == 0:
        for tt in range(Ttrain):
            Svt = Svx[:, tt].reshape(Nvx, 1)
            dSt = dS[:, tt].reshape(Nvx, 1)
            mut = mu[:, :, tt].reshape(NL, Nh, 1)
            dmut = dmu[:, :, tt].reshape(NL, Nh, 1)
            mutm1t = mutm1[:, :, tt].reshape(NL, Nh, 1)

            Wgrad[:, :, tt] = eta*dSt@Svt.T  # (with W)
            Kgrad[:, :, tt] = eta*dSt@mut[NL-1].T
            hgrad = eta*dSt
            if TRAINJ == True:
                J0grad[:, :, tt] = eta*dmut[0]@Svt.T
                for layer in range(NL-1):
                    Jgrad[layer, :, :, tt] = eta*dmut[layer+1]@mut[layer].T
            if TRAINM == True:
                for layer in range(NL):
                    Mgrad[layer, :, :, tt] = eta*dmut[layer]@mutm1t[layer].T

        Wvaria = np.var(Wgrad, axis=2)
        Kvaria = np.var(Kgrad, axis=2)
        J0varai = np.var(J0grad, axis=2)
        JvarLab = np.var(Jgrad, axis=3)
        MvarLab = np.var(Mgrad, axis=3)

        Wvar = np.sum(Wvaria)
        Kvar = np.sum(Kvaria)
        J0var = np.sum(J0varai)
        Jvar = np.sum(JvarLab)
        Mvar = np.sum(MvarLab)

        nparams = 2*Nvx*Nh + (2*NL-1)*Nh*Nh
        nparams += Nvx*Nvx  # (with W)

        noisevar = (Wvar + Kvar + J0var + Jvar + Mvar) /(mbs*nparams)   
        noises.append(noisevar)
 
    # divide by the effective noise variance and accumulate Delta:
    if step == 0:
        Deltastep = DD/noisevar
    else:
        Deltastep += DD/noisevar
        
    if step % testinterval == 0:
        L0 = - np.log(np.ones(np.shape(Svx)) + np.exp(-2*Svtarget*H))
        newloss = -np.sum(L0)/(Ttrain*Nvx*np.log(2))
        lossout[int(step/testinterval)] = newloss
        Delta[int(step/testinterval)]= Deltastep
 
    if step % outinterval == 0:
        rightnow = time.time()
        hours, rem = divmod(rightnow-then, 3600)
        minutes, seconds = divmod(rem, 60)
        print('    ', step, 'steps,  time elapsed: ',
              "{:0>2}:{:0>2}:{:05.2f}".format(
                  int(hours), int(minutes), seconds), 'noisevar:', noisevar)

plt.figure()
plt.plot(eta*np.asarray(plotrange), Delta)
plt.show
plt.title('mean normalized weight change variance')
 
plt.figure()
plt.loglog(eta*np.asarray(plotrange), Delta)
plt.show
plt.title('mean normalized weight change variance (loglog)')


# now write highlights of results to terminal:
print('     Ttrain =', Ttrain)
print(nc, 'chorales, beginning with number', nbegin)
print(' eta =', eta, ' Nh =', Nh, 'NL =', NL, 'batchsize =', mbs)
print('Train M? ', TRAINM)
print('enforce orthogonality of Js? ', ORTHJ, ' of J0? ', ORTHJ0)
print('autoregressive matrix W used')

now = time.time()
hours, rem = divmod(now-then, 3600)
minutes, seconds = divmod(rem, 60)
print(step+1, 'steps, total run time:',
      "{:0>2}:{:0>2}:{:02.0f}".format(int(hours), int(minutes), seconds))

if BACKUP == True:
    with open(BACKFILEout, 'wb') as f:
        pickle.dump([lossout, Delta, Deltastep, DWold, DKold, DJ0old, DJold, DMold,
            Wtw, Ktw, J0tw, Jtw, Mtw, DJnew, DMnew, DDJ, DDM,
            Wgrad, Kgrad, J0grad, Jgrad, Mgrad, noisevar, noises,
            mbs, eta, plotrange, Svx, Svtarget, begins, ends,
            NL, Nvx, Nh, nc, Ttrain, gJ, gM, gJ0,
            W, J0, J, K, M, hbias, h0,
            mu, mutm1, H, dS, dmu, X, tp, nbegin, totalsteps], f)

    print('BACKFILEout: ', BACKFILEout)
