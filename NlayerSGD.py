#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NlayerSGD.py
version 19 May 2025
Stochastic gradient descent learning with N layers of tanh hiddern units
orthogonal intralayer connection matrices M
ORTHJ, ORTHJ0 options to enforce orthogonality of FF matrices J
saves loss every testinterval steps, orthogonalizes Ms every orthinterval steps
saves test loss statistics
uses training data of nc chorales starting with nbegin 
@authors: Hertz, Tyrcha
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import pickle
import random
from scipy.stats import ortho_group
 
TRAINM = True
TRAINJ = True
ORTHJ = False   # enforce orthogonality of J matrices
ORTHJ0 = False # enforce orthogonality of J0 matrix
CONTINUE = False  # set False for first run, thereafter True

BACKUP = True    # save BACKFILEout to use as BACKFILEin in next run
BACKFILEin =  'SGDbackupNL2Nh70i00Keta0004mbs300S100.pkl'
BACKFILEout = 'SGDbackupNL2Nh70i00Keta0004mbs300S100.pkl'

RETRIEVE = False   #set to False for first run, thereafter True
if RETRIEVE == True:
    with open(BACKFILEin, 'rb') as f:
         plotrange,lossout,testlossout,Svtest,Svtesttarget,testbegins,mutest,mutesttm1,\
            Sv, Svtarget, begins, ends, mSv, stdSv, Svx,\
            Nv, Nvx, sqrtNvx, Ttrain,Ttest, gJ, gJ0, gM,\
            W, J0, J, K, M, J0init, Jinit, Minit, bbias, hbias, h0,\
            mu, mutm1, H, dS, dmu, X, tp, outinterval, et, steprange,\
            totalsteps, nbegin = pickle.load(f)
             

niter = 10000          # number of iterations
outinterval = 1000      #  write progress to terminal every outinterval steps
orthinterval = 200      # interval between orthogonalizations M matrices 
testinterval = 500      # interval between saves of losses to file 
datalength = int(niter/testinterval)
 
eta0 = .0004            # learning rate
startT = 5000           # gradual turning on of learning rate first startT STEPS

nbegin = 100    # number of first chorale in training set
nc = 80         # number of chorales to learn
NL = 2          # model depth
Nh = 70         # model width
mbs = 300       # minibatch size

                    # for first run:     
if CONTINUE == False:
                        # read Bach data file:
    with open('JSBfile2.pkl', 'rb') as f:       # pickle.load statement
        Sv, Svtest, Svval, Svtarget, Svtesttarget, Svvaltarget, begins, ends,\
            testbegins, testends, valbegins, valends = pickle.load(f)
    Sv = Sv[:, begins[nbegin]:ends[nbegin+nc-1]+1]
    Svtarget = Svtarget[:, begins[nbegin]:ends[nbegin+nc-1]+1]
    
    begins = begins[nbegin:nbegin+nc]
    firstbegin = begins[0]
    begins -= firstbegin
    ends = ends[nbegin:nbegin+nc]
    ends -= firstbegin
                        # normalize inputs
    mSv = np.mean(Sv)
    stdSv = np.std(Sv)
    Sv -= mSv*np.ones(np.shape(Sv))
    Sv /= stdSv
    Svtest -= mSv*np.ones(np.shape(Svtest))
    Svtest /= stdSv
    Svx = np.copy(Sv)
                        # convert targets to Ising variables
    Svtarget = 2*Svtarget - np.ones(np.shape(Svtarget))
    Svtesttarget = 2*Svtesttarget - np.ones(np.shape(Svtesttarget))

     
    Nv = np.shape(Sv)[0]
    Nvx = np.shape(Svx)[0]          # extended input
    sqrtNvx = np.sqrt(Nvx)
    Ttrain = np.shape(Sv)[1]
    Ttest = np.shape(Svtest)[1]

    # network parameters:
    gJ0 = np.sqrt(0.1)
    gJ = 1
    gM = 1   
    
    np.random.seed(42)

                    # first FF layer:    
    if ORTHJ0 == True:
        J0 = ortho_group.rvs(dim=Nh)
        u, s, vh = np.linalg.svd(J0, full_matrices=True)
        vh = vh[:Nvx,:Nvx]
        u = u[:,:Nvx]
        J0 = gJ0*u@vh
    else:
        J0 = gJ0*np.random.randn(Nh, Nvx)/sqrtNvx  
    J0init = np.copy(J0)
                    # remaining FF layers initialized to orthogonal    
    J = [gJ*ortho_group.rvs(dim=Nh) for _ in range(NL-1)]
    Jinit = np.copy(J)

                        # intralayer connection matrices   note: layer indices go from 0 to NL-1                             
    M = [np.zeros((Nh, Nh)) for _ in range(NL)]
    for layer in range(NL):
        ML = np.eye(Nh)
        np.random.shuffle(ML)
        while np.all(ML == np.eye(Nh)) == True:
            np.random.shuffle(ML)
        M[layer] = ML
    Minit = np.copy(M)        
    bbias = [np.zeros(Nh) for _ in range(NL)]  # not used in this version
                                    
                            # final hidden to output connection matrix and bias
    K = np.zeros((Nv, Nh))        
    hbias = np.ones(Nv)*np.arctanh(np.mean(Svtarget))   # np.zeros(Nv)
    h0 = np.tile(hbias, Ttrain).reshape(Ttrain, Nv).T
                            # direct input-to--output matrix
    W = np.zeros((Nv, Nvx))
 
    
                             # hidden layer activations   
                                # mu[n] is unit values in hidden layer n+1
    mu = [np.zeros((Nh, Ttrain)) for _ in range(NL)]
    mu[0] = J0@Svx
    for layer in range(1, NL):
        mu[layer] = J[layer-1]@mu[layer-1]
        
    mutest = [np.zeros((Nh, Ttest)) for _ in range(NL)]
    mutest[0] = J0@Svtest
    for layer in range(1, NL):
        mutest[layer] = J[layer-1]@mutest[layer-1]
     
    # initializations: 
    mu = np.asarray(mu)

    mut = [np.zeros((Nh, 1)) for _ in range(NL)]
    for t in range(Ttrain):
        if t > 0:
            Svxt= Svx[:, t]
            Svxt = np.reshape(Svxt, (Nv, 1))
            mu0tm1 = mu[0, :, t-1].reshape(Nh, 1)
            mu0t = (J0@Svxt + M[0]@mu0tm1).reshape(1, Nh)
            mu[0, :, t]= mu0t
            for layer in range(1, NL):
                mulayert = J[layer-1]@mu[layer-1, :, t] + M[layer]@mu[layer, :, t-1]
                mu[layer, :, t] = mulayert

    mutm1 = np.roll(mu, 1, 2)
    mutm1[:, :, begins] = 0
    X = [np.zeros((Nh, Ttrain)) for _ in range(NL)]
    dmu = [np.zeros((Nh, Ttrain)) for _ in range(NL)]
    mu[0] = np.tanh(J0@Svx + M[0]@mutm1[0])
    X[0] = 1 - mu[0]*mu[0]
    for layer in range(1, NL):
        mu[layer] = J[layer-1]@mu[layer-1] + M[layer]@mutm1[layer]
        X[layer] = 1 - mu[layer]*mu[layer]

    H = K@mu[NL-1] + W@Svx + h0
    dS = Svtarget - np.tanh(H)
    dmu[NL-1] = X[NL-1]*(K.T@dS)
    for layer in reversed(range(NL-1)):
        dmu[layer] = X[layer]*(J[layer].T@dmu[layer+1])
    dmu = np.asarray(dmu)
    dmutp1 = np.roll(dmu, -1, 2)
    dmutp1[:, :, ends] = 0
    
        # for test loss evaluation:
    mutest = np.asarray(mutest)
    mutestt = [np.zeros((Nh, 1)) for _ in range(NL)]
    for t in range(Ttest):
        if t > 0:
            Svtestt = Svtest[:, t]
            Svtestt = np.reshape(Svtestt, (Nv, 1))
            mu0testtm1 = mutest[0, :, t-1].reshape(Nh, 1)
            mu0testt = (J0@Svtestt + M[0]@mu0testtm1).reshape(1, Nh)
            mutest[0, :, t] = mu0testt
            for layer in range(1, NL):
                mutestlayert = J[layer-1]@mutest[layer-1,
                                                  :, t] + M[layer]@mutest[layer, :, t-1]
                mutest[layer, :, t] = mutestlayert
    
    mutesttm1 = np.roll(mutest, 1, 2)
    mutesttm1[:, :, testbegins] = 0    
    mutest[0] = np.tanh(J0@Svtest + M[0]@mutesttm1[0])
    for layer in range(1, NL):
        mutest[layer] = J[layer-1]@mutest[layer-1] + M[layer]@mutesttm1[layer]
    hb = h0[:, 0].reshape(Nv, 1)
    h0test = np.tile(hb, Ttest).reshape(Ttest, Nv).T
    Htest = K@mutest[NL-1] + W@Svtest + h0test
    
        # loss statistics:        
    lossout = np.zeros(datalength)
    testlossout = np.zeros(datalength)
     
    steprange = np.arange(niter)
    plotrange = np.arange(datalength) 
    tp = []  
    totalsteps = len(steprange)
    
            # for subsequent runs:
if CONTINUE == True: 
    dsteprange = np.arange(niter) 
    steprange = dsteprange + totalsteps
    totalsteps += niter
    drange = np.arange(datalength)
    drange += len(plotrange)
    plotrange = np.concatenate([plotrange, drange])

    lossout = np.concatenate([lossout, np.zeros(datalength)])
    testlossout = np.concatenate([testlossout, np.zeros(datalength)])
         
    dmu = np.asarray(dmu)
    dmutp1 = np.roll(dmu,-1,2)


then = time.time()
start = time.ctime(then)
print('NlayerSGD learning run started: ', start)


                        # learning loop:
for step in steprange:

    if step < startT:                   # grqduqal turning on of leqrning rate
        eta = eta0*np.sin(np.pi*step/(2*startT))
    else:
        eta = eta0
                    # keeping track of proper times:
    if step == 0:
        et = 0
        tp = [et]
    else:
        et += eta
        if step % testinterval == 0:
            tp.append(et)
        
           # choose the minibatch:     
    t0s = random.sample(range(Ttrain),mbs)               
    Svxs = Svx[:,t0s]
    dSs = dS[:,t0s]
    mus = mu[:,:,t0s]
    dmus = dmu[:,:,t0s]
    mutm1s = mutm1[:,:,t0s]

            # update all the weights:
    W += eta*dSs@Svxs.T/mbs
    K += eta*dSs@mus[NL-1].T/mbs
    h0 += eta*np.reshape(np.mean(dSs, axis=1), (Nv, 1))#/(mbs*nb)                     
    if TRAINJ == True:
        J0 += eta*dmus[0]@Svxs.T/mbs
        for layer in range(NL-1):
            J[layer] += eta*dmus[layer+1]@mus[layer].T/mbs
    if TRAINM == True:
        for layer in range(NL):
            M[layer] += eta*dmus[layer]@mutm1s[layer].T/mbs
        # bbias += eta*np.mean(dmus,axis=2)             # not used in this version
        # for layer in range(NL):
        #     b0[layer] = np.tile(bbias[layer],Ttrain).reshape(Ttrain,Nh).T

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
        dmu[layer] = X[layer]*(J[layer].T@dmu[layer+1]\
            + M[layer].T@dmutp1[layer])           
      
                # reorthogonalizations:                
    if step % orthinterval == 0:
        if ORTHJ0 == True:
            u, s, vh = np.linalg.svd(J0, full_matrices=False)
            J0 = gJ0*u@vh    
        if ORTHJ == True:
            if step > 10:
                for layer in range(NL-1):
                    u, s, vh = np.linalg.svd(J[layer], full_matrices=True)
                    J[layer] = gJ*u@vh  # 0.5*(u@vh + N)        
        if step > 10:
            for layer in range(NL):
                u, s, vh = np.linalg.svd(M[layer], full_matrices=True)
                M[layer] = gM*u@vh  # 0.5*(u@vh + N)
                                                                                
                    # and for test set:               
    mutest[0] = np.tanh(J0@Svtest + M[0]@mutesttm1[0])
    for layer in range(1, NL):
        mutest[layer] = np.tanh(
            J[layer-1]@mutest[layer-1] + M[layer]@mutesttm1[layer])
    mutesttm1 = np.roll(mutest, 1, 2)
    mutesttm1[:, :, testbegins] = 0
    hb = h0[:, 0].reshape(Nv, 1)
    h0test = np.tile(hb, Ttest).reshape(Ttest, Nv).T
    Htest = K@mutest[NL-1] + W@Svtest + h0test
 
                            # calculate the losses:                                           
    if step % testinterval == 0:
        L0 = - np.log(np.ones(np.shape(Sv)) + np.exp(-2*Svtarget*H))
        newloss = -np.sum(L0)/(Ttrain*Nv*np.log(2))
        L0test = - np.log(np.ones(np.shape(Svtest)) + np.exp(-2*Svtesttarget*Htest))
        newtestloss = -np.sum(L0test)/(Ttest*Nv*np.log(2))
        lossout[int(step/testinterval)] = newloss
        testlossout[int(step/testinterval)] = newtestloss
    
                        # printing progress to terminal: 
    if step % outinterval == 0:        
        rightnow = time.time()
        hours, rem = divmod(rightnow-then, 3600)
        minutes, seconds = divmod(rem, 60)
        print('    ', step, 'steps @',
              "{:0>2}:{:0>2}:{:05.2f}".format(
                  int(hours), int(minutes), seconds),
              'train loss', lossout[int(step/testinterval)], 
              'test loss', testlossout[int(step/testinterval)])
# learning loop ends here 


                        # plots:  
plt.figure()
plt.plot(tp, lossout, tp, testlossout)
plt.show
plt.title('training and test losses vs tau')

plt.figure()
plt.semilogx(tp[1:], lossout[1:], tp[1:], testlossout[1:])
plt.show
plt.title('losses vs tau (semilogx)')


            # now write highlights of results to terminal:
print('     Ttrain =', Ttrain)
print(nc, 'chorales, beginning with number', nbegin)
print(' eta =', eta, ' Nh =', Nh, 'NL =', NL, 'batchsize =', mbs)
print('Train M? ', TRAINM)
print('enforce orthogonality of Js? ', ORTHJ, ' of J0? ', ORTHJ0 )
print('autoregressive matrix W used')
print('NlayerSGD results:')
print('     training: min NLL = {:07.5f} at tau = {:07.5f}'.format(
    np.min(lossout), tp[np.argmin(lossout)]))
print('     training: final loss = {:07.5f}'.format(lossout[-1]))
print('     test: min loss = {:07.5f} at tau = {:07.5f}'.format(
    np.min(testlossout), tp[np.argmin(testlossout)]))
print('     test: final loss = {:07.5f}'.format(testlossout[-1]))
print('total proper time elapsed:', tp[-1])

now = time.time()
hours, rem = divmod(now-then, 3600)
minutes, seconds = divmod(rem, 60)
print(step+1, 'steps, total run time:',
      "{:0>2}:{:0>2}:{:02.0f}".format(int(hours), int(minutes), seconds))

if BACKUP == True: 
    with open(BACKFILEout, 'wb') as f:
        pickle.dump([plotrange, lossout,testlossout,Svtest,Svtesttarget,testbegins,mutest,mutesttm1,\
               Sv, Svtarget, begins, ends, mSv, stdSv, Svx,\
               Nv, Nvx, sqrtNvx, Ttrain,Ttest, gJ, gJ0, gM,\
               W, J0, J, K, M, J0init, Jinit, Minit, bbias, hbias, h0,\
               mu, mutm1, H, dS, dmu, X, tp, outinterval, et, steprange,\
               totalsteps, nbegin], f)
    print('BACKFILEout: ', BACKFILEout)

 
