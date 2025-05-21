We provide here the python programs used in our Phys Rev E article 'Glassy dynamics near the interpolation transition in deep recurrrent networks".
They are NlayerSGD.py and NlayerSGDaging.py, together with the Bach chorale data file in the form used by NlayerSGD.py.  
The programs are in a form in which the user can make short tests.  The length of the runs is specified by niter, currently set to 10000 iterations. 
The programs are designed so that one can continue the calculation after a run is finished.  Since the total run time in our calculations sometimes 
exceeded a day, we found it useful to run them in pieces of a few hours.  
In the first run of a lengthy calculation, the control parameters CONTINUE and RETRIEVE should be set to False.  After that they should be set to TRUE.
Furthermore, the BACKFILEin of the new run has to be the BACKFILEout of the previous run.  Unfortunately, these names have to be changed manually.
After one or more runs of the basic learning code NlayerSGD.py, one can do the aging calculation using NlayerSGDaging.py.  On the first run of 
NlayerSGDaging.py, TRAINEDNET has to be set equal to the BACKFILEout of the last NlayerSGD.py run.  
We recommend verifying the code by running NlayerSGD.py twice, once with CONTINUE and RETRIEVE set to False and once with them set to True.  Then try two 
runs, in the same way, of NlayerSGDaging.py.  (With niter=10000, these runs take 6-9 minutes on my Macbook Pro M1.)
