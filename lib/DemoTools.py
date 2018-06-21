import numpy as np
import pickle
from scipy.ndimage.filters import convolve
import ICCFitTools as ICCFT
import BVGFitTools as BVGFT
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets
import matplotlib.pyplot as plt

def showPeakFit(peakNumber,peaks_ws, MDData, UBMatrix, dQ, padeCoefficients, predpplCoefficients, mindtBinWidth=15,
                dQPixel=0.003, fracHKL=0.5, q_frame='lab', neigh_length_m=3, dtS = 0.015, pplmin_frac=0.4, 
                pplmax_frac=1.5, nTheta=50, nPhi=50, intensityCutoff=250, edgeCutoff=3,fracStop=0.05,plotResults=False,
                strongPeakParams=None ):
    
    #Get some peak variables
    peak = peaks_ws.getPeak(peakNumber)
    wavelength = peak.getWavelength() #in Angstrom
    energy = 81.804 / wavelength**2 / 1000.0 #in eV
    flightPath = peak.getL1() + peak.getL2() #in m
    scatteringHalfAngle = 0.5*peak.getScattering()
    Box = ICCFT.getBoxFracHKL(peak, peaks_ws, MDData, UBMatrix, peakNumber, dQ, fracHKL=0.5, dQPixel=dQPixel,  q_frame=q_frame)
    box = Box
    #Set up our filters
    qMask = ICCFT.getHKLMask(UBMatrix, frac=fracHKL, dQPixel=dQPixel, dQ=dQ)
    n_events = Box.getNumEventsArray()
    nX, nY, nZ = n_events.shape
    cX = nX//2; cY = nY//2; cZ = nZ//2;
    dP = 5
    qMask[cX-dP:cX+dP, cY-dP:cY+dP, cZ-dP:cZ+dP] = 0
    neigh_length_m=3
    convBox = 1.0 * \
    np.ones([neigh_length_m, neigh_length_m,
             neigh_length_m]) / neigh_length_m**3
    conv_n_events = convolve(n_events, convBox)
    bgMask = np.logical_and(conv_n_events>0, qMask>0)
    meanBG = np.mean(n_events[bgMask])
    #predppl = np.polyval(f,meanBG)*1.96
    predppl = np.polyval([1,0],meanBG)*1.96
    qMask = ICCFT.getHKLMask(UBMatrix, frac=0.5, dQPixel=dQPixel, dQ=dQ)
    
    #Calculate the 
    
    Y3D1, gIDX1, pp_lambda, params1 = BVGFT.get3DPeak(peak, box, padeCoefficients,qMask,nTheta=nTheta, nPhi=nPhi, plotResults=plotResults, 
                                                   zBG=1.96,fracBoxToHistogram=1.0,bgPolyOrder=1, strongPeakParams=strongPeakParams, 
                                                   predCoefficients=predpplCoefficients, q_frame=q_frame, mindtBinWidth=mindtBinWidth, 
                                                   pplmin_frac=pplmin_frac, pplmax_frac=pplmax_frac,forceCutoff=intensityCutoff,
                                                    edgeCutoff=edgeCutoff
                                                    )
    given_ppl=predppl
    predpplCoefficients2 = [0,0,predppl]
    Y3D2, gIDX2, pp_lambda2, params2 = BVGFT.get3DPeak(peak, box, padeCoefficients,qMask,nTheta=nTheta, nPhi=nPhi, plotResults=False, 
                                                   zBG=1.96,fracBoxToHistogram=1.0,bgPolyOrder=1, strongPeakParams=strongPeakParams, 
                                                   predCoefficients=predpplCoefficients2, q_frame=q_frame, mindtBinWidth=mindtBinWidth, 
                                                   pplmin_frac=0.99999, pplmax_frac=1.0001,forceCutoff=intensityCutoff,
                                                    edgeCutoff=edgeCutoff, figureNumber=3)
    I1 = np.sum(Y3D1[Y3D1/Y3D1.max()>fracStop])
    I2 = np.sum(Y3D2[Y3D2/Y3D2.max()>fracStop])
    print('Peak %i: Old: %i; New: %i; Ell: %i'%(peakNumber, I1,I2, peak.getIntensity()))

    slider = widgets.IntSlider(value=Y3D1.shape[1]//2, min=0,max=Y3D1.shape[2]-1, step=1, description='z Slice:', 
                               disabled=False, continuous_update=False, orientation='horizontal', 
                               readout=True, readout_format='d')
    return slider, n_events, Y3D1, Y3D2
