import numpy as np
import pickle
from scipy.ndimage.filters import convolve
import ICCFitTools as ICCFT
import BVGFitTools as BVGFT
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets
import matplotlib.pyplot as plt
from mantid.simpleapi import SetInstrumentParameter
reload(ICCFT)
reload(BVGFT)

def addInstrumentParameters(peaks_ws):
    """
    This function adds parameters to instrument files.  This is only done as a TEMPORARY workaround 
    until the instrument files with these parameters are included in the stable release of mantid 
    which is available on the analysis jupyter server.
    """
    instrumentName = peaks_ws.getInstrument().getName()
    if instrumentName == 'MANDI':
        SetInstrumentParameter(Workspace='peaks_ws', ParameterName='fitConvolvedPeak', ParameterType='Bool', Value='False')
        SetInstrumentParameter(Workspace='peaks_ws', ParameterName='sigX0Scale', ParameterType='Number', Value='1.0')
        SetInstrumentParameter(Workspace='peaks_ws', ParameterName='sigY0Scale', ParameterType='Number', Value='1.0')
        SetInstrumentParameter(Workspace='peaks_ws', ParameterName='numDetRows', ParameterType='Number', Value='255')
        SetInstrumentParameter(Workspace='peaks_ws', ParameterName='numDetCols', ParameterType='Number', Value='255')
        SetInstrumentParameter(Workspace='peaks_ws', ParameterName='numBinsTheta', ParameterType='Number', Value='50')
        SetInstrumentParameter(Workspace='peaks_ws', ParameterName='numBinsPhi', ParameterType='Number', Value='50')
        SetInstrumentParameter(Workspace='peaks_ws', ParameterName='fracHKL', ParameterType='Number', Value='0.4')
        SetInstrumentParameter(Workspace='peaks_ws', ParameterName='dQPixel', ParameterType='Number', Value='0.003')
        SetInstrumentParameter(Workspace='peaks_ws', ParameterName='mindtBinWidth', ParameterType='Number', Value='15.0')
        SetInstrumentParameter(Workspace='peaks_ws', ParameterName='maxdtBinWidth', ParameterType='Number', Value='50.0')
        SetInstrumentParameter(Workspace='peaks_ws', ParameterName='peakMaskSize', ParameterType='Number', Value='5')
        SetInstrumentParameter(Workspace='peaks_ws', ParameterName='iccKConv', ParameterType='String', Value='100.0 140.0 120.0')

    elif instrumentName == 'TOPAZ':
        SetInstrumentParameter(Workspace='peaks_ws', ParameterName='fitConvolvedPeak', ParameterType='Bool', Value='False')
        SetInstrumentParameter(Workspace='peaks_ws', ParameterName='sigX0Scale', ParameterType='Number', Value='3.0')
        SetInstrumentParameter(Workspace='peaks_ws', ParameterName='sigY0Scale', ParameterType='Number', Value='3.0')
        SetInstrumentParameter(Workspace='peaks_ws', ParameterName='numDetRows', ParameterType='Number', Value='255')
        SetInstrumentParameter(Workspace='peaks_ws', ParameterName='numDetCols', ParameterType='Number', Value='255')
        SetInstrumentParameter(Workspace='peaks_ws', ParameterName='numBinsTheta', ParameterType='Number', Value='50')
        SetInstrumentParameter(Workspace='peaks_ws', ParameterName='numBinsPhi', ParameterType='Number', Value='50')
        SetInstrumentParameter(Workspace='peaks_ws', ParameterName='fracHKL', ParameterType='Number', Value='0.4')
        SetInstrumentParameter(Workspace='peaks_ws', ParameterName='dQPixel', ParameterType='Number', Value='0.01')
        SetInstrumentParameter(Workspace='peaks_ws', ParameterName='mindtBinWidth', ParameterType='Number', Value='2.0')
        SetInstrumentParameter(Workspace='peaks_ws', ParameterName='maxdtBinWidth', ParameterType='Number', Value='15.0')
        SetInstrumentParameter(Workspace='peaks_ws', ParameterName='peakMaskSize', ParameterType='Number', Value='15')
        SetInstrumentParameter(Workspace='peaks_ws', ParameterName='iccB', ParameterType='String', Value='0.001 0.3 0.005')
        SetInstrumentParameter(Workspace='peaks_ws', ParameterName='iccKConv', ParameterType='String', Value='10.0 800.0 100.0')

    elif instrumentName == 'CORELLI':
        SetInstrumentParameter(Workspace='peaks_ws', ParameterName='fitConvolvedPeak', ParameterType='Bool', Value='False')
        SetInstrumentParameter(Workspace='peaks_ws', ParameterName='sigX0Scale', ParameterType='Number', Value='2.0')
        SetInstrumentParameter(Workspace='peaks_ws', ParameterName='sigY0Scale', ParameterType='Number', Value='2.0')
        SetInstrumentParameter(Workspace='peaks_ws', ParameterName='numDetRows', ParameterType='Number', Value='255')
        SetInstrumentParameter(Workspace='peaks_ws', ParameterName='numDetCols', ParameterType='Number', Value='16')
        SetInstrumentParameter(Workspace='peaks_ws', ParameterName='numBinsTheta', ParameterType='Number', Value='50')
        SetInstrumentParameter(Workspace='peaks_ws', ParameterName='numBinsPhi', ParameterType='Number', Value='50')
        SetInstrumentParameter(Workspace='peaks_ws', ParameterName='fracHKL', ParameterType='Number', Value='0.4')
        SetInstrumentParameter(Workspace='peaks_ws', ParameterName='dQPixel', ParameterType='Number', Value='0.007')
        SetInstrumentParameter(Workspace='peaks_ws', ParameterName='mindtBinWidth', ParameterType='Number', Value='2.0')
        SetInstrumentParameter(Workspace='peaks_ws', ParameterName='maxdtBinWidth', ParameterType='Number', Value='60.0')
        SetInstrumentParameter(Workspace='peaks_ws', ParameterName='peakMaskSize', ParameterType='Number', Value='10')
        SetInstrumentParameter(Workspace='peaks_ws', ParameterName='iccA', ParameterType='String', Value='0.25 0.75 0.5')
        SetInstrumentParameter(Workspace='peaks_ws', ParameterName='iccB', ParameterType='String', Value='0.001 0.3 0.005')
        SetInstrumentParameter(Workspace='peaks_ws', ParameterName='iccR', ParameterType='String', Value='0.05 0.15 0.1')
        SetInstrumentParameter(Workspace='peaks_ws', ParameterName='iccKConv', ParameterType='String', Value='10.0 800.0 100.0')




def showPeakFit(peakNumber,peaks_ws, MDData, UBMatrix, dQ, padeCoefficients, mindtBinWidth=15, maxdtBinWidth=50,
                dQPixel=0.003, fracHKL=0.5, q_frame='lab', neigh_length_m=3, pplmin_frac=0.4, 
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
    
    iccFitDict = ICCFT.parseConstraints(peaks_ws)
    Y3D, gIDX2, pp_lambda2, params2 = BVGFT.get3DPeak(peak, peaks_ws, box, padeCoefficients,qMask,nTheta=nTheta, nPhi=nPhi, plotResults=False, 
                                                   zBG=1.96,fracBoxToHistogram=1.0,bgPolyOrder=1, strongPeakParams=strongPeakParams, 
                                                   q_frame=q_frame, mindtBinWidth=mindtBinWidth, maxdtBinWidth=maxdtBinWidth, 
                                                   pplmin_frac=0.9, pplmax_frac=1.1,forceCutoff=intensityCutoff,
                                                   edgeCutoff=edgeCutoff, peakMaskSize = 5, figureNumber=2, iccFitDict=iccFitDict)
    I1 = np.sum(Y3D[Y3D/Y3D.max()>fracStop])
    print('Peak %i: New: %i; Ell: %i'%(peakNumber, I1, peak.getIntensity()))

    slider = widgets.IntSlider(value=Y3D.shape[1]//2, min=0,max=Y3D.shape[2]-1, step=1, description='z Slice:', 
                               disabled=False, continuous_update=False, orientation='horizontal', 
                               readout=True, readout_format='d')
    return slider, n_events, Y3D
