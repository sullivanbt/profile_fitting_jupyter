from __future__ import print_function
import numpy as np
import ICCFitTools as ICCFT
import pickle
from mantid.simpleapi import *
import sys

# These are parameters for fitting.
RunNumber = 5921
qLow = -5  # Lowest value of q for ConvertToMD 
qHigh = 5; # Highest value of q for ConvertToMD
Q3DFrame='Q_lab' # Either 'Q_lab' or 'Q_sample'; Q_lab recommended if using a strong peaks
                 # profile library from a different sample
eventFileName = '/SNS/MANDI/IPTS-8776/0/5921/NeXus/MANDI_%i_event.nxs' % RunNumber #Full path to the event nexus file
peaksFile = '/SNS/MANDI/shared/ProfileFitting/demo_%i.integrate' % RunNumber #Full path to the ISAW peaks file
UBFile = '/SNS/MANDI/shared/ProfileFitting/demo_%i.mat' % RunNumber #Full path to the ISAW UB file
strongPeakParamsFile = '/SNS/MANDI/shared/ProfileFitting/strongPeakParams_beta_lac_mut_mbvg.pkl' #Full path to pkl file
moderatorCoefficientsFile = '/SNS/MANDI/shared/ProfileFitting/franz_coefficients_2017.dat' #Full path to pkl file
PredPplCoefficients=np.array([3.56405187,  8.34071842,  0.14134522]) #Coefficients for background calculation
DQPixel = 0.003 # The side length of each voxel in the non-MD histogram used for fitting (1/Angstrom)
IntensityCutoff = 200 # Minimum number of counts to not force a profile
EdgeCutoff = 3 # Pixels within EdgeCutoff from a detector edge will be have a profile forced. Currently for Anger cameras only.
FracHKL = 0.5 # Fraction of HKL to consider for profile fitting.
FracStop = 0.05 # Fraction of max counts to include in peak selection.
MinpplFrac = 0.99 # Min fraction of predicted background level to check
MaxpplFrac = 1.01 # Max fraction of predicted background level to check
MindtBinWidth = 15 # Smallest spacing (in microseconds) between data points for TOF profile fitting.
NTheta = 50 # Number of bins for bivarite Gaussian along the scattering angle.
NPhi = 50 # Number of bins for bivariate Gaussian along the azimuthal angle.
DQMax = 0.15 # Largest total side length (in Angstrom) to consider for profile fitting.
DtSpread = 0.015 # The fraction of the peak TOF to consider for TOF profile fitting.

#-------------Do not edit below here.------------------------------------
event_ws = Load(Filename=eventFileName, OutputWorkspace='event_ws')
MDdata = ConvertToMD(InputWorkspace='event_ws',  QDimensions='Q3D', dEAnalysisMode='Elastic',
                         Q3DFrames=Q3DFrame, QConversionScales='Q in A^-1',
                         MinValues='%f, %f, %f' % (qLow, qLow, qLow), Maxvalues='%f, %f, %f' % (qHigh, qHigh, qHigh), MaxRecursionDepth=10,
                         LorentzCorrection=False)
peaks_ws = LoadIsawPeaks(Filename=peaksFile, OutputWorkspace='peaks_ws')
LoadIsawUB(InputWorkspace=peaks_ws, Filename=UBFile)
UBMatrix = peaks_ws.sample().getOrientedLattice().getUB()
dQ = np.abs(ICCFT.getDQFracHKL(UBMatrix, frac=0.5))
dQ[dQ>DQMax]=DQMax
if sys.version_info[0] == 2:
    strongPeakParams = pickle.load(open(strongPeakParamsFile, 'rb'))
else:
    strongPeakParams = pickle.load(open(strongPeakParamsFile, 'rb'),encoding='latin1')
padeCoefficients = ICCFT.getModeratorCoefficients(moderatorCoefficientsFile)

#This will integrate the whole peakset
IntegratePeaksProfileFitting(OutputPeaksWorkspace='peaks_ws_out', OutputParamsWorkspace='params_ws',
         InputWorkspace=MDdata, PeaksWorkspace=peaks_ws, RunNumber=RunNumber, DtSpread=DtSpread,
         UBFile=UBFile,
         ModeratorCoefficientsFile=moderatorCoefficientsFile,
         predpplCoefficients=PredPplCoefficients,
         MinpplFrac=MinpplFrac, MaxpplFrac=MaxpplFrac, MindtBinWidth=MindtBinWidth,
         StrongPeakParamsFile=strongPeakParamsFile)

