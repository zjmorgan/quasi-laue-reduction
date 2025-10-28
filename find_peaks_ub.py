# import mantid algorithms, numpy and matplotlib
from mantid.simpleapi import *
import matplotlib.pyplot as plt
import numpy as np

from mantid.geometry import UnitCell

from functools import partial
from PIL import Image

import skimage.feature
import scipy.optimize
import scipy.spatial

import optimize_refine_ub

import imp
imp.reload(optimize_refine_ub)

from optimize_refine_ub import subhkl

filename = '/HFIR/CG4D/shared/images/ndip_data_test/meso_may/meso_2_15min_2-0_4-5_078.tif'

if not mtd.doesExist('CG4D'):
    CreateSimulationWorkspace(
        Instrument='/HFIR/CG4D/shared/instrument/CG4D_Definition.xml',
        OutputWorkspace='CG4D',
        UnitX='TOF',
        BinParams="0,0.5,1",
    )

    Rebin(
        InputWorkspace='CG4D',
        OutputWorkspace='CG4D',
        Params='0,1,1'
    )

    im = Image.open(filename)

    for i, y in enumerate(np.array(im).flatten(order='F').tolist()):
        mtd['CG4D'].setY(i, [y])


AddSampleLog(Workspace='CG4D', LogName='image', LogText=filename)
SetGoniometer(Workspace='CG4D', Axis0='0,0,1,0,1', Axis1='0,0,0,1,1', Axis2='0,0,1,0,1')

SetUB(Workspace='CG4D', a=12, b=12, c=12)
#PredictPeaks(InputWorkspace='CG4D', WavelengthMin=2, WavelengthMax=4.5, MaxDSpacing=15, ReflectionCondition='Rhombohedrally centred, reverse', OutputWorkspace='test')

class QuasiLaue:

    def __init__(self, ws):
        self.ws = ws
        self.instrument = mtd[self.ws].getInstrument().getFullName()

        self.pixelation = {
            "CG4D": (5000, 1800),
            "IMAGINE": (512, 512),
            "MANDI": (256, 256),
        }

        self.detectors()
        self.detector_to_indices()

        t = np.linspace(0, np.pi, 1024)
        cdf = (t - np.sin(t)) / np.pi

        self._angle = scipy.interpolate.interp1d(cdf, t, kind="linear")

    def detectors(self):
       if not mtd.doesExist(self.ws + '_detectors'):
            PreprocessDetectorsToMD(
                InputWorkspace=self.ws,
                OutputWorkspace=self.ws + '_detectors',
            )

    def extract_images(self):
        return mtd[self.ws].extractY().reshape(-1, *self.pixelation[self.instrument])

    def extract_detector_IDs(self):
        inds = mtd[self.ws + '_detectors'].column('DetectorID')
        return np.array(inds).reshape(-1, *self.pixelation[self.instrument])

    def detector_to_indices(self):
        indices = self.extract_detector_IDs()
        self.shape = indices.shape
        self.indices = indices.flatten().tolist()

    def find_peaks(self, max_peaks=200, min_pix=3, min_rel_intens=0.05):
        CreatePeaksWorkspace(
            InstrumentWorkspace=self.ws,
            NumberOfPeaks=0,
            OutputType="Peak",
            OutputWorkspace=self.ws + '_peaks',
        )

        images = self.extract_images()
        indices = self.extract_detector_IDs()

        for index, image in zip(indices, images):
            coords = skimage.feature.peak_local_max(
                image,
                num_peaks=max_peaks,
                min_distance=min_pix,
                threshold_rel=min_rel_intens,
                exclude_border=min_pix,
            )
            for coord in coords:
                ind = int(index[*coord])
                AddPeak(
                    PeaksWorkspace=self.ws + '_peaks', 
                    RunWorkspace=self.ws,
                    TOF=1,
                    DetectorID=ind,
                    Height=0,
                    BinCount=0,
                )

    def uncertainty_line_segements(self):
        """
        The scattering vector scaled with the (unknown) wavelength.

        Returns
        -------
        kf_ki_dir : list
            Difference between scattering and incident beam directions.

        """

        kf_ki_dir = []
        for peak in mtd[self.ws + '_peaks']:
            kf_ki_dir.append(peak.getDetectorDirectionSampleFrame() + peak.getSourceDirectionSampleFrame())

        return np.array(kf_ki_dir)

    def projection_factor(self):
        two_sin_theta = []
        for peak in mtd[self.ws + '_peaks']:
            two_sin_theta.append(2 * np.sin(0.5 * peak.getScattering()))

        return np.array(two_sin_theta) 

    def find_UB(self, a, b, c, alpha, beta, gamma, wavelength, n_proc=-1):
        """
        Fit the orientation and other parameters.

        Parameters
        ----------
        a, b, c : float
            Lattice lengths.
        alpha, beta, gamma : float
            Lattice angles.
        wavelength : list
            Bandwidth of each reflection.
        n_proc : int, optional
            Number of processes to use. The default is -1.

        """

        kf_ki_dir = self.uncertainty_line_segements()

        opt = subhkl(a, b, c, alpha, beta, gamma)

        UB, hkls, lamdas = opt.minimize(kf_ki_dir, wavelength, n_proc)

        SetUB(Workspace=self.ws + '_peaks', UB=UB)

        for lamda, hkl, peak in zip(lamdas.tolist(), hkls.tolist(), mtd[self.ws + '_peaks']):
            peak.setWavelength(lamda)
            peak.setHKL(*hkl)

        IndexPeaks(PeaksWorkspace=self.ws + '_peaks')

        self.opt = opt

    def peak_indices(self):
        indices = []
        for peak in mtd[self.ws + '_peaks']:
            det_id = peak.getDetectorID()
            ind = self.indices.index(det_id)
            indices.append(ind)
        return np.unravel_index(indices, self.shape)

    def refine_UB(self, cell, wavelength):
        kf_ki_dir = self.uncertainty_line_segements()
        two_sin_theta = self.projection_factor()

        UB, hkls, lamdas, uncertainties = self.opt.refine(kf_ki_dir, two_sin_theta, wavelength, cell)

        SetUB(Workspace=self.ws + '_peaks', UB=UB)

        for lamda, hkl, peak in zip(lamdas.tolist(), hkls.tolist(), mtd[self.ws + '_peaks']):
            peak.setWavelength(lamda)
            peak.setHKL(*hkl)

        ol = mtd[self.ws + '_peaks'].sample().getOrientedLattice()

        ol.setError(*uncertainties)

        IndexPeaks(PeaksWorkspace=self.ws + '_peaks')

ql = QuasiLaue('CG4D')
ql.find_peaks()
ql.find_UB(18.40, 56.67, 6.54, 90, 90, 90, [2, 4.5])
ql.refine_UB('Orthorhombic', [2, 4.5])

images = ql.extract_images()
bank, row, col = ql.peak_indices()

for i, image in enumerate(images):
    plt.close('all')
    fig, ax = plt.subplots(1, 1)
    ax.imshow(image.T, vmin=0, vmax=np.percentile(image, 99), cmap='binary')
    mask = bank == i
    ax.scatter(row[mask], col[mask], color='r', marker='.')
    fig.show()