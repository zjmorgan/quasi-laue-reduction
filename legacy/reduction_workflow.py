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

from matplotlib.patches import Ellipse, Rectangle

import optimize_refine_ub
import integrate_peaks
import utilities

import imp
imp.reload(optimize_refine_ub)
imp.reload(integrate_peaks)
imp.reload(utilities)

from optimize_refine_ub import CalculateUB
from integrate_peaks import IntegratePeaks
from utilities import WeightedKernel

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

        self.reflection_lattice = {
            "P": "Primitive",
            "I": "Body centred",
            "F": "All-face centred",
            "R": "Rhombohedrally centred, obverse",  # rhomb/hex axes
            "R(obv)": "Rhombohedrally centred, obverse",  # hex axes
            "R(rev)": "Rhombohedrally centred, reverse",  # hex axes
            "A": "A-face centred",
            "B": "B-face centred",
            "C": "C-face centred",
        }

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

        opt = CalculateUB(a, b, c, alpha, beta, gamma)

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

    def predict_peaks(self, wavelength, centering="P", d_min=1.2):

        PredictPeaks(
            InputWorkspace=self.ws + '_peaks',
            WavelengthMin=wavelength[0],
            WavelengthMax=wavelength[1],
            MinDSpacing=d_min,
            MaxDSpacing="inf",
            ReflectionCondition=self.reflection_lattice[centering],
            OutputWorkspace=self.ws + '_peaks',
        )

    def integrate_peaks(self, roi_pixels=10):

        images = self.extract_images()
        detector_IDs = self.extract_detector_IDs()

        hkls, lamdas, det_ids, delete_rows = [], [], [], []

        for row, peak in enumerate(mtd[self.ws + '_peaks']):
            hkl = peak.getHKL()
            lamda = peak.getWavelength()
            det_id = peak.getDetectorID()
            if det_id < 0:
                delete_rows.append(delete_rows)
            else:
                hkls.append(hkl)
                lamdas.append(lamda)
                det_ids.append(det_id)

        DeleteTableRows(TableWorkspace=self.ws + '_peaks', Rows=delete_rows)

        RenameWorkspace(
            InputWorkspace=self.ws + '_peaks',
            OutputWorkspace=self.ws + '_tmp'
        )

        CreatePeaksWorkspace(
            InstrumentWorkspace=self.ws,
            NumberOfPeaks=0,
            OutputType="Peak",
            OutputWorkspace=self.ws + '_peaks',
        )

        CopySample(
            InputWorkspace=self.ws + '_tmp',
            OutputWorkspace=self.ws + '_peaks',
            CopyName=False,
            CopyMaterial=False,
            CopyEnvironment=False,
            CopyShape=False
        )

        det_ids = np.array(det_ids)

        flat = detector_IDs.ravel()
        order = np.argsort(flat)
        sorted_ids = flat[order]

        pos = np.searchsorted(sorted_ids, det_ids)

        i, j, k = np.unravel_index(order[pos], detector_IDs.shape)

        hkls = np.array(hkls)
        lamdas = np.array(lamdas)

        for ind, (index, image) in enumerate(zip(detector_IDs, images)):

            mask = ind == i

            x, y = j[mask], k[mask]

            intgr = IntegratePeaks(image, x, y)

            sx, sy, sxy, w = [], [], [], []

            peak_dict = intgr.fit(roi_pixels=roi_pixels)

            fig, ax = plt.subplots(1, 1)
            ax.imshow(
                image.T,
                cmap='binary',
                origin='lower',
                vmin=0,
                vmax=np.percentile(image, 99),
            )
            ax.minorticks_on()

            for key in peak_dict.keys():

                I, sig, mu_1, mu_2, sigma_1, sigma_2, theta = peak_dict[key]

                ellipse = Ellipse(
                    xy=(mu_1, mu_2),
                    width=6 * sigma_1,
                    height=6 * sigma_2,
                    angle=np.rad2deg(theta),
                    linestyle="-",
                    edgecolor="r",
                    facecolor="none",
                    rasterized=False,
                    zorder=100,
                )
                ax.add_patch(ellipse)

                rectangle = Rectangle(
                    xy=(x[key] - roi_pixels, y[key] - roi_pixels),
                    width=2 * roi_pixels,
                    height=2 * roi_pixels,
                    linestyle="-",
                    edgecolor="w",
                    facecolor="none",
                )
                ax.add_patch(rectangle)

                sigma_x = np.sqrt(sigma_1**2 * np.cos(theta)**2 + sigma_2**2 * np.sin(theta)**2)
                sigma_y = np.sqrt(sigma_1**2 * np.sin(theta)**2 + sigma_2**2 * np.cos(theta)**2)
                sigma_xy = (sigma_1**2 - sigma_2**2) * np.sin(theta) * np.cos(theta)

                sx.append(sigma_x)
                sy.append(sigma_y)
                sxy.append(sigma_xy)
                w.append(I / sig if sig > 0 else 0)

            fig.show()

            s = np.column_stack([sx, sy, sxy])
            w = np.array(w)

            strong = w > 5

            wk = WeightedKernel(k=5).fit(x[strong], y[strong], s[strong], w[strong])
            sx, sy, sxy = wk.predict(x, y)

            A = np.zeros((s.shape[0], 2, 2))
            A[:, 0, 0] = sx**2
            A[:, 1, 1] = sy**2
            A[:, 0, 1] = A[:, 1, 0] = sxy

            vals, vecs = np.linalg.eigh(A)
            vals = np.clip(vals, 1e-12, None)
            Ap = vecs @ (vals[..., None] * np.swapaxes(vecs, -1, -2))

            sx = np.sqrt(Ap[:, 0, 0])
            sy = np.sqrt(Ap[:, 1, 1])
            sxy = Ap[:, 0, 1]

            theta = (0.5 * np.arctan2(2 * sxy, sx**2 - sy**2) + np.pi / 2) % np.pi - np.pi / 2

            sigma_1 = np.sqrt(0.5 * (sx**2 + sy**2) + np.hypot(0.5 * (sx**2 - sy**2), sxy))
            sigma_2 = np.sqrt(0.5 * (sx**2 + sy**2) - np.hypot(0.5 * (sx**2 - sy**2), sxy))

            roi_pixels *= 2

            peak_dict = intgr.fit(roi_pixels, sigma_1, sigma_2, theta)

            fig, ax = plt.subplots(1, 1)
            ax.imshow(
                image.T,
                cmap='binary',
                origin='lower',
                vmin=0,
                vmax=np.percentile(image, 99),
            )
            ax.minorticks_on()

            for key in peak_dict.keys():

                I, sig, mu_1, mu_2, sigma_1, sigma_2, theta = peak_dict[key]

                ellipse = Ellipse(
                    xy=(mu_1, mu_2),
                    width=6 * sigma_1,
                    height=6 * sigma_2,
                    angle=np.rad2deg(theta),
                    linestyle="-",
                    edgecolor="r",
                    facecolor="none",
                    rasterized=False,
                    zorder=100,
                )
                ax.add_patch(ellipse)

                rectangle = Rectangle(
                    xy=(x[key] - roi_pixels, y[key] - roi_pixels),
                    width=2 * roi_pixels,
                    height=2 * roi_pixels,
                    linestyle="-",
                    edgecolor="w",
                    facecolor="none",
                )
                ax.add_patch(rectangle)

                row = int(round(mu_1))
                col = int(round(mu_2))
                
                if row < 0:
                    row = 0
                elif row >= index.shape[0]:
                    row = index.shape[0] - 1

                if col < 0:
                    col = 0
                elif col >= index.shape[1]:
                    col = index.shape[1] - 1

                ind = int(index[row, col])
                AddPeak(
                    PeaksWorkspace=self.ws + '_peaks', 
                    RunWorkspace=self.ws,
                    TOF=1,
                    DetectorID=ind,
                    Height=0,
                    BinCount=0,
                )

                n =  mtd[self.ws + '_peaks'].getNumberPeaks()
                peak = mtd[self.ws + '_peaks'].getPeak(n - 1)

                peak.setWavelength(lamdas[mask][key])
                peak.setHKL(*hkls[mask][key])
                peak.setIntensity(I)
                peak.setSigmaIntensity(sig)

            fig.show()

ql = QuasiLaue('CG4D')
ql.find_peaks()
ql.find_UB(18.40, 56.67, 6.54, 90, 90, 90, [2, 4.5])
ql.refine_UB('Orthorhombic', [2, 4.5])
ql.predict_peaks([2, 4.5], 'F', 1.0)
ql.integrate_peaks()
ql.refine_UB('Orthorhombic', [2, 4.5])
ql.predict_peaks([2, 4.5], 'F', 1.0)
ql.integrate_peaks()

images = ql.extract_images()
bank, row, col = ql.peak_indices()

for i, image in enumerate(images):
    fig, ax = plt.subplots(1, 1)
    ax.imshow(image.T, vmin=0, vmax=np.percentile(image, 99), cmap='binary', origin='lower')
    mask = bank == i
    ax.scatter(row[mask], col[mask], color='r', marker='.')
    fig.show()