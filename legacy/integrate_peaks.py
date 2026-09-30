import numpy as np
import scipy.optimize

from utilities import ParallelProcessor

class IntegratePeaks:
    
    def __init__(self, counts, x, y, roi_pixels=20):
        self.counts = counts
        self.x = x
        self.y = y

    def fit(self, roi_pixels=50, sigma_1=None, sigma_2=None, theta=None):
        im = self.counts
 
        X, Y = np.meshgrid(
            np.arange(im.shape[0]), 
            np.arange(im.shape[1]), indexing="ij"
        )

        data = {}
        for i, (x_val, y_val) in enumerate(zip(self.x, self.y)):
            x_min = int(max(x_val - roi_pixels, 0))
            x_max = int(min(x_val + roi_pixels + 1, im.shape[0]))

            y_min = int(max(y_val - roi_pixels, 0))
            y_max = int(min(y_val + roi_pixels + 1, im.shape[1]))

            x = X[x_min:x_max, y_min:y_max].copy()
            y = Y[x_min:x_max, y_min:y_max].copy()

            z = im[x_min:x_max, y_min:y_max].copy()

            z_level = np.percentile(z, 95)

            indices = np.argwhere(z.flatten() > z_level)

            j, k = np.unravel_index(indices, z.shape)

            j = np.mean(j)
            k = np.mean(k)

            x0 = np.array([
                z.max(),
                0.25 * (z.min() + z.max()),
                j + x_min,
                k + y_min,
                roi_pixels / 6 if sigma_1 is None else sigma_1[i],
                roi_pixels / 6 if sigma_2 is None else sigma_2[i],
                0 if theta is None else theta[i],
            ])

            xmin = np.array([
                z.min(),
                z.min(),
                x_min,
                y_min,
                1 if sigma_1 is None else 0.8 * sigma_1[i],
                1 if sigma_2 is None else 0.8 * sigma_2[i],
                -np.pi if theta is None else theta[i] - np.pi / 6,
            ])

            xmax = np.array([
                2 * z.max(),
                z.max(),
                x_max,
                y_max,
                roi_pixels / 3 if sigma_1 is None else 1.2 * sigma_1[i],
                roi_pixels / 3 if sigma_2 is None else 1.2 * sigma_2[i],
                np.pi if theta is None else theta[i] + np.pi / 6,
            ])

            data[i] = (x_val, y_val, x0, xmin, xmax, x, y, z)

        self.roi_pixels = roi_pixels

        para = ParallelProcessor(1)

        return para.process_dict(data, self._fit)

    def _peak(self, x, y, A, B, mu_x, mu_y, sigma_1, sigma_2, theta):
        a = np.cos(theta) ** 2 / sigma_1**2 + np.sin(theta) ** 2 / sigma_2**2
        b = np.sin(theta) ** 2 / sigma_1**2 + np.cos(theta) ** 2 / sigma_2**2
        c = (1 / sigma_1**2 - 1 / sigma_2**2) * np.sin(2 * theta)

        dx = x - mu_x
        dy = y - mu_y

        k = np.exp(-0.5 * (a * dx ** 2 + b * dy ** 2 + c * dx * dy))

        return A * k + B

    def _intensity(self, A, B, sigma1, sigma2, cov_matrix):
        I = A * 2 * np.pi * sigma1 * sigma2 - B

        dI = np.array([
            2 * np.pi * sigma1 * sigma2,
            -1,
            2 * np.pi * A * sigma2,
            2 * np.pi * A * sigma1,
        ])

        sigma = np.sqrt(dI @ cov_matrix @ dI.T)

        return I, sigma

    def _residual(self, params, x, y, z, x_val, y_val, lamda=0.01):
        A, B, mu_x, mu_y, *_ = params
        penalty = [lamda * (mu_x - x_val), lamda * (mu_y - y_val)]
        return (self._peak(x, y, *params) - z).flatten().tolist() + penalty

    def _fit(self, key_value):
        key, value = key_value

        x_val, y_val, x0, xmin, xmax, x, y, z = value

        I, sig = 0, 0
        mu_1, mu_2 = x_val, y_val
        sigma_1, sigma_2, theta = x0[4:]

        if np.all(x0 > xmin) and np.all(x0 < xmax):

            bounds = np.array([xmin, xmax])

            args = (x, y, z, x_val, y_val)

            sol = scipy.optimize.least_squares(
                self._residual, x0=x0, bounds=bounds, args=args, loss='linear'
            )

            J = sol.jac
            inv_cov = J.T.dot(J)

            A, B, mu_1, mu_2, sigma_1, sigma_2, theta = sol.x

            if np.linalg.det(inv_cov) > 0:

                inds = [0, 1, 4, 5]

                cov = np.linalg.inv(inv_cov)[inds][:,inds]

                I, sig = self._intensity(A, B, sigma_1, sigma_2, cov)

        else:
            print(np.column_stack([xmin, x0, xmax]))

        value = (I, sig, mu_1, mu_2, sigma_1, sigma_2, theta)

        return key, value
