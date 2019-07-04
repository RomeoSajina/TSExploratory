from __future__ import division, print_function
import pandas as pd
from pandas import DataFrame
from abc import ABC, abstractmethod
from sklearn.metrics import mean_absolute_error
from bfast import BFASTCPU as BFAST
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kruskal


# Concept drift detection method's (CDDM)
class ConceptDriftDetectionMethod(ABC):

    def __init__(self):
        self._config = None
        self._train = DataFrame()
        self._new_values = DataFrame()
        self._predictions = list()
        self._next_predictions = list()
        self._yearly_freq = 365
        super().__init__()

    def _init_attrs(self):
        self._config = None
        self._train = DataFrame()
        self._new_values = DataFrame()
        self._predictions = list()
        self._next_predictions = list()

    def reset(self):
        self._init_attrs()

    def is_drifting(self, config, train: DataFrame, new_values: DataFrame, predictions: list):
        self._config = config
        self._train = train
        self._new_values = new_values
        self._predictions = predictions

        is_drifting = self._is_drifting()

        return is_drifting

    def notify_new_predictions(self, predictions: list):
        self._next_predictions = predictions
        self._notify_new_predictions()

    @abstractmethod
    def _is_drifting(self):
        pass

    def _notify_new_predictions(self):
        pass

    def plot(self):
        pass


class IgnoreCDDM(ConceptDriftDetectionMethod):

    def _is_drifting(self):
        return False


class AdaptiveCDDM(ConceptDriftDetectionMethod):

    def _is_drifting(self):
        return True


class KruskalWallisInputCDDM(ConceptDriftDetectionMethod):

    def _is_drifting(self):

        y_pred_range = len(self._new_values)
        n_steps = self._config.n_steps

        new_dist = np.concatenate([self._train.y.values[-n_steps+y_pred_range:], self._new_values.y.values])
        old_dist = self._train.y.values[-self._yearly_freq+y_pred_range:-self._yearly_freq+y_pred_range+n_steps]

        print("Length's: " + str(len(new_dist)) + ", " + str(len(old_dist)))
        _new = np.concatenate([self._train.index.values[-n_steps+y_pred_range:], self._new_values.index.values])
        _old = self._train.index.values[-self._yearly_freq+y_pred_range:-self._yearly_freq+y_pred_range+n_steps]
        print("Ranges: " + str(min(_new)) + " - " + str(max(_new)) + ", " + str(min(_old)) + " - " + str(max(_old)))

        stat, p = kruskal(old_dist, new_dist)

        if self._config.verbose > 1:
            print('Statistics=%.3f, p=%.3f' % (stat, p))

        alpha = 0.05 # TODO: add in class as param
        if p > alpha:
            if self._config.verbose > 1:
                print('Same distributions (fail to reject H0)')
            return False

        else:
            if self._config.verbose > 1:
                print('Different distributions (reject H0)')
            return True


class KruskalWallisOutputCDDM(ConceptDriftDetectionMethod):

    def _is_drifting(self):

        y_pred_range = len(self._new_values)
        n_steps = self._config.n_steps

        new_dist = np.array(self._predictions).flatten()
        old_dist = self._train.y.values[-self._yearly_freq:-self._yearly_freq + y_pred_range]

        print("Length's: " + str(len(new_dist)) + ", " + str(len(old_dist)))
        _new = self._new_values.index.values
        _old = self._train.index.values[-self._yearly_freq:-self._yearly_freq + y_pred_range]
        print("Ranges: " + str(min(_new)) + " - " + str(max(_new)) + ", " + str(min(_old)) + " - " + str(max(_old)))

        stat, p = kruskal(old_dist, new_dist)

        if self._config.verbose > 1:
            print('Statistics=%.3f, p=%.3f' % (stat, p))

        alpha = 0.05  # TODO: add in class as param
        if p > alpha:
            if self._config.verbose > 1:
                print('Same distributions (fail to reject H0)')
            return False

        else:
            if self._config.verbose > 1:
                print('Different distributions (reject H0)')
            return True


class BFASTCDDM(ConceptDriftDetectionMethod):

    # https://pypi.org/project/bfast/#files

    def _is_drifting(self):

        x_l = pd.DatetimeIndex(np.concatenate([self._train.index, self._new_values.index]))
        y_l = np.concatenate([self._train.y.values, self._new_values.y.values])

        y_pred_range = len(self._new_values)

        start = (len(x_l) - y_pred_range)

        bf = BFAST(start=x_l[start], dates=x_l, verbose=0)

        bf.fit(y_l)

        drifts = np.array(np.where(bf.breaks)).flatten().any()

        return drifts

    def _plot(self, x_l, y_l, start, bf):

        plt.plot(x_l, y_l)
        plt.axvline(x=x_l[start], color="red")

        plt.plot(x_l, bf.y_pred, color="brown")

        indexes = start + np.array(np.where(bf.breaks)).flatten()
        plt.plot(x_l[indexes], y_l[indexes], 'ro', color="orange")  # , marker='o')
        plt.scatter(x_l[indexes], y_l[indexes], marker='o', edgecolors="orange", color="orange")

        plt.legend(list(["y", "Start", "BFAST Model", "Breakpoints"]), loc="best")
        plt.title("Breaks For Additive Season and Trend (BFAST)")

    @staticmethod
    def plot_bfast(df: DataFrame, start):

        bf = BFAST(start=start, dates=df.index, verbose=0)

        bf.fit(df.y.values)

        BFASTCDDM()._plot(df.index, df.y.values, np.where(start == df.index), bf)


class CUSUMCDDM(ConceptDriftDetectionMethod):

    def __init__(self):
        self._errors = list()
        self._last_predictions = list()
        super().__init__()

    def _init_attrs(self):
        super()._init_attrs()
        self._errors = list()
        self._last_predictions = list()

    def _is_drifting(self):

        mae = self._calculate_error(self._new_values.y.values, self._last_predictions)

        self._errors.extend(mae)

        #errors = np.concatenate([self._errors, mae])

        ta, tai, taf, amp = self.detect_cusum(self._errors, threshold=15, drift=10, ending=True, show=False)

        # If there is signal in index that represents last prediction errors
        drifts = np.any(np.array(ta) >= len(self._errors) - len(self._last_predictions))

        #drifts = np.array(ta).any()

        return drifts

    def _notify_new_predictions(self):
        self._last_predictions = self._next_predictions

    def _calculate_error(self, real: list, predictions: list):
        return np.array([mean_absolute_error(np.array([real[i]]), np.array([predictions[i]])) for i in range(len(predictions))])

    def detect_cusum(self, x, threshold=1, drift=0, ending=False, show=True, ax=None):
        """Cumulative sum algorithm (CUSUM) to detect abrupt changes in data.

        Parameters
        ----------
        x : 1D array_like
            data.
        threshold : positive number, optional (default = 1)
            amplitude threshold for the change in the data.
        drift : positive number, optional (default = 0)
            drift term that prevents any change in the absence of change.
        ending : bool, optional (default = False)
            True (1) to estimate when the change ends; False (0) otherwise.
        show : bool, optional (default = True)
            True (1) plots data in matplotlib figure, False (0) don't plot.
        ax : a matplotlib.axes.Axes instance, optional (default = None).

        Returns
        -------
        ta : 1D array_like [indi, indf], int
            alarm time (index of when the change was detected).
        tai : 1D array_like, int
            index of when the change started.
        taf : 1D array_like, int
            index of when the change ended (if `ending` is True).
        amp : 1D array_like, float
            amplitude of changes (if `ending` is True).

        Notes
        -----
        Tuning of the CUSUM algorithm according to Gustafsson (2000)[1]_:
        Start with a very large `threshold`.
        Choose `drift` to one half of the expected change, or adjust `drift` such
        that `g` = 0 more than 50% of the time.
        Then set the `threshold` so the required number of false alarms (this can
        be done automatically) or delay for detection is obtained.
        If faster detection is sought, try to decrease `drift`.
        If fewer false alarms are wanted, try to increase `drift`.
        If there is a subset of the change times that does not make sense,
        try to increase `drift`.

        Note that by default repeated sequential changes, i.e., changes that have
        the same beginning (`tai`) are not deleted because the changes were
        detected by the alarm (`ta`) at different instants. This is how the
        classical CUSUM algorithm operates.

        If you want to delete the repeated sequential changes and keep only the
        beginning of the first sequential change, set the parameter `ending` to
        True. In this case, the index of the ending of the change (`taf`) and the
        amplitude of the change (or of the total amplitude for a repeated
        sequential change) are calculated and only the first change of the repeated
        sequential changes is kept. In this case, it is likely that `ta`, `tai`,
        and `taf` will have less values than when `ending` was set to False.

        See this IPython Notebook [2]_.

        References
        ----------
        .. [1] Gustafsson (2000) Adaptive Filtering and Change Detection.
        .. [2] http://nbviewer.ipython.org/github/demotu/BMC/blob/master/notebooks/DetectCUSUM.ipynb

        Examples
        --------

        from detect_cusum import detect_cusum
        x = np.random.randn(300)/5
        x[100:200] += np.arange(0, 4, 4/100)
        ta, tai, taf, amp = detect_cusum(x, 2, .02, True, True)

        x = np.random.randn(300)
        x[100:200] += 6
        detect_cusum(x, 20, 10, True, True)

        x = 2*np.sin(2*np.pi*np.arange(0, 3, .01))
        ta, tai, taf, amp = detect_cusum(x, 1, .05, True, True)
        """

        x = np.atleast_1d(x).astype('float64')
        gp, gn = np.zeros(x.size), np.zeros(x.size)
        ta, tai, taf = np.array([[], [], []], dtype=int)
        tap, tan = 0, 0
        amp = np.array([])
        # Find changes (online form)
        for i in range(1, x.size):
            s = x[i] - x[i - 1]
            gp[i] = gp[i - 1] + s - drift  # cumulative sum for + change
            gn[i] = gn[i - 1] - s - drift  # cumulative sum for - change
            if gp[i] < 0:
                gp[i], tap = 0, i
            if gn[i] < 0:
                gn[i], tan = 0, i
            if gp[i] > threshold or gn[i] > threshold:  # change detected!
                ta = np.append(ta, i)  # alarm index
                tai = np.append(tai, tap if gp[i] > threshold else tan)  # start
                gp[i], gn[i] = 0, 0  # reset alarm
        # THE CLASSICAL CUSUM ALGORITHM ENDS HERE

        # Estimation of when the change ends (offline form)
        if tai.size and ending:
            _, tai2, _, _ = self.detect_cusum(x[::-1], threshold, drift, show=False)
            taf = x.size - tai2[::-1] - 1
            # Eliminate repeated changes, changes that have the same beginning
            tai, ind = np.unique(tai, return_index=True)
            ta = ta[ind]
            # taf = np.unique(taf, return_index=False)  # corect later
            if tai.size != taf.size:
                if tai.size < taf.size:
                    taf = taf[[np.argmax(taf >= i) for i in ta]]
                else:
                    ind = [np.argmax(i >= ta[::-1]) - 1 for i in taf]
                    ta = ta[ind]
                    tai = tai[ind]
            # Delete intercalated changes (the ending of the change is after
            # the beginning of the next change)
            ind = taf[:-1] - tai[1:] > 0
            if ind.any():
                ta = ta[~np.append(False, ind)]
                tai = tai[~np.append(False, ind)]
                taf = taf[~np.append(ind, False)]
            # Amplitude of changes
            amp = x[taf] - x[tai]

        if show:
            self._plot(x, threshold, drift, ending, ax, ta, tai, taf, gp, gn)

        return ta, tai, taf, amp

    def _plot(self, x, threshold, drift, ending, ax, ta, tai, taf, gp, gn):
        """Plot results of the detect_cusum function, see its help."""

        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print('matplotlib is not available.')
        else:
            if ax is None:
                _, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))

            t = range(x.size)
            ax1.plot(t, x, 'b-', lw=2)
            if len(ta):
                ax1.plot(tai, x[tai], '>', mfc='g', mec='g', ms=10,
                         label='Start')
                if ending:
                    ax1.plot(taf, x[taf], '<', mfc='g', mec='g', ms=10,
                             label='Ending')
                ax1.plot(ta, x[ta], 'o', mfc='r', mec='r', mew=1, ms=5,
                         label='Alarm')
                ax1.legend(loc='best', framealpha=.5, numpoints=1)
            ax1.set_xlim(-.01 * x.size, x.size * 1.01 - 1)
            ax1.set_xlabel('Data #', fontsize=14)
            ax1.set_ylabel('Amplitude', fontsize=14)
            ymin, ymax = x[np.isfinite(x)].min(), x[np.isfinite(x)].max()
            yrange = ymax - ymin if ymax > ymin else 1
            ax1.set_ylim(ymin - 0.1 * yrange, ymax + 0.1 * yrange)
            ax1.set_title('Time series and detected changes ' +
                          '(threshold= %.3g, drift= %.3g): N changes = %d'
                          % (threshold, drift, len(tai)))
            ax2.plot(t, gp, 'y-', label='+')
            ax2.plot(t, gn, 'm-', label='-')
            ax2.set_xlim(-.01 * x.size, x.size * 1.01 - 1)
            ax2.set_xlabel('Data #', fontsize=14)
            ax2.set_ylim(-0.01 * threshold, 1.1 * threshold)
            ax2.axhline(threshold, color='r')
            ax1.set_ylabel('Amplitude', fontsize=14)
            ax2.set_title('Time series of the cumulative sums of ' +
                          'positive and negative changes')
            ax2.legend(loc='best', framealpha=.5, numpoints=1)
            plt.tight_layout()
            plt.show()
