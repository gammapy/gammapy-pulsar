import matplotlib.pyplot as plt
import numpy as np
from gammapy.datasets import Dataset
from gammapy.maps import Map
from gammapy.modeling.models import DatasetModels
from gammapy.stats import cash, cash_sum_cython
from gammapy.utils.random import get_random_state
from gammapy.utils.scripts import make_name


class CountsDataset(Dataset):
    """Counts dataset class.

    Parameters
    ----------
    counts : `~gammapy.datasets.CountsMap`
        Counts map
    models : `~gammapy.modeling.models.Models`
        Models
    name : str
        Dataset name
    meta_table : `~astropy.table.Table`
        Table with meta info
    """

    tag = "CountsDataset"
    stat_type = "cash"

    def __init__(
        self, counts=None, models=None, name=None, mask_safe=None, meta_table=None
    ):
        self.counts = counts
        self.models = models
        self._name = make_name(name)
        self.mask_safe = mask_safe
        self.mask_fit = None
        self.meta_table = meta_table

    @property
    def models(self):
        """Models set on the dataset as a `~gammapy.modeling.models.Models`."""
        return self._models

    @models.setter
    def models(self, models):
        """Models setter."""
        if models is not None:
            models = DatasetModels(models)
            models = models.select(datasets_names=self.name)

        self._models = models

    def __str__(self):
        """String representation."""
        str_ = f"{self.__class__.__name__}\n"
        str_ += "-" * len(self.__class__.__name__) + "\n"
        str_ += "\n"
        str_ += "\t{:32}: {{name}} \n\n".format("Name")

        str_ += "\t{:32}: {{counts:.0f}} \n".format("Total counts")
        str_ += "\t{:32}: {{npred:.2f}}\n\n".format("Predicted counts")

        str_ += "\t{:32}: {{n_bins}} \n".format("Number of total bins")
        str_ += "\t{:32}: {{n_fit_bins}} \n\n".format("Number of fit bins")

        # likelihood section
        str_ += "\t{:32}: {{stat_type}}\n".format("Fit statistic type")
        str_ += "\t{:32}: {{stat_sum:.2f}}\n\n".format(
            "Fit statistic value (-2 log(L))"
        )

        info = self.info_dict()
        str_ = str_.format(**info)

        # model section
        n_models, n_pars, n_free_pars = 0, 0, 0
        if self.models is not None:
            n_models = len(self.models)
            n_pars = len(self.models.parameters)
            n_free_pars = len(self.models.parameters.free_parameters)

        str_ += "\t{:32}: {} \n".format("Number of models", n_models)
        str_ += "\t{:32}: {}\n".format("Number of parameters", n_pars)
        str_ += "\t{:32}: {}\n\n".format("Number of free parameters", n_free_pars)

        if self.models is not None:
            str_ += "\t" + "\n\t".join(str(self.models).split("\n")[2:])

        return str_.expandtabs(tabsize=2)

    def info_dict(self, in_safe_data_range=True):
        """Info dict with summary statistics, summed over energy.

        Parameters
        ----------
        in_safe_data_range : bool
            Whether to sum only in the safe energy range. Default is True.

        Returns
        -------
        info_dict : dict
            Dictionary with summary info.
        """
        info = {}
        info["name"] = self.name

        if self.mask_safe and in_safe_data_range:
            mask = self.mask_safe.data.astype(bool)
        else:
            mask = slice(None)

        counts = 0

        if self.counts:
            counts = self.counts.data[mask].sum()

        info["counts"] = int(counts)

        npred = np.nan
        if self.models:
            npred = self.npred().data[mask].sum()

        info["npred"] = float(npred)

        # data section
        n_bins = 0
        if self.counts is not None:
            n_bins = self.counts.data.size
        info["n_bins"] = int(n_bins)

        n_fit_bins = 0
        if self.mask is not None:
            n_fit_bins = np.sum(self.mask.data)

        info["n_fit_bins"] = int(n_fit_bins)
        info["stat_type"] = self.stat_type

        stat_sum = np.nan
        if self.counts is not None and self.models is not None:
            stat_sum = self.stat_sum()

        info["stat_sum"] = float(stat_sum)

        return info

    @property
    def _geom(self):
        """Main analysis geometry."""
        return self.counts.geom or ValueError("No geometry defined")

    def npred(self):
        """Predicted counts for the models.

        Returns
        -------
        npred : `~gammapy.maps.Map`
            Predicted counts map.
        """
        total_npred = Map.from_geom(self._geom)
        phase = self.counts.geom.axes["phase"].center

        for model in self.models:
            data = model(phase).value
            total_npred.data = data
        return total_npred

    def stat_sum(self):
        # TODO: implement prior
        """Total statistic function value given the current parameters."""

        counts = self.counts.data.astype(float)
        npred = self.npred().data.astype(float)

        return cash_sum_cython(counts.ravel(), npred.ravel())

    def stat_array(self):
        """Statistic function value per bin given the current model parameters."""
        return cash(n_on=self.counts.data, mu_on=self.npred().data)

    def plot_residuals(self, ax=None, **kwargs):
        """Plot residuals."""
        if ax is None:
            ax = plt.gca()
        ax.plot(
            self.counts.geom.axes[0].center,
            (self.counts.data - self.npred().data).squeeze(),
            **kwargs,
        )
        return ax

    @classmethod
    def create(cls, geom, **kwargs):
        """Create empty `CountsDataset` with the given geometry.

        Parameters
        ----------
        geom : `~gammapy.maps.Geom`
            Reference geometry

        Returns
        -------
        empty_counts_dataset : `CountsDataset`
            Empty counts dataset
        """
        counts = Map.from_geom(geom)
        return cls(counts=counts, **kwargs)

    def stack(self, other, nan_to_num=True):
        """Stack another dataset in place.

        Parameters
        ----------
        other : `~gammapy.datasets.CountsDataset`
            Counts dataset
        nan_to_num : bool
            Replace nan and inf with zeros. Default is True.
        """
        self.counts.stack(other.counts, weights=other.mask_safe, nan_to_num=nan_to_num)
        return self

    def fake(self):
        """Simulate fake counts for the current model parameters.

        Returns
        -------
        dataset : `CountsDataset`
            Dataset with fake counts
        """
        random_state = get_random_state(self)
        npred = self.npred()
        data = np.nan_to_num(npred.data, copy=True, nan=0.0, posinf=0.0, neginf=0.0)
        npred.data = random_state.poisson(data)
        self.counts = npred
