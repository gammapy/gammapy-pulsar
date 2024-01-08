import matplotlib.pyplot as plt
from gammapy.datasets import Dataset
from gammapy.modeling.models import DatasetModels
from gammapy.stats import cash, cash_sum_cython
from gammapy.utils.scripts import make_name


class CountsDataset(Dataset):
    """Counts dataset abstract base class."""

    tag = "CountsDataset"
    stat_type = "cash"

    def __init__(self, counts=None, models=None, name=None, meta_table=None):
        self.counts = counts
        self.models = models
        self._evaluators = {}
        self._name = make_name(name)
        self.meta_table = meta_table

    @property
    def models(self):
        return self._models

    @models.setter
    def models(self, models):
        if models is not None:
            models = DatasetModels(models)
            models = models.select(datasets_names=self.name)

        self._models = models

    def npred(self):
        """Predicted counts (model + background)"""
        total_npred = CountsMap(axes=self.counts.axes)
        phase = self.counts.axes[0].center

        for model in self.models:
            data = model(phase)
            total_npred.data = data
        return total_npred

    def stat_sum(self):
        # TODO: implement prior
        """Total statistic function value given the curent parameters."""

        counts = self.counts.data.astype(float)
        npred = self.npred().data.astype(float)

        return cash_sum_cython(counts.ravel(), npred.ravel())

    def stat_array(self):
        return cash(n_on=self.counts.data, mu_on=self.npred().data)

    def plot_residuals(self, ax=None, **kwargs):
        if ax is None:
            ax = plt.gca()
        ax.plot(
            self.counts.axes[0].center, self.counts.data - self.npred().data, **kwargs
        )
        return ax


class CountsMap:
    def __init__(self, data=None, meta=None, geom=None, axes=None):
        self.data = data
        self.meta = meta
        self.axes = axes
        self.geom = geom

    def plot(self, ax=None, **kwargs):
        if ax is None:
            ax = plt.gca()
        ax.plot(self.axes[0].center, self.data, **kwargs)
        return ax
