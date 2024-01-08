import operator

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from astropy.visualization import quantity_support
from gammapy.maps import MapAxis, RegionNDMap
from gammapy.modeling import Covariance, Parameter, Parameters
from gammapy.modeling.models import ModelBase
from gammapy.utils.scripts import make_name

__all__ = [
    "PhaseModel",
    "SkyModelPhase",
    "ConstantPhaseModel",
    "LorentzianPhaseModel",
    "AsymmetricLorentzianPhaseModel",
    "LogNormalPhaseModel",
    "CompoundPhaseModel",
]


class SkyModelPhase(ModelBase):
    def __init__(
        self, phase_model, name=None, datasets_names=None, covariance_data=None
    ):

        self.phase_model = phase_model
        self._name = make_name(name)
        self.datasets_names = datasets_names
        super().__init__(covariance_data=covariance_data)

    @property
    def _models(self):
        return [self.phase_model]

    def _check_covariance(self):
        if not self.parameters == self._covariance.parameters:
            self._covariance = Covariance.from_stack(
                [model.covariance for model in self._models],
            )

    @property
    def covariance(self):
        self._check_covariance()

        for model in self._models:
            self._covariance.set_subcovariance(model.covariance)
        return self._covariance

    @covariance.setter
    def covariance(self, covariance):
        self._check_covariance()
        self._covariance.data = covariance

        for model in self._models:
            model.covariance = self._covariance.get_subcovariance(
                model.covariance.parameters
            )

    @property
    def name(self):
        return self._name

    @property
    def parameters(self):
        parameters = []

        parameters.append(self.phase_model.parameters)

        return Parameters.from_stack(parameters)

    @property
    def phase_model(self):
        return self._phase_model

    @phase_model.setter
    def phase_model(self, phase_model):
        if not isinstance(phase_model, PhaseModel):
            raise TypeError(f"Invalid type: {type(phase_model)}")
        self._phase_model = phase_model

    def __call__(self, phase):
        return self.evaluate(phase)

    def evaluate(self, phase):
        return self.phase_model(phase)

    def to_dict(self, full_output=False):
        pass

    @classmethod
    def from_dict(cls, data):
        pass

    def __str__(self):
        return f"SkyModelPhase: {self.phase_model.__class__.__name__}"

    @classmethod
    def create(cls, phase_model, **kwargs):
        pass

    def freeze(self):
        pass

    def unfreeze(self):
        pass


class PhaseModel(ModelBase):
    _type = "phase"

    def __call__(self, phase):
        kwargs = {par.name: par.quantity for par in self.parameters}
        return self.evaluate(phase, **kwargs)

    def __add__(self, model):
        if not isinstance(model, PhaseModel):
            raise TypeError(f"Invalid type: {type(model)}")
        return CompoundPhaseModel([self, model], operator.add)

    def __mul__(self, model):
        if not isinstance(model, PhaseModel):
            raise TypeError(f"Invalid type: {type(model)}")
        return CompoundPhaseModel([self, model], operator.mul)

    def __sub__(self, model):
        if not isinstance(model, PhaseModel):
            raise TypeError(f"Invalid type: {type(model)}")
        return CompoundPhaseModel([self, model], operator.sub)

    def __radd__(self, model):
        return self.__add__(model)

    def __rsub__(self, model):
        return self.__sub__(model)

    def plot(self, phase_bounds=[0, 1], ax=None, n_points=100, **kwargs):
        """Plot the model"""
        ax = plt.gca() if ax is None else ax

        phase_min, phase_max = phase_bounds
        phase = MapAxis.from_bounds(phase_min, phase_max, n_points)

        phase_shape, _ = self._get_plot_shape(phase=phase)

        with quantity_support():
            ax.plot(phase.center, phase_shape.quantity[:, 0, 0], **kwargs)

        self._plot_format_ax(ax)

        return ax

    def plot_error(self, phase_bounds=[0, 1], ax=None, n_points=100, **kwargs):

        ax = plt.gca() if ax is None else ax

        phase_min, phase_max = phase_bounds
        phase = MapAxis.from_bounds(phase_min, phase_max, n_points)

        kwargs.setdefault("facecolor", "black")
        kwargs.setdefault("alpha", 0.2)
        kwargs.setdefault("linewidth", 0)

        phase_shape, phase_shape_err = self._get_plot_shape(phase=phase)
        y_lo = (phase_shape - phase_shape_err).quantity[:, 0, 0]
        y_hi = (phase_shape + phase_shape_err).quantity[:, 0, 0]

        with quantity_support():
            ax.fill_between(phase.center, y_lo, y_hi, **kwargs)

        self._plot_format_ax(ax)
        return ax

    def _get_plot_shape(self, phase):
        shape = RegionNDMap.create(region=None, axes=[phase])
        shape_err = RegionNDMap.create(region=None, axes=[phase])

        shape.quantity, shape_err.quantity = self.evaluate_error(phase.center)

        return shape, shape_err

    def evaluate_error(self, phase, epsilon=1e-3):
        """Evaluate the model"""
        return self._propagate_error(epsilon=epsilon, fct=self, phase=phase)

    def _propagate_error(self, epsilon, fct, **kwargs):
        eps = np.sqrt(np.diag(self.covariance)) * epsilon

        n, f_0 = len(self.parameters), fct(**kwargs)
        shape = (n, len(np.atleast_1d(f_0)))
        df_dp = np.zeros(shape)

        for idx, parameter in enumerate(self.parameters):
            if parameter.frozen or eps[idx] == 0:
                continue

            parameter.value += eps[idx]
            df = fct(**kwargs) - f_0

            df_dp[idx] = df.value / eps[idx]
            parameter.value -= eps[idx]

        f_cov = df_dp.T @ self.covariance @ df_dp
        f_err = np.sqrt(np.diagonal(f_cov))
        return u.Quantity([np.atleast_1d(f_0.value), f_err], unit=f_0.unit).squeeze()

    @staticmethod
    def _plot_format_ax(ax):
        ax.set_xlabel("Phase")
        ax.set_ylabel("Counts")
        ax.set_ylim(0, None)
        return ax


class CompoundPhaseModel(PhaseModel):
    """"""

    tag = ["CompoundPhaseModel", "compoundphase"]

    def __init__(self, models, operator):
        if not isinstance(models, list):
            models = [models]
        self.models = models
        self.operator = operator
        super().__init__()

    @property
    def parameters(self):
        parameters = Parameters()
        for model in self.models:
            parameters += model.parameters
        return parameters

    def __str__(self):
        pass

    def __call__(self, phase):
        val = [model(phase) for model in self.models]
        if self.operator == operator.add:
            return self._add_vals(val)

    def __add__(self, model):
        if not isinstance(model, PhaseModel):
            raise TypeError(f"Invalid type: {type(model)}")
        if isinstance(model, CompoundPhaseModel):
            return CompoundPhaseModel(self.models + model.models, operator.add)
        return CompoundPhaseModel(self.models + [model], operator.add)

    def evaluate(self, phase, *args):
        slice_list = self._get_slice(self.models)
        val = [
            model.evaluate(phase, *args[s]) for model, s in zip(self.models, slice_list)
        ]
        if self.operator == operator.add:
            return self._add_vals(val)

    @staticmethod
    def _add_vals(vals):
        return sum(vals)

    @staticmethod
    def _get_slice(models):
        """get the slice of the parameters"""
        slice_list = []
        lenght = [len(model.parameters) for model in models]
        for idx, l in enumerate(lenght):
            if idx == 0:
                slice_list.append(slice(0, l))
            else:
                slice_list.append(slice(lenght[idx - 1], lenght[idx - 1] + l))
        return slice_list


class ConstantPhaseModel(PhaseModel):
    """"""

    tag = ["ConstantPhaseModel", "constphase"]
    const = Parameter("const", 1, is_norm=True)

    @staticmethod
    def evaluate(phase, const):
        """Evaluate the model"""
        return np.ones(np.atleast_1d(phase).shape) * const


class LorentzianPhaseModel(PhaseModel):
    """"""

    tag = ["LorentzianPhaseModel", "lor"]
    amplitude = Parameter("amplitude", 1, is_norm=True)
    center = Parameter("center", 0)
    width = Parameter("width", 1)

    @staticmethod
    def evaluate(phase, center, amplitude, width):
        """Evaluate the model"""
        return amplitude / (1 + np.power((phase - center) / width, 2))


class AsymmetricLorentzianPhaseModel(PhaseModel):
    """"""

    tag = ["AsymmetricLorentzianPhaseModel", "asymlor"]
    amplitude = Parameter("amplitude", 1, is_norm=True)
    center = Parameter("center", 0.5)
    width_1 = Parameter("width_1", 0.1)
    width_2 = Parameter("width_2", 0.1)

    @staticmethod
    def evaluate(phase, center, amplitude, width_1, width_2):
        """Evaluate the model"""
        l1 = amplitude / (1 + np.power((phase - center) / width_1, 2))
        l2 = amplitude / (1 + np.power((phase - center) / width_2, 2))
        return np.where(phase < center, l1, l2)


class LogNormalPhaseModel(PhaseModel):
    """"""

    tag = ["LogNormalPhaseModel", "lognorm"]
    amplitude = Parameter("amplitude", 1, is_norm=True)
    center = Parameter("center", 0)
    width = Parameter("width", 1)

    @staticmethod
    def evaluate(phase, center, amplitude, width):
        """Evaluate the model"""
        return amplitude * (
            1
            / (phase * width * np.sqrt(2 * np.pi))
            * np.exp(-0.5 * np.power((np.log(phase) - center) / width, 2))
        )

    @classmethod
    def from_expected(cls, amplitude, center, width):
        """Create a lognormal model from expected values.

        Parameters
        ----------
        amplitude : float
            Amplitude of the lognormal distribution at center.
        center : float
            Center of the lognormal distribution.
        width : float
            Width of the lognormal distribution.
        """
        mu = np.log(
            np.power(center, 2) / np.sqrt(np.power(width, 2) + np.power(center, 2))
        )
        sigma = np.sqrt(np.log(np.power(width, 2) / np.power(center, 2) + 1))
        a = amplitude * center * np.sqrt(2 * np.pi) * sigma
        return cls(amplitude=a, center=mu, width=sigma)
