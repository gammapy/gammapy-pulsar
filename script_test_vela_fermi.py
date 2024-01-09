import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table
from gammapy.maps import MapAxis, RegionGeom
from gammapy.modeling import Fit
from gammapy.modeling.models import Models
from gammapy.utils.scripts import make_path

from gammapypulsar import (
    AsymmetricLorentzianPhaseModel,
    CountsDataset,
    LogNormalPhaseModel,
    SkyModelPhase,
)

filename = make_path(
    "$GAMMAPY_DATA/catalogs/fermi/3PC_auxiliary_20230728/J0835-4510_3PC_data.fits.gz"
)

table = Table.read(filename)
table = table[: int(len(table) / 2)]

# [int(0.18*len(table)):int(0.4*len(table))] to only have the bridge
phase_min = table["Ph_Min"]
phase_max = table["Ph_Max"]
phase_tot = np.append(phase_min, phase_max[-1])
phases = MapAxis.from_edges(phase_tot, interp="lin", name="phase")

data = table["300_1000_WtCt"]

geom = RegionGeom(region=None, axes=[phases])
counts_dataset = CountsDataset.create(geom=geom)
counts_dataset.counts.data = data

print(counts_dataset)

model1 = AsymmetricLorentzianPhaseModel(
    amplitude=8000, center=0.13, width_1=0.006, width_2=0.03
)
model1.amplitude.frozen = True
model2 = LogNormalPhaseModel.from_expected(amplitude=300, center=0.23, width=0.01)
model2 = LogNormalPhaseModel(amplitude=755, center=-1.36, width=0.5)
model2.amplitude.frozen = True
model2.center.frozen = True
model2.width.frozen = True
model3 = AsymmetricLorentzianPhaseModel(
    amplitude=8000, center=0.56, width_1=0.03, width_2=-0.005
)
model3.amplitude.frozen = True
compound_model = model1 + model2 + model3
compound_model.plot()
for model in compound_model.models:
    model.plot()
counts_dataset.counts.plot()
plt.show()
sky_model = SkyModelPhase(phase_model=compound_model, name="test")
counts_dataset.models = Models(sky_model)
counts_dataset
print(counts_dataset.models)
fit = Fit()
result = fit.run(counts_dataset)
print(result)
print(result.covariance_result)
for param in counts_dataset.models.parameters:
    print(param.name, param.value, param.error)


print(counts_dataset.models[0].parameters["amplitude"])

print(counts_dataset.npred().data)

ax = plt.gca()
counts_dataset.models[0].phase_model.plot(ax=ax, label="Model")
counts_dataset.models[0].phase_model.plot_error(ax=ax, label="error")
counts_dataset.counts.plot(ax=ax, label="Counts")
plt.legend()
plt.show()

counts_dataset.plot_residuals()
plt.show()
