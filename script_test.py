import matplotlib.pyplot as plt
from gammapy.maps import MapAxis
from gammapy.modeling import Fit
from gammapy.modeling.models import Models
from scipy.stats import cauchy

from gammapypulsar import CountsDataset, CountsMap, LorentzianPhaseModel, SkyModelPhase

phases = MapAxis.from_bounds(0, 1, 500, interp="lin", name="phase")
data = cauchy.pdf(phases.center, loc=0.5, scale=0.1)


counts = CountsMap(data=data, axes=[phases])

counts_dataset = CountsDataset(counts=counts)

model = LorentzianPhaseModel(amplitude=1, center=0.4, width=0.2)
sky_model = SkyModelPhase(phase_model=model, name="test")
counts_dataset.models = Models(sky_model)
print(counts_dataset.models)
fit = Fit()
result = fit.run(counts_dataset)
print(result)
for param in counts_dataset.models.parameters:
    print(param.name, param.value, param.error)

ax = plt.gca()
counts_dataset.counts.plot(ax=ax, label="Counts")
counts_dataset.models[0].phase_model.plot(ax=ax, label="Model")
plt.legend()
plt.show()
