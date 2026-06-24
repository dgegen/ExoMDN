import matplotlib

matplotlib.use("Agg")

import pytest

from exomdn import ExoMDN

MODEL_INPUTS = {
    "mass_radius_Teq": [1.0, 1.0, 500],
    "mass_radius_k2_Teq": [1.0, 1.0, 0.3, 500],
}
MODEL_ERRORS = {
    "mass_radius_Teq": [0.1, 0.1, 50],
    "mass_radius_k2_Teq": [0.1, 0.1, 0.05, 50],
}


@pytest.fixture(scope="module", params=sorted(MODEL_INPUTS))
def exo(request):
    exo = ExoMDN(model_path="./models", data_path="./data")
    exo.load_model(exo.model_path / request.param)
    exo.model_name = request.param
    return exo


@pytest.fixture
def model_input(exo):
    return MODEL_INPUTS[exo.model_name]


@pytest.fixture
def model_error(exo):
    return MODEL_ERRORS[exo.model_name]
