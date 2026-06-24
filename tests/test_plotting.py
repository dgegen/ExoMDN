"""Smoke tests for the cornerplot/cornerplot_logratios plotting functions."""
from exomdn.plotting import cornerplot, cornerplot_logratios


def test_cornerplot_logratios_radius_fractions(exo, model_input):
    exo.predict([model_input], samples=1000)

    grid = cornerplot_logratios(
        data=exo.prediction, data_components=exo.mixture_components, columns=exo.rf_logratios, height=2
    )

    assert grid is not None


def test_cornerplot_radius_fractions(exo, model_input):
    exo.predict([model_input], samples=1000)

    grid = cornerplot(data=exo.prediction, columns=exo.rf, height=2)

    assert grid is not None
