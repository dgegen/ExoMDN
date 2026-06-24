"""Smoke tests for the ExoMDN prediction pipeline.

Both shipped models are exercised so that the keras model and the
preprocessor (log-scale + standardize, loaded from preprocessor.json)
loading paths are covered for each.
"""
def test_predict_returns_expected_dataframes(exo, model_input):
    prediction, mixture_components, input_prompt = exo.predict([model_input], samples=1000)

    assert len(input_prompt) == 1
    assert len(prediction) == 1000
    assert len(mixture_components) > 0
    assert exo.prediction is prediction
    assert exo.mixture_components is mixture_components
    assert exo.input_prompt is input_prompt


def test_predict_with_error_returns_expected_dataframes(exo, model_input, model_error):
    prediction, mixture_components, input_prompt = exo.predict_with_error(
        [model_input], [model_error], samples=(100, 10)
    )

    # Some sampled uncertainty points can fall outside the model's parameter
    # limits and are filtered out, so the count is an upper bound, not exact.
    assert 0 < len(input_prompt) <= 100
    assert len(prediction) > 0
    assert len(mixture_components) > 0
    assert exo.prediction is prediction
    assert exo.mixture_components is mixture_components
    assert exo.input_prompt is input_prompt
