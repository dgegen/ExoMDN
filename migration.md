# Migration notes: tensorflow / tensorflow-probability

Current pins in [pyproject.toml](pyproject.toml):
```
tensorflow==2.15.*
tensorflow-probability==0.23.*
```
(Originally `tensorflow==2.11.*` / `tensorflow-probability==0.15.*` in [requirements.txt](requirements.txt) / [environment.yml](environment.yml).)

## Finding: 2.11 -> 2.15 / 0.15 -> 0.23 is safe

Tested end-to-end (model loading, custom `MDN` Keras layer in [mdn_layer.py](exomdn/mdn_layer.py), `tfd.Categorical` / `tfd.MultivariateNormalDiag` / `tfd.Mixture` construction, sampling, log-ratio inverse transform in [mdn_model.py](exomdn/mdn_model.py)) against the already-installed `.venv` (tf 2.15.1 / tfp 0.23.0) using the pretrained models in `models/`. Full `Model.predict()` pipeline ran correctly, no deprecation warnings from the exomdn/TF/TFP code paths exercised. The code only touches long-stable core APIs (`tfd.*` distributions, Keras `Layer` subclassing: `build`, `call`, `get_config`, `trainable_weights`), none of which had breaking changes in this range.

TF 2.15 is the **last TF version where Keras 2 is the default** (`tf.keras` == Keras 2). TF 2.16+ defaults to Keras 3. This boundary matters for the next finding.

## Finding: dropping the tensorflow/tensorflow-probability pins entirely is NOT safe

Tested against latest releases on PyPI (tensorflow 2.21.0, tensorflow-probability 0.25.0) in an isolated venv, running the actual `Model.predict()` pipeline against the pretrained models in `models/mass_radius_Teq/`.

### 1. Saved models fail to load (hard failure)
The pretrained models in `models/*/model/` are in TF **SavedModel** directory format. Keras 3 (default from TF 2.16+) does not support loading SavedModel dirs via `tf.keras.models.load_model()`:
```
ValueError: File format not supported: filepath=models/mass_radius_Teq/model.
Keras 3 only supports V3 `.keras` files and legacy H5 format files (`.h5` extension).
...use `keras.layers.TFSMLayer(...)` for inference-only.
```
This breaks [mdn_model.py:34](exomdn/mdn_model.py#L34) (`tf.keras.models.load_model(..., custom_objects={"MDN": ..., "mdn_loss_func": ...})`) for every shipped pretrained model.

**Fix options:**
- Re-save/convert existing models to `.keras` (V3) or `.h5` format, or
- Switch inference loading to `keras.layers.TFSMLayer` (inference-only — loses the custom-object/loss-function loading path currently used, would need rework of `mdn_model.py`)

### Recommended storage format: `.keras` (V3), tested forward-compatible

Re-saved `models/mass_radius_Teq/model` (SavedModel dir, 6.4 MB) as `.keras` using the current `.venv` (tf 2.15.1, `m.keras_model.save("model.keras")`), then loaded that same file from the latest-TF env (tf 2.21.0 / Keras 3) with `tf.keras.models.load_model(..., custom_objects={"MDN": ..., "mdn_loss_func": ...}, compile=False)` — loads correctly and `.predict()` produces correct-shaped output. So `.keras` written today is forward-compatible with the future Keras-3 upgrade, unlike the current SavedModel dirs.

Other benefits over the current SavedModel directory format:

- Single file vs. a directory of 3+ files (`saved_model.pb`, `keras_metadata.pb`, `variables/`) — simpler to track/ship in git, easier `joblib`-style co-location with `preprocessor.pkl`.
- ~3x smaller on disk for this model (6.4 MB SavedModel dir -> 2.1 MB `.keras`, same weights).
- Native Keras 3 format going forward — no `TFSMLayer` inference-only workaround needed at the eventual TF 2.16+ migration.

Caveat found: the `.keras` file also serializes the **compile state** (optimizer/loss config). Loading with `compile=False` works fine on Keras 3; loading with default `compile=True` fails on Keras 3 with `ValueError: Could not deserialize 'keras.optimizers.legacy.Adam' because it is not a KerasSaveable subclass` (the models here were trained with `keras.optimizers.legacy.Adam`, which Keras 3 dropped). Since `mdn_model.py` only ever does inference (no further training from the loaded model), switching the loader to `compile=False` removes this caveat entirely and costs nothing in the current code path.

**Action:** as part of the TF 2.15 -> 2.16+ migration (or even before, since it's forward-compatible and cheap): re-save all `models/*/model` SavedModel dirs as `models/*/model.keras` using the current `.venv`, update `mdn_model.py:34` to load `model.keras` with `compile=False`, and drop the old SavedModel directories.

### 2. `tensorflow-probability` needs an explicit extra dependency
Since TF 2.16 (Keras 3 default), TFP requires the separate `tf_keras` package (Keras 2 compat shim) to use TF at all. It is not pulled in automatically by `pip install tensorflow-probability`:
```
ModuleNotFoundError: No module named 'tf_keras'
Failed to import TF-Keras... install the tf-keras or tf-keras-nightly package...
or the tensorflow-probability[tf] extra.
```
**Fix:** add `tf-keras` as an explicit dependency (or depend on `tensorflow-probability[tf]`).

### 3. `requires-python` upper bound is already a problem for latest TF
Current `requires-python = ">=3.9,<3.12"` in [pyproject.toml](pyproject.toml). Latest `tensorflow` (2.21.0) requires `python>=3.10` — `>=3.9` is already stale for that. Wheels exist for cp310–cp313, so the `<3.12` upper bound would also need raising to use newer TF.

## Open question for full migration plan
Combine with the scikit-learn-constraint analysis (separate agent) before finalizing — there's a known interaction: newer TF/Keras pulls numpy 2.x, and `scikit-learn==1.1.1` (current pin) has compiled extensions incompatible with numpy 2.x ABI:
```
ValueError: numpy.dtype size changed, may indicate binary incompatibility. Expected 96 from C header, got 88 from PyObject
```
Bumping sklearn also raises a pickle-compat concern: loading the existing pickled preprocessors (`preprocessor.pkl`) under a much newer sklearn already throws `InconsistentVersionWarning` (tested: pickled under 1.1.1, loaded under 1.7.2) — preprocessors may need to be re-fit/re-pickled as part of any sklearn bump.

## Recommendation (pending scikit-learn analysis)
- Short term: keep `tensorflow==2.15.*` / `tensorflow-probability==0.23.*` — last version line where Keras 2 is default and existing SavedModel artifacts load without changes.
- Full upgrade to latest TF/TFP requires, as a bundled migration (not a simple pin bump):
  1. Re-save pretrained models in Keras 3-compatible format (`.keras`) or rework loading to `TFSMLayer`.
  2. Add `tf-keras` dependency.
  3. Raise `requires-python` upper bound.
  4. Resolve scikit-learn/numpy ABI conflict and re-pickle preprocessors (see scikit-learn analysis).

## DONE: migrated to `tensorflow==2.21.*` / `tensorflow-probability==0.25.*`

All four blockers above are resolved:

1. Both `models/*/model.keras` already existed and were re-verified to load and predict correctly under TF 2.21/Keras 3 (native, no `TF_USE_LEGACY_KERAS` needed).
2. Added `tf-keras==2.21.*` — confirmed required (TFP's environment validation imports `tf_keras` as soon as `tensorflow` is also imported, even though the `.keras` models themselves load fine natively under Keras 3 without it).
3. Raised `requires-python` to `>=3.10,<3.14` (latest `tensorflow`/`tf-keras` require `python>=3.10`; wheels exist through 3.13).
4. scikit-learn/numpy ABI conflict is moot — scikit-learn was already removed from runtime dependencies entirely (see the numpy/scikit-learn section below); inference no longer touches scikit-learn or numpy 2.x ABI surfaces at all.

Verified in an isolated venv (tensorflow 2.21.0, tensorflow-probability 0.25.0, tf-keras 2.21.0) before applying pins: both shipped models load, predict, and produce mixture components numerically identical to the old TF 2.15 environment (max abs diff ~1e-8, floating-point noise from different kernel implementations). Full project test suite (8/8) passes against the bumped pins.

---

## Migration notes: numpy / scikit-learn / seaborn

Current pins in [pyproject.toml](pyproject.toml):
```
scikit-learn==1.1.1
numpy==1.26.*
seaborn          # unpinned
```

## Finding: `numpy==1.26.*` pin is redundant, safe to drop

No code in `exomdn/*.py` or the notebooks uses any NumPy API removed/changed in 2.0 (checked for `np.float`/`np.int`/`np.bool`/`np.object`/`np.str`/`np.NaN`/`np.in1d`/`np.row_stack`/etc. — none found).

The real constraint is transitive: `tensorflow==2.15.*` itself pins `numpy <2.0.0,>=1.23.5` (py<=3.11) in its own metadata. So even with the explicit `numpy==1.26.*` line removed, the resolver still can't pull in numpy 2.x while TF 2.15 is pinned — dropping ExoMDN's own pin is a safe no-op until/unless the TF pin is also bumped (see TF section above: numpy 2.x only becomes reachable as part of the Keras-3 / TF 2.16+ migration bundle).

**Action:** safe to delete `"numpy==1.26.*"` from [pyproject.toml](pyproject.toml) and the matching line in [requirements.txt](requirements.txt) independent of anything else.

## Finding: `scikit-learn==1.1.1` pin is NOT safe to drop without re-pickling artifacts

Not an API-usage problem — `exomdn/mdn_model.py:33` only calls `joblib.load(...)` then `.transform()`/`.inverse_transform()` on the loaded object, all long-stable sklearn APIs.

The actual constraint: the repo ships **pre-fitted, pickled** sklearn objects (`models/*/preprocessor.pkl`, each a `Pipeline` with a `ColumnTransformer`(log10 `FunctionTransformer`) + `StandardScaler`), pickled under sklearn 1.1.1. scikit-learn does not guarantee pickle compatibility across versions — confirmed: loading the same `preprocessor.pkl` (pickled under 1.1.1) under sklearn 1.7.2 throws `InconsistentVersionWarning`, and combined with a numpy 2.x environment throws the hard `ValueError: numpy.dtype size changed...` ABI failure already noted above (compiled-extension ABI mismatch, not just a version warning).

**Migration steps for a real sklearn bump** (not just a pin edit):

1. Locate/recreate the fitting code for each `preprocessor.pkl` (one per `models/*/` directory — there are multiple, not just one).
2. In a new env with the target sklearn (+ compatible numpy/TF), re-fit and `joblib.dump` each preprocessor, overwriting the existing `.pkl` files.
3. Validate numerically, not just "loads without error": diff `.transform()` outputs old vs. new sklearn on identical sample inputs, then diff full pipeline predictions (preprocessor → TF model) old vs. new.
4. Bump the pin and commit the regenerated `.pkl` files in the same commit, so the pin and artifacts never drift apart.
5. Spot-check sklearn 1.1→2.x changes relevant to the objects in use: `ColumnTransformer` `verbose_feature_names_out` default change, `set_output()` API added in 1.2+ (not used here, but check `get_feature_names_out` if anything downstream relies on column names).

This bundles with the TF/numpy migration above: the hard ABI failure only appears once numpy is *also* bumped to 2.x, so scikit-learn and numpy/TF should be upgraded together, not independently.

## Finding: `seaborn` (unpinned) is not an issue

Currently resolves to seaborn 0.13.2, whose own metadata requires only `numpy>=1.20`, `pandas>=1.2`, `matplotlib>=3.4` — all loose bounds, no numpy-2.0 conflict. Usages in [exomdn/plotting.py](exomdn/plotting.py) (`PairGrid`, `histplot`) and the notebooks (`FacetGrid`, `kdeplot`) are all stable, long-standing APIs. No action needed; leave unpinned.

## Combined recommendation

- Drop `numpy==1.26.*` now — independent, zero-risk cleanup.
- Leave `scikit-learn==1.1.1` and `seaborn` as-is until the TF/Keras-3 migration is undertaken.
- When doing the full TF 2.16+/numpy-2.x migration, bundle in the scikit-learn bump + preprocessor re-pickling as one combined migration step (steps 1-5 above), since numpy 2.x is what turns the sklearn pickle-compat *warning* into a hard ABI *failure*.
