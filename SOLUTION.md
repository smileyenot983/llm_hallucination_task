# SMILES-2026 Hallucination Detection Solution

## Reproducibility

Run from the repository root:

```bash
pip install -r requirements.txt
python solution.py
```

The script loads `Qwen/Qwen2.5-0.5B`, extracts hidden states for
`data/dataset.csv` and `data/test.csv`, evaluates the probe, writes
`results.json`, and writes `predictions.csv`.

## Approach

`aggregation.py` replaces the final-token-only baseline with a richer hidden
state representation. It selects three spaced late transformer layers and
concatenates three views from each selected layer: the last real token, the
mean over the final 48 tokens, and the mean over the full non-padding
sequence. The tail-window mean is a practical proxy for response-focused
pooling because `solution.py` passes hidden states and masks, but not token
IDs or prompt lengths, into the aggregation function. A few scalar features
for sequence length, activation scale, and final-layer drift are appended
directly inside `aggregate`, so they are used even though `USE_GEOMETRIC`
remains `False` in the fixed pipeline.

`probe.py` uses a deterministic, regularized sklearn probe instead of training
a larger neural network on the small labeled dataset. Features are
standardized, compressed with a fast truncated-SVD projection when the feature
dimension is large, and classified with one class-balanced logistic-regression
probe. The low-dimensional scalar features appended by `aggregate` are kept
outside the SVD projection and concatenated back into each reduced feature
vector. The validation split is used only to tune the prediction threshold.

`splitting.py` uses five stratified folds. Each fold has a stratified held-out
test split and a stratified validation split carved from the remaining
examples for threshold tuning. This gives a more stable estimate than a single
random split while preserving the class ratio.

## Experiments and Decisions

The starting point used only the final transformer layer's last token and a
single hidden-layer MLP. That setup is simple, but it is fragile for this
dataset because there are only 689 labeled examples and the hallucination
signal may be distributed across multiple layers and token positions.

The final implementation keeps the classifier linear after dimensionality
reduction because it is less prone to memorizing the training fold than an MLP
on thousands of raw features. Using three spaced late transformer layers keeps
the representation smaller than the original five-layer version while
preserving signal from multiple late stages of answer formation. Additional
geometric features are implemented, but the core scalar
features are included directly in `aggregate` so the fixed `solution.py`
configuration uses them. Validation-based model selection was avoided because
the validation folds are small and can overstate improvements that do not carry
to the held-out folds.

## Expected Outputs

Successful execution of `python solution.py` should produce:

- `results.json`
- `predictions.csv`
