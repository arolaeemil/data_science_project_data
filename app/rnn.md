# RNN next-value predictor

`app/rnn.py` contains a small recurrent neural network (RNN) for learning a sequence of numeric values and predicting the value that comes next.

The module is designed for one-dimensional sequences such as:

```python
values = [10, 12, 15, 14, 18, 21, 20, 24]
```

It uses PyTorch and trains on overlapping windows from the supplied list.

## Quick start

From the project root:

```python
from app.rnn import train

values = [1, 2, 3, 4, 5, 6, 7, 8]
model = train(values, sequence_length=3)
prediction = model.predict_next(values[-3:])

print(prediction)
```

The model receives `[6, 7, 8]` and predicts a future value relative to that window. By default, it predicts the value immediately after the window. The prediction is a floating-point approximation, not a guaranteed exact value.

The same example can be run directly from the project root:

```bash
python app/rnn.py
```

## Public API

### `train`

```python
train(
    values,
    sequence_length=5,
    target_offset=1,
    epochs=500,
    learning_rate=0.01,
    hidden_size=32,
) -> RNNRegressor
```

`train` creates and trains an `RNNRegressor` from a sequence of numbers.

| Argument | Description |
| --- | --- |
| `values` | A sequence of finite numeric values. A list, tuple, or similar sequence can be used. |
| `sequence_length` | Number of previous values in each input window. It must be at least `1`, and `values` must contain more items than this value. |
| `target_offset` | How far after the end of each input window the target is located. `1` predicts the immediately following value; `2` predicts the second value after the window; larger positive values predict farther ahead. |
| `epochs` | Number of complete passes over the generated training windows. Higher values can improve fitting but take longer. Must be at least `1`. |
| `learning_rate` | Step size used by the Adam optimizer. `0.01` is the default. |
| `hidden_size` | Number of values stored in the RNN hidden state. A larger value gives the model more capacity but uses more parameters. |

The function returns a trained `RNNRegressor` in evaluation mode.

### `RNNRegressor.predict_next`

```python
prediction = model.predict_next(values)
```

`predict_next` accepts the values to use as the input window and returns one Python `float`. It returns the future horizon selected when `train` was called. For example, a model trained with `target_offset=5` predicts the fifth value after its input window. The method can receive the same number of values as `sequence_length`, although the underlying RNN can technically process a different window length too.

An empty input raises `ValueError`.

## How training data is made

Suppose the input is:

```python
values = [1, 2, 3, 4, 5]
sequence_length = 2
target_offset = 2
```

The code creates these input/target pairs:

```text
input       target
[1, 2]      4
[2, 3]      5
```

Each input window is one training example. With `target_offset=2`, one value is skipped between the end of the input window and its target. In general, the target index is `window_start + sequence_length + target_offset - 1`, so at least `sequence_length + target_offset` values are required.

## Model structure

`RNNRegressor` contains three important pieces:

1. `torch.nn.RNN` reads the sequence one value at a time. Its input size is `1` because every time step contains one numeric feature.
2. The final hidden representation from the last time step is passed to a linear layer.
3. The linear layer produces one output: the predicted next value.

The RNN is configured with `batch_first=True`, so tensors use this shape:

```text
(batch_size, sequence_length, number_of_features)
```

For this module, the training input has the shape:

```text
(number_of_windows, sequence_length, 1)
```

The model output has the shape:

```text
(number_of_windows, 1)
```

## Normalization

Before training, the values are standardized using their mean and standard deviation:

```text
normalized_value = (value - mean) / scale
```

Training on normalized values generally makes optimization more stable, especially when the original values are large. The mean and scale are stored in the model as buffers, so they move with the model if it is moved to another device or saved.

When `predict_next` is called, its inputs are normalized with the same training statistics. The resulting prediction is converted back to the original value scale before it is returned.

For a constant input sequence, the scale is clamped to a small positive value to avoid division by zero.

## Validation and errors

`train` raises `ValueError` when:

- `values` does not contain more items than `sequence_length`;
- `sequence_length` is less than `1`;
- `target_offset` is less than `1`;
- `epochs` is less than `1`; or
- any value is `NaN`, positive infinity, or negative infinity.

`predict_next` raises `ValueError` when it receives an empty sequence.

The function does not split data into training and test sets. It trains on every window generated from the supplied list, so a separate validation sequence should be used when measuring how well the model generalizes.

## Choosing parameters

- Use a larger `sequence_length` when a prediction depends on a longer recent history.
- Use a smaller `sequence_length` for short lists or simple local patterns.
- Increase `target_offset` when the required prediction is farther into the future. This also reduces the number of available training windows.
- Increase `epochs` if the model is underfitting, but expect training to take longer.
- Increase `hidden_size` only when the sequence has more complicated patterns; larger models are not automatically more accurate.
- Keep the learning rate near `0.01` initially. If training is unstable, try a smaller value such as `0.001`.

The default `epochs=500` is suitable for small examples, but it may be too slow for many independent sequences or very large datasets.

## Predicting several future values

The model predicts one value at a time. To forecast multiple future values, append each prediction to the current sequence and use the updated last window for the next prediction:

```python
from app.rnn import train

values = [1, 2, 3, 4, 5, 6, 7, 8]
model = train(values, sequence_length=3, target_offset=2)

forecast = []
window = list(values[-3:])
for _ in range(3):
    next_value = model.predict_next(window)
    forecast.append(next_value)
    window = window[1:] + [next_value]

print(forecast)
```

This is recursive forecasting: later predictions depend on earlier predictions, so errors can accumulate.

## Reproducibility

PyTorch initializes the model weights randomly. Two calls to `train` can therefore produce slightly different predictions. For repeatable experiments, set a seed before training:

```python
import torch
from app.rnn import train

torch.manual_seed(42)
model = train([1, 2, 3, 4, 5, 6], sequence_length=2)
```
