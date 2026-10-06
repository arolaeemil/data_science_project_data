from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


def normalize(
    values: torch.Tensor,
    mean: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    return (values - mean) / scale


class RNNRegressor(nn.Module):
    """RNN that predicts a target after a fixed-size input window."""

    def __init__(self, hidden_size: int = 32) -> None:
        super().__init__()
        self.rnn = nn.RNN(input_size=1, hidden_size=hidden_size, batch_first=True)
        self.output = nn.Linear(hidden_size, 1)
        self.register_buffer("mean", torch.zeros(1))
        self.register_buffer("scale", torch.ones(1))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        output, _ = self.rnn(values)
        return self.output(output[:, -1, :])

    @torch.inference_mode()
    def predict_next(self, values: Sequence[float]) -> float:
        """Predict the value following the last values in a sequence."""
        if len(values) == 0:
            raise ValueError("values must contain at least one number")

        inputs = torch.tensor(values, dtype=torch.float32)
        inputs = normalize(inputs, self.mean, self.scale).reshape(1, -1, 1)
        prediction = self(inputs).squeeze()
        return float(prediction * self.scale + self.mean)


def train(
    values: Sequence[float],
    sequence_length: int = 5,
    target_offset: int = 1,
    epochs: int = 500,
    learning_rate: float = 0.01,
    hidden_size: int = 32,
) -> RNNRegressor:
    """Train an RNN to predict a future value in ``values``.

    ``sequence_length`` controls how many previous values are used for each
    prediction. ``target_offset=1`` predicts the value immediately after the
    window; larger offsets predict farther into the future. The returned model
    can be used with ``model.predict_next``.
    """
    if sequence_length < 1:
        raise ValueError("sequence_length must be at least 1")
    if target_offset < 1:
        raise ValueError("target_offset must be at least 1")
    if epochs < 1:
        raise ValueError("epochs must be at least 1")
    if len(values) <= sequence_length + target_offset - 1:
        raise ValueError(
            "values must contain enough items for sequence_length and target_offset"
        )

    data = torch.tensor(values, dtype=torch.float32)
    if not torch.isfinite(data).all():
        raise ValueError("values must contain only finite numbers")

    model = RNNRegressor(hidden_size=hidden_size)
    model.mean.copy_(data.mean())
    model.scale.copy_(data.std().clamp_min(1e-6))
    normalized = normalize(data, model.mean, model.scale)

    inputs = torch.stack(
        [
            normalized[index : index + sequence_length]
            for index in range(len(data) - sequence_length - target_offset + 1)
        ]
    ).unsqueeze(-1)
    target_index = sequence_length + target_offset - 1
    targets = normalized[target_index:].unsqueeze(-1)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_function = nn.MSELoss()
    model.train()
    for i in range(epochs):
        optimizer.zero_grad()
        loss = loss_function(model(inputs), targets)
        loss.backward()
        optimizer.step()
        print(f"Epoch {i + 1}/{epochs} - loss: {loss.item():.6f}")

    model.eval()
    return model


def train_further(
    model: RNNRegressor,
    values: Sequence[float],
    sequence_length: int = 5,
    target_offset: int = 1,
    epochs: int = 500,
    learning_rate: float = 0.01,
) -> RNNRegressor:
    """Continue training ``model`` on additional values."""
    if sequence_length < 1:
        raise ValueError("sequence_length must be at least 1")
    if target_offset < 1:
        raise ValueError("target_offset must be at least 1")
    if epochs < 1:
        raise ValueError("epochs must be at least 1")
    if len(values) <= sequence_length + target_offset - 1:
        raise ValueError(
            "values must contain enough items for sequence_length and target_offset"
        )

    data = torch.tensor(values, dtype=torch.float32, device=model.mean.device)
    if not torch.isfinite(data).all():
        raise ValueError("values must contain only finite numbers")

    normalized = normalize(data, model.mean, model.scale)
    inputs = torch.stack(
        [
            normalized[index : index + sequence_length]
            for index in range(len(data) - sequence_length - target_offset + 1)
        ]
    ).unsqueeze(-1)
    target_index = sequence_length + target_offset - 1
    targets = normalized[target_index:].unsqueeze(-1)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_function = nn.MSELoss()
    model.train()
    for i in range(epochs):
        optimizer.zero_grad()
        loss = loss_function(model(inputs), targets)
        loss.backward()
        optimizer.step()
        print(f"Epoch {i + 1}/{epochs} - loss: {loss.item():.6f}")

    model.eval()
    return model


def evaluate(
    model: RNNRegressor,
    values: Sequence[float],
    sequence_length: int = 5,
    target_offset: int = 1,
):
    """Return the mean squared prediction error in the original value scale."""
    if sequence_length < 1:
        raise ValueError("sequence_length must be at least 1")
    if target_offset < 1:
        raise ValueError("target_offset must be at least 1")
    if len(values) <= sequence_length + target_offset - 1:
        raise ValueError(
            "values must contain enough items for sequence_length and target_offset"
        )

    data = torch.tensor(values, dtype=torch.float32, device=model.mean.device)
    if not torch.isfinite(data).all():
        raise ValueError("values must contain only finite numbers")

    normalized = normalize(data, model.mean, model.scale)
    inputs = torch.stack(
        [
            normalized[index : index + sequence_length]
            for index in range(len(data) - sequence_length - target_offset + 1)
        ]
    ).unsqueeze(-1)
    target_index = sequence_length + target_offset - 1
    targets = data[target_index:]

    model.eval()
    with torch.inference_mode():
        predictions = model(inputs).squeeze(-1)
        predictions = predictions * model.scale + model.mean
        error = torch.mean((predictions - targets) ** 2)
    return float(error)


if __name__ == "__main__":
    example = [1, 2, 3, 4, 5, 6, 7, 8]
    model = train(example, sequence_length=3, target_offset=2)
    print(model.predict_next(example[-3:]))
