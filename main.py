import numpy as np
import matplotlib.pyplot as plt
from rnn import RNN
from typing import Tuple


def generate_sine_wave_sequences(
    seq_len: int, batch_size: int
) -> Tuple[np.array, np.array]:
    """
    Generates sine wave sequences with random frequencies and phases.
    Returns two sequences: the input sequence and the target sequence.
    The input sequence is the first half of the sine wave, and the target sequence is a continuation of the
    sine wave.

    Args:
        seq_len: The length of the sequence
        batch_size: The batch size

    Returns:
        inputs: The input sequence. Shape = (seq_len, batch_size, 1)
        targets: The target sequence. Shape = (seq_len, batch_size, 1)
    """
    freqs = np.random.uniform(1, 1.3, size=batch_size)
    phases = np.random.uniform(0, 2 * np.pi, size=batch_size)
    inputs = []
    targets = []
    for freq, phase in zip(freqs, phases):
        delta_t = (2 * np.pi * freq) / (seq_len - 1)  # step size
        t1 = np.linspace(0, 2 * np.pi * freq, seq_len)
        t2 = np.linspace(
            2 * np.pi * freq + delta_t, 4 * np.pi * freq + delta_t, seq_len
        )
        _input = (np.sin(t1 + phase)).reshape((seq_len, 1))
        _target = (np.sin(t2 + phase)).reshape((seq_len, 1))
        inputs.append(_input)
        targets.append(_target)

    inputs = np.array(inputs).transpose((1, 0, 2))
    targets = np.array(targets).transpose((1, 0, 2))

    return inputs, targets


def mse(preds: np.ndarray, actuals: np.ndarray) -> float:
    """
    Computes the mean squared error between the predictions and the actuals.

    Args:
        preds: The predictions of the model. Shape = (seq_len, batch_size, output_size)
        actuals: The actual values. Shape = (seq_len, batch_size, output_size)

    Returns:
        The mean squared error between the predictions and the actuals (scalar)
    """
    return np.mean((actuals - preds) ** 2)


def mse_bw(preds: np.ndarray, actuals: np.ndarray) -> np.ndarray:
    """
    Computes the gradient of the predictions with respect to the mean squared error loss function.

    Args:
        preds: The predictions of the model. Shape = (seq_len, batch_size, output_size)
        actuals: The actual values. Shape = (seq_len, batch_size, output_size)

    Returns:
        The gradient of predictions with respect to the mean squared error loss function.
        Shape = (seq_len, batch_size, output_size)
    """
    N = np.prod(preds.shape)
    out = 2 * (preds - actuals) / N
    return out


rnn = RNN(input_size=1, hidden_size=5, output_size=1)
epochs = 500
seq_len = 30
batch_size = 1000


plt.ion()
fig, ax = plt.subplots()


def update_ax(input_seq, target_seq, preds, epoch):
    ax.clear()
    inputs = np.squeeze(input_seq[:, 0, :])
    inputs_x = np.arange(0, len(inputs))
    targets = np.squeeze(target_seq[:, 0, :])
    targets_x = np.arange(len(inputs), len(targets) + len(inputs))
    preds = np.squeeze(preds[:, 0, :])
    preds_x = np.arange(len(inputs), len(targets) + len(inputs))

    ax.plot(inputs_x, inputs, label="Input (Sine Wave)")
    ax.plot(targets_x, targets, label="Target")
    ax.plot(preds_x, preds, label="Predicted")
    ax.legend()
    ax.set_title(f"After Training Epoch: {epoch}")
    plt.pause(0.01)


for epoch in range(epochs):
    input_seq, target_seq = generate_sine_wave_sequences(seq_len, batch_size)

    preds, hiddens = rnn.forward(input_seq)

    loss = mse(preds, target_seq)

    out_grads = mse_bw(preds, target_seq)

    ww_grad, wv_grad, wu_grad, bh_grad, bo_grad = rnn.backward(
        preds, target_seq, hiddens, input_seq, out_grads
    )
    learning_rate = 0.1 if epoch < 200 else 0.001
    rnn.update_parameters(learning_rate, ww_grad, wv_grad, wu_grad, bh_grad, bo_grad)
    print(f"Epoch: {epoch} Loss: {loss}")

    if epoch % 30 == 0:
        update_ax(input_seq, target_seq, preds, epoch)


sequences = [generate_sine_wave_sequences(seq_len, batch_size) for _ in range(3)]


fig, axes = plt.subplots(3, 1, figsize=(5, 10))
plt.ioff()

for i, (input_seq, target_seq) in enumerate(sequences):
    preds, _ = rnn.forward(input_seq)

    inputs = np.squeeze(input_seq[:, 0, :])
    inputs_x = np.arange(0, len(inputs))

    targets = np.squeeze(target_seq[:, 0, :])
    targets_x = np.arange(len(inputs), 2 * len(inputs))

    preds = np.squeeze(preds[:, 0, :])
    preds_x = np.arange(len(inputs), 2 * len(inputs))

    # Plot on the i-th subplot
    axes[i].plot(inputs_x, inputs, label="Input (Sine Wave)")
    axes[i].plot(targets_x, targets, label="Target")
    axes[i].plot(preds_x, preds, label="Predicted")
    axes[i].legend()
    axes[i].set_title(f"Sequence {i+1} After Training")

plt.tight_layout()
plt.show()
