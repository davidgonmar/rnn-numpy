import numpy as np
import matplotlib.pyplot as plt
from rnn import RNN, RNNForwardState
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


if __name__ == "__main__":
    import argparse

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--single", action="store_true")
    args = arg_parser.parse_args()
    if not args.single:
        rnn = RNN(input_size=1, hidden_size=5, output_size=1)
        rnn2 = RNN(input_size=1, hidden_size=5, output_size=1)
        epochs = 500
        seq_len = 30
        batch_size = 1000

        plt.ion()
        fig, ax = plt.subplots()
        fig2, ax2 = plt.subplots()
        losses = []

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

            # plot loss in a separate window
            ax2.plot(losses)
            ax2.set_title("Loss")

            plt.pause(0.01)

        truncated_steps = 5  # The number of steps to backpropagate through time ()
        assert (
            seq_len % truncated_steps == 0
        ), "seq_len must be divisible by truncated_steps"

        for epoch in range(epochs):
            input_seq, target_seq = generate_sine_wave_sequences(seq_len, batch_size)

            fw_st = RNNForwardState()
            fw_st2 = RNNForwardState()
            avg_loss = 0
            for t in range(seq_len):
                partial_input_seq = input_seq[t, :, :]
                rnn.forward_step(fw_st, input_seq, t)
                rnn2.forward_step(fw_st2, fw_st.get_current_outputs_and_hiddens()[0], t)
                if t % truncated_steps == truncated_steps - 1:
                    preds, hiddens = fw_st2.get_current_outputs_and_hiddens()
                    loss = mse(
                        preds[t - truncated_steps + 1 : t + 1, :, :],
                        target_seq[t - truncated_steps + 1 : t + 1, :, :],
                    )
                    out_grads = mse_bw(
                        preds[t - truncated_steps + 1 : t + 1, :, :],
                        target_seq[t - truncated_steps + 1 : t + 1, :, :],
                    )
                    (
                        ww_grad,
                        wv_grad,
                        wu_grad,
                        bh_grad,
                        bo_grad,
                        inputs_grad,
                    ) = rnn.truncated_backward(
                        preds,
                        hiddens,
                        input_seq,
                        out_grads,
                        t - truncated_steps + 1,
                        t + 1,
                    )
                    (
                        ww_grad2,
                        wv_grad2,
                        wu_grad2,
                        bh_grad2,
                        bo_grad2,
                        inputs_grad2,
                    ) = rnn2.truncated_backward(
                        preds,
                        hiddens,
                        fw_st.get_current_outputs_and_hiddens()[0],
                        out_grads,
                        t - truncated_steps + 1,
                        t + 1,
                    )
                    learning_rate = 0.1 if epoch < 200 else 0.001
                    rnn.update_parameters(
                        learning_rate, ww_grad, wv_grad, wu_grad, bh_grad, bo_grad
                    )
                    rnn2.update_parameters(
                        learning_rate, ww_grad2, wv_grad2, wu_grad2, bh_grad2, bo_grad2
                    )
                    avg_loss += loss

            losses.append(avg_loss / seq_len)

            print(f"Epoch: {epoch} Loss: {avg_loss / seq_len}")

            if epoch % 30 == 0:
                update_ax(input_seq, target_seq, preds, epoch)

        sequences = [
            generate_sine_wave_sequences(seq_len, batch_size) for _ in range(3)
        ]

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
    else:
        rnn = RNN(input_size=1, hidden_size=5, output_size=1)
        epochs = 500
        seq_len = 30
        batch_size = 1000

        plt.ion()
        fig, ax = plt.subplots()
        fig2, ax2 = plt.subplots()
        losses = []

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

            # plot loss in a separate window
            ax2.plot(losses)
            ax2.set_title("Loss")

            plt.pause(0.01)

        truncated_steps = 5  # The number of steps to backpropagate through time ()
        assert (
            seq_len % truncated_steps == 0
        ), "seq_len must be divisible by truncated_steps"

        for epoch in range(epochs):
            input_seq, target_seq = generate_sine_wave_sequences(seq_len, batch_size)

            fw_st = RNNForwardState()
            avg_loss = 0
            for t in range(seq_len):
                partial_input_seq = input_seq[t, :, :]
                rnn.forward_step(fw_st, input_seq, t)

                if t % truncated_steps == truncated_steps - 1:
                    preds, hiddens = fw_st.get_current_outputs_and_hiddens()
                    loss = mse(
                        preds[t - truncated_steps + 1 : t + 1, :, :],
                        target_seq[t - truncated_steps + 1 : t + 1, :, :],
                    )
                    out_grads = mse_bw(
                        preds[t - truncated_steps + 1 : t + 1, :, :],
                        target_seq[t - truncated_steps + 1 : t + 1, :, :],
                    )
                    (
                        ww_grad,
                        wv_grad,
                        wu_grad,
                        bh_grad,
                        bo_grad,
                        inputs_grad,
                    ) = rnn.truncated_backward(
                        preds,
                        hiddens,
                        input_seq,
                        out_grads,
                        t - truncated_steps + 1,
                        t + 1,
                    )
                    learning_rate = 0.1 if epoch < 200 else 0.001
                    rnn.update_parameters(
                        learning_rate, ww_grad, wv_grad, wu_grad, bh_grad, bo_grad
                    )

                    avg_loss += loss

            losses.append(avg_loss / seq_len)

            print(f"Epoch: {epoch} Loss: {avg_loss / seq_len}")

            if epoch % 30 == 0:
                update_ax(input_seq, target_seq, preds, epoch)

        sequences = [
            generate_sine_wave_sequences(seq_len, batch_size) for _ in range(3)
        ]

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
