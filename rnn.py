import numpy as np
from typing import Tuple


class RNN:
    """A simple RNN implementation using numpy"""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
    ):
        """
        Initializates a RNN model with the given parameters

        Args:
            input_size: The size of the input vector
            hidden_size: The size of the hidden layer
            output_size: The size of the output vector
        """

        self.input_size = input_size
        self.hidden_size = hidden_size

        # Initialize the weights using xavier initialization
        self.wu = np.random.normal(
            0, 2 / (input_size + hidden_size), (hidden_size, input_size)
        )
        self.wv = np.random.normal(0, 2 / (2 * hidden_size), (hidden_size, hidden_size))
        self.ww = np.random.normal(
            0, 2 / (hidden_size + output_size), (output_size, hidden_size)
        )

        # Initialize the biases as 0s
        self.bh = np.zeros((hidden_size))
        self.bo = np.zeros((output_size))

    def forward(self, inputs: np.array) -> Tuple[np.array, np.array]:
        """
        Performs a forward pass on the RNN model.
        Inputs must be batched and have the shape (seq_len, batch_size, input_size), even if batch_size = 1
        and seq_len = 1. seq_len and batch_size denote the sequence length and batch size respectively, and
        can be of any size. input_size is the size of the input vector, and must match the input_size parameter
        of the model.

        Args:
            inputs: The input sequence. Shape = (seq_len, batch_size, input_size)

        Returns:
            outputs: The output sequence. Shape = (seq_len, batch_size, output_size)
            hiddens: The hidden states. Shape = (seq_len, batch_size, hidden_size)
        """
        seq_len, batch_size, input_size = inputs.shape
        assert (
            input_size == self.input_size
        ), "input size does not match the model's input_size parameter, got {}".format(
            input_size
        )

        hiddens = []
        outputs = []

        # Initialize the first hidden layer as 0s
        prev_h = np.zeros((batch_size, self.hidden_size))

        for x_t in inputs:
            h = x_t @ self.wu.T + prev_h @ self.wv.T + self.bh
            h = np.tanh(h)
            o = h @ self.ww.T + self.bo
            prev_h = h
            outputs.append(o)
            hiddens.append(h)

        return np.array(outputs), np.array(hiddens)

    def update_parameters(
        self,
        learning_rate: float,
        ww_grad: np.ndarray,
        wv_grad: np.ndarray,
        wu_grad: np.ndarray,
        bh_grad: np.ndarray,
        bo_grad: np.ndarray,
    ) -> None:
        """
        Updates the parameters of the model using the given gradients and learning rate.
        All the provided grads must have the same shape as the corresponding parameters.

        Args:
            learning_rate: The learning rate to use
            ww_grad: The gradient of the output weights. Shape = (output_size, hidden_size)
            wv_grad: The gradient of the hidden weights. Shape = (hidden_size, hidden_size)
            wu_grad: The gradient of the input weights. Shape = (hidden_size, input_size)
            bh_grad: The gradient of the hidden bias. Shape = (hidden_size)
            bo_grad: The gradient of the output bias. Shape = (output_size)
        """

        self.ww -= learning_rate * ww_grad
        self.wv -= learning_rate * wv_grad
        self.wu -= learning_rate * wu_grad
        self.bh -= learning_rate * bh_grad
        self.bo -= learning_rate * bo_grad

    def backward(
        self,
        preds: np.ndarray,
        actuals: np.ndarray,
        hiddens: np.ndarray,
        inputs: np.ndarray,
        out_grads: np.ndarray,
    ):
        """
        Performs a backward pass on the RNN model. preds denote the outputs of the model, actuals denote the
        target outputs, hiddens denote the hidden states, inputs denote the input sequence and out_grads denote
        the gradients of the output sequence, for example, with respect to the loss function.
        It is designed to be used with a forward pass, so inputs should be the same as the inputs of the forward,
        and preds/hiddens should be the outputs of the forward pass.

        Args:
            preds: The predicted sequence. Shape = (seq_len, batch_size, output_size)
            actuals: The actual sequence. Shape = (seq_len, batch_size, output_size)
            hiddens: The hidden states. Shape = (seq_len, batch_size, hidden_size)
            inputs: The input sequence. Shape = (seq_len, batch_size, input_size)
            out_grads: The gradients of the output sequence. Shape = (seq_len, batch_size, output_size)

        Returns:
            ww_grad: The gradient of the output weights. Shape = (output_size, hidden_size)
            wv_grad: The gradient of the hidden weights. Shape = (hidden_size, hidden_size)
            wu_grad: The gradient of the input weights. Shape = (hidden_size, input_size)
            bh_grad: The gradient of the hidden bias. Shape = (hidden_size)
            bo_grad: The gradient of the output bias. Shape = (output_size)
        """

        ww_grad = np.zeros_like(self.ww)
        wv_grad = np.zeros_like(self.wv)
        wu_grad = np.zeros_like(self.wu)
        bh_grad = np.zeros_like(self.bh)
        bo_grad = np.zeros_like(self.bo)

        # Iterate in 'reverse time' for the backpropagation algorithm
        for i in reversed(range(len(preds))):
            hi = hiddens[i]
            out_grad = out_grads[i]

            ww_grad += (hi.T @ out_grad).T
            bo_grad += np.mean(out_grad, axis=0)

            h_grad = out_grad @ self.ww.T.T

            # Backpropagate through the tanh
            h_grad = h_grad * (1 - hi**2)

            wu_grad += (inputs[i].T @ h_grad).T

            if i != 0:
                prev_hidden = hiddens[i - 1]
            else:
                prev_hidden = np.zeros_like(
                    prev_hidden
                )  # this was the first one on the forward pass.

            wv_grad += (prev_hidden.T @ h_grad).T

            bh_grad += np.mean(h_grad, axis=0)

        assert wu_grad.shape == self.wu.shape
        assert wv_grad.shape == self.wv.shape
        assert ww_grad.shape == self.ww.shape
        assert bh_grad.shape == self.bh.shape
        assert bo_grad.shape == self.bo.shape

        # Clip the gradients to prevent exploding gradients
        for grad in [wu_grad, wv_grad, ww_grad, bh_grad, bo_grad]:
            np.clip(grad, -1, 1, out=grad)

        return ww_grad, wv_grad, wu_grad, bh_grad, bo_grad