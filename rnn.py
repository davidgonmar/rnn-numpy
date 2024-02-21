import numpy as np
from typing import Tuple


class RNNBackwardState:
    def __init__(self, rnn: "RNN", seq_len: int, batch_size: int):
        self.ww_grad = np.zeros_like(rnn.ww)
        self.wv_grad = np.zeros_like(rnn.wv)
        self.wu_grad = np.zeros_like(rnn.wu)
        self.bh_grad = np.zeros_like(rnn.bh)
        self.bo_grad = np.zeros_like(rnn.bo)
        self.input_grad = np.zeros((seq_len, batch_size, rnn.input_size))

class RNNForwardState:
    def __init__(self):
        self.hiddens = []
        self.outputs = []
    
    def get_current_outputs_and_hiddens(self, start_time: int = 0, end_time: int = None):
        return np.array(self.outputs[start_time:end_time]), np.array(self.hiddens[start_time:end_time])

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

    def forward_step(self, st: RNNForwardState, inputs: np.array, time: int) -> Tuple[np.ndarray, np.ndarray]:
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

        # Initialize the first hidden layer as 0s
        prev_h = np.zeros((batch_size, self.hidden_size)) if time == 0 else st.hiddens[-1]
        x_t = inputs[time]
        h = x_t @ self.wu.T + prev_h @ self.wv.T + self.bh
        h = np.tanh(h)
        o = h @ self.ww.T + self.bo
        st.outputs.append(o)
        st.hiddens.append(h)

    def forward(self, inputs: np.array) -> Tuple[np.ndarray, np.ndarray]:
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
        st = RNNForwardState()
        for t in range(len(inputs)):
            self.forward_step(st, inputs, t)
        return np.array(st.outputs), np.array(st.hiddens)

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

    def backward_step(
        self,
        st: RNNBackwardState,
        hidden: np.ndarray,
        prev_hidden: np.ndarray,
        input: np.ndarray,
        out_grad: np.ndarray,
    ):
        """
        Performs a single backward step on the RNN model.
        Modifies the gradients in the backward state.
        Args:
            st: The backward state
            hidden: The hidden state. Shape = (batch_size, hidden_size)
            prev_hidden: The previous hidden state. Shape = (batch_size, hidden_size)
            input: The input sequence. Shape = (batch_size, input_size)
            out_grad: The gradients of the output sequence. Shape = (batch_size, output_size)

        Returns:
            None

        """
        st.ww_grad += (hidden.T @ out_grad).T
        st.bo_grad += np.mean(out_grad, axis=0)

        h_grad = out_grad @ self.ww.T.T

        # Backpropagate through the tanh
        h_grad = h_grad * (1 - hidden**2)
        st.wu_grad += (input.T @ h_grad).T

        st.wv_grad += (prev_hidden.T @ h_grad).T

        st.bh_grad += np.mean(h_grad, axis=0)

        st.input_grad += h_grad @ self.wu.T.T
    
    def backward(
        self,
        preds: np.ndarray,
        hiddens: np.ndarray,
        inputs: np.ndarray,
        out_grads: np.ndarray,

    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return self.truncated_backward(preds, hiddens, inputs, out_grads, 0, None)

    def truncated_backward(
        self,
        preds: np.ndarray,
        hiddens: np.ndarray,
        inputs: np.ndarray,
        out_grads: np.ndarray,
        start_time: int = 0,
        end_time: int = None,
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

        st = RNNBackwardState(self, preds.shape[0], preds.shape[1])
        # Iterate in 'reverse time' for the backpropagation algorithm
        for i in reversed(range(start_time, end_time or len(preds))):
            hidden = hiddens[i]
            # out_grad is local, so it doesnt have lenght of seq_len, but of the truncated steps size
            out_grad = out_grads[i - start_time]
            input = inputs[i]
            if i != 0:
                prev_hidden = hiddens[i - 1]
            else:
                prev_hidden = np.zeros_like(
                    hidden
                )  # this was the first one on the forward pass.
            self.backward_step(st, hidden, prev_hidden, input, out_grad)

        assert st.wu_grad.shape == self.wu.shape
        assert st.wv_grad.shape == self.wv.shape
        assert st.ww_grad.shape == self.ww.shape
        assert st.bh_grad.shape == self.bh.shape
        assert st.bo_grad.shape == self.bo.shape
        partial_inputs_shape = (preds.shape[0], preds.shape[1], self.input_size) # since it is truncated, we need to return the partial inputs shape
        assert st.input_grad.shape == partial_inputs_shape

        # Clip the gradients to prevent exploding gradients
        st.ww_grad = np.clip(st.ww_grad, -1, 1)
        st.wv_grad = np.clip(st.wv_grad, -1, 1)
        st.wu_grad = np.clip(st.wu_grad, -1, 1)
        st.bh_grad = np.clip(st.bh_grad, -1, 1)
        st.bo_grad = np.clip(st.bo_grad, -1, 1)
        st.input_grad = np.clip(st.input_grad, -1, 1)

        return st.ww_grad, st.wv_grad, st.wu_grad, st.bh_grad, st.bo_grad, st.input_grad
