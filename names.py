import os
from collections import OrderedDict
import unicodedata
import string
from typing import List
import numpy as np
from rnn import RNNForwardState, RNN, RNNBackwardState
import matplotlib.pyplot as plt


# To run this, you must download the data from https://download.pytorch.org/tutorial/data.zip and put
# it in the ./data/names folder
def cross_entropy_loss(preds: np.ndarray, actuals: np.ndarray) -> float:
    """
    preds is array of shape (output_size,)
    """
    preds = np.exp(preds) / np.sum(np.exp(preds), keepdims=False)

    idx = np.argmax(actuals)
    return -np.log(preds[idx])


def cross_entropy_loss_bw(preds: np.ndarray, actuals: np.ndarray) -> np.ndarray:
    """
    preds is array of shape (output_size,)
    """
    softmax = np.exp(preds) / np.sum(np.exp(preds), keepdims=False)

    idx = np.argmax(actuals)

    softmax[idx] -= 1

    return softmax


all_letters = string.ascii_lowercase + " .,;'"
n_letters = len(all_letters)


def unicodeToAscii(s):
    return "".join(
        c.lower()
        for c in unicodedata.normalize("NFD", s)
        if unicodedata.category(c) != "Mn" and c.lower() in all_letters
    )


char2tok = {v: k for k, v in enumerate(all_letters)}
tok2char = {v: k for k, v in char2tok.items()}

DATA_FOLDER = os.path.join(os.path.dirname(__file__), ".", "data", "names")

names = OrderedDict()  # key = nationality, value = list of names


plt.ion()

plot = False
max_classes = 100
for path in os.listdir(DATA_FOLDER):
    nationality = os.path.splitext(path)[0]
    if len(names) == max_classes:
        break
    with open(
        os.path.join(DATA_FOLDER, path), mode="r", encoding="utf-8", errors="ignore"
    ) as f:
        names[nationality] = [unicodeToAscii(line) for line in f.read().splitlines()]


def one_hot(idxs: np.ndarray, n_classes: int) -> np.ndarray:
    one_hot = np.zeros((len(idxs), n_classes))
    one_hot[np.arange(len(idxs)), idxs] = 1
    return one_hot


def tokenize_names(names: List[str]):
    return [one_hot([char2tok[char] for char in name], n_letters) for name in names]


def untokenize_names(names: List[np.ndarray]):
    return [
        "".join([tok2char[np.argmax(one_hot)] for one_hot in name]) for name in names
    ]


tokenized_names = {na: tokenize_names(li) for na, li in names.items()}

iters = 300000

class_to_idx = {na: i for i, na in enumerate(tokenized_names.keys())}
idx_to_class = {i: na for i, na in enumerate(tokenized_names.keys())}

n_classes = len(tokenized_names.keys())
rnn = RNN(n_letters, 10, n_classes)


def get_random_sample():
    random_class = np.random.choice(list(tokenized_names.keys()))
    random_name = np.random.choice(tokenized_names[random_class])
    return class_to_idx[random_class], random_name


def lr(step: int) -> float:
    return 0.01 / (1 + 0.005 * (step / 100))


losses = []
last_avg_loss = 0
for iter in range(iters):
    target_class, input_name = get_random_sample()
    target_class_oh = one_hot([target_class], n_classes)
    # simulate batch size of 1 since network expects batched input of shape (seq_len, batch_size, input_size)
    input_name = input_name.reshape(input_name.shape[0], 1, input_name.shape[1])
    fw_st = RNNForwardState()

    preds, hiddens = rnn.forward(input_name)
    preds = preds[-1][0]  # only select the last prediction

    loss = cross_entropy_loss(preds, target_class_oh)
    last_avg_loss += loss
    if iter % 1000 == 0:
        print(f"iter {iter} loss: {loss/100}")
        losses.append(last_avg_loss / 100)
        last_avg_loss = 0
        if plot:
            plt.plot(losses)
            plt.pause(0.001)
            plt.show()
    out_grad = cross_entropy_loss_bw(preds, target_class_oh)

    bw_st = RNNBackwardState(rnn, len(input_name), 1)  # simulate batch size of 1

    for t in reversed(range(len(input_name))):
        # we'll only pass out grad if we're at the last time step
        out_grad = np.zeros_like(preds) if t < len(input_name) - 1 else out_grad
        out_grad = out_grad.reshape(1, -1)
        hidden = hiddens[t]
        prev_hidden = hiddens[t - 1] if t > 0 else np.zeros_like(hidden)
        rnn.backward_step(bw_st, hidden, prev_hidden, input_name[t], out_grad)

    rnn.update_parameters(
        lr(iter),
        bw_st.ww_grad,
        bw_st.wv_grad,
        bw_st.wu_grad,
        bw_st.bh_grad,
        bw_st.bo_grad,
    )


# ask for a name
while True:
    name = input("Enter a name: ")
    name = unicodeToAscii(name)
    name = one_hot([char2tok[char] for char in name], n_letters)
    name = name.reshape(name.shape[0], 1, name.shape[1])
    preds, _ = rnn.forward(name)
    preds = preds[-1][0]

    # softmax
    preds = np.exp(preds - np.max(preds)) / np.sum(np.exp(preds - np.max(preds)))
    top3 = np.argsort(preds)[-3:][::-1]

    for i in top3:
        print(f"predicted {idx_to_class[i]}: {preds[i]}")
