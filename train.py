"""Run once to train the model and save weights to weights.npz."""
import numpy as np
from datasets import load_dataset

# ── Load MNIST ────────────────────────────────────────────────────────────────
print("Loading MNIST...")
_ds = load_dataset("mnist")

def _flatten(split):
    imgs = np.array([np.array(img).flatten() for img in _ds[split]["image"]], dtype=np.float32)
    return imgs.T / 255.0, np.array(_ds[split]["label"])

X_train, Y_train = _flatten("train")
X_test,  Y_test  = _flatten("test")

# ── Neural network ────────────────────────────────────────────────────────────
def relu(X):      return np.maximum(X, 0)

def softmax(Z):
    e = np.exp(Z - np.max(Z, axis=0, keepdims=True))
    return e / e.sum(axis=0, keepdims=True)

def one_hot(Y):
    oh = np.zeros((Y.size, int(Y.max()) + 1))
    oh[np.arange(Y.size), Y.astype(int)] = 1
    return oh.T

def forward(W1, B1, W2, B2, X):
    Z1 = W1.dot(X) + B1;  A1 = relu(Z1)
    Z2 = W2.dot(A1) + B2; A2 = softmax(Z2)
    return Z1, A1, Z2, A2

def backward(W1, B1, W2, B2, Z1, A1, A2, X, Y):
    m   = X.shape[1]
    dZ2 = A2 - one_hot(Y)
    dW2 = dZ2.dot(A1.T) / m
    dB2 = np.sum(dZ2, axis=1, keepdims=True) / m
    dZ1 = W2.T.dot(dZ2) * (Z1 > 0)
    dW1 = dZ1.dot(X.T) / m
    dB1 = np.sum(dZ1, axis=1, keepdims=True) / m
    return dW1, dB1, dW2, dB2

# ── Train ─────────────────────────────────────────────────────────────────────
EPOCHS = 1000
LR     = 0.1

np.random.seed(42)
W1 = np.random.rand(10, 784) - 0.5
B1 = np.random.rand(10, 1)   - 0.5
W2 = np.random.rand(10, 10)  - 0.5
B2 = np.random.rand(10, 1)   - 0.5

print(f"Training ({EPOCHS} epochs, lr={LR})...")
for i in range(EPOCHS):
    Z1, A1, _, A2 = forward(W1, B1, W2, B2, X_train)
    dW1, dB1, dW2, dB2 = backward(W1, B1, W2, B2, Z1, A1, A2, X_train, Y_train)
    W1 -= LR * dW1;  B1 -= LR * dB1
    W2 -= LR * dW2;  B2 -= LR * dB2
    if (i + 1) % 100 == 0:
        _, _, _, A2 = forward(W1, B1, W2, B2, X_train)
        acc = np.mean(np.argmax(A2, axis=0) == Y_train)
        print(f"  epoch {i+1:4d}  train acc: {acc*100:.1f}%")

_, _, _, A2_test = forward(W1, B1, W2, B2, X_test)
test_acc = np.mean(np.argmax(A2_test, axis=0) == Y_test)
print(f"Test accuracy: {test_acc*100:.1f}%")

# ── Save ──────────────────────────────────────────────────────────────────────
np.savez("weights.npz", W1=W1, B1=B1, W2=W2, B2=B2, test_acc=test_acc)
print("Weights saved to weights.npz")
