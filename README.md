# Neural Network from Scratch

A handbuilt neural network for MNIST digit classification — implemented in **pure NumPy**, no ML frameworks. Includes a real-time training visualizer and an interactive Gradio demo.

---

## What's in here

| File | Description |
|---|---|
| `nnfs.py` | Original exploratory implementation — end-to-end in a single script |
| `train.py` | Clean training script; saves weights to `weights.npz` |
| `app.py` | Gradio inference UI — pick a test image, get a prediction |
| `nn_visualizer.py` | Live training visualizer — watch weights update in real time |

---

## Architecture

```
Input      Hidden 1   Hidden 2   Output
784  ──►   10  ──►    10  ──►    1
           ReLU       Softmax
```

- **Loss**: categorical cross-entropy
- **Optimizer**: vanilla gradient descent (lr = 0.1, 1000 epochs)

Backprop is implemented by hand — gradients are derived analytically and computed with NumPy matrix ops.

---

## The math

**Forward pass:**

```
Z1 = W1 · X + B1        (linear)
A1 = ReLU(Z1)           (hidden layer 1)
Z2 = W2 · A1 + B2       (linear)
A2 = Softmax(Z2)        (output probabilities)
```

**Loss:** categorical cross-entropy = `−ln(p)` where `p` is the predicted probability of the true class.

**Backward pass** (derived by hand via chain rule):

```
dZ2 = A2 − Y_one_hot
dW2 = (1/m) · dZ2 · A1ᵀ
dB2 = (1/m) · Σ dZ2

dZ1 = W2ᵀ · dZ2  ⊙  [Z1 > 0]   (ReLU derivative)
dW1 = (1/m) · dZ1 · Xᵀ
dB1 = (1/m) · Σ dZ1
```

---

## Colab

Run it in the browser without any setup:

https://colab.research.google.com/drive/17XhZ4SCnEYh0EbYKBdzUbpiVUZ9hRmaW

---


**venv and package setup**

```bash
python3 -m venv /Users/prashantgulati/Documents/dev/python/nnfs/.venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Github**

Create github repo, create .gitignore, then:

```bash
git init && git remote add origin https://github.com/prashant-gulati/neural-network-from-scratch.git
git add README.md nnfs.py nn_visualizer.py requirements.txt .gitignore && git status
git commit -m "$(cat <<'EOF'
Initial commit: neural network from scratch implementation
Includes core NN implementation, visualizer, and setup instructions.
EOF
)"
git push -u origin main
```
