import numpy as np
import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

# ── Load MNIST once at startup ────────────────────────────────────────────────
print("Loading MNIST dataset...")
from datasets import load_dataset

_ds = load_dataset("mnist")

def _flatten(split):
    imgs = np.array([np.array(img).flatten() for img in _ds[split]["image"]], dtype=np.float32)
    labels = np.array(_ds[split]["label"])
    return imgs.T / 255.0, labels   # shape: (784, N), (N,)

X_test, Y_test = _flatten("test")    # (784, 10000)
print(f"MNIST loaded — test: {X_test.shape}")

# ── Pre-generate a gallery of 100 test images ─────────────────────────────────
GALLERY_SIZE = 100
np.random.seed(0)
gallery_indices = np.random.choice(X_test.shape[1], GALLERY_SIZE, replace=False)

def _to_pil(col_vec):
    """784-float column → upscaled RGB PIL image for display."""
    arr = (col_vec.reshape(28, 28) * 255).astype(np.uint8)
    return Image.fromarray(arr, mode="L").resize((112, 112), Image.NEAREST).convert("RGB")

gallery_images = [_to_pil(X_test[:, i]) for i in gallery_indices]
gallery_labels = [str(int(Y_test[i])) for i in gallery_indices]


# ── Pure-NumPy neural network (784 → 10 → 10) ─────────────────────────────────

def forward(W1, B1, W2, B2, X):
    Z1 = W1.dot(X) + B1
    A1 = np.maximum(Z1, 0)                                    # ReLU
    Z2 = W2.dot(A1) + B2
    e  = np.exp(Z2 - np.max(Z2, axis=0, keepdims=True))
    A2 = e / e.sum(axis=0, keepdims=True)                     # softmax
    return A2


# ── Load pre-trained weights ──────────────────────────────────────────────────
print("Loading pre-trained weights...")
_w = np.load("weights.npz")
W1, B1, W2, B2 = _w["W1"], _w["B1"], _w["W2"], _w["B2"]
_test_acc = float(_w["test_acc"])
print(f"Weights loaded — test accuracy: {_test_acc*100:.1f}%")


# ── Gradio callback ───────────────────────────────────────────────────────────

def predict_digit(gallery_pos):
    if gallery_pos is None:
        return None, "Select an image from the gallery above."

    test_idx = gallery_indices[int(gallery_pos)]
    x        = X_test[:, test_idx].reshape(784, 1)
    actual   = int(Y_test[test_idx])

    A2 = forward(W1, B1, W2, B2, x)
    probs = A2.flatten()
    pred  = int(np.argmax(probs))

    correct = pred == actual
    colors  = ["crimson" if i == pred else "steelblue" for i in range(10)]
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.bar(range(10), probs * 100, color=colors)
    ax.set_xlabel("Digit")
    ax.set_ylabel("Confidence (%)")
    ax.set_title(f"Predicted: {pred}  |  Actual: {actual}  ({'✓' if correct else '✗'})")
    ax.set_xticks(range(10))
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()

    verdict = "correct" if correct else "wrong"
    label = (f"Predicted: **{pred}** ({probs[pred]*100:.1f}%)  —  "
             f"Actual: **{actual}**  — {verdict}")
    return fig, label


# ── Gradio UI ─────────────────────────────────────────────────────────────────

with gr.Blocks(title="Neural Network from Scratch") as demo:

    selected_pos = gr.State(None)   # index into gallery_indices

    gr.Markdown(
        "# Neural Network from Scratch\n"
        f"A 2-layer network (784 → 10 → 10) trained on MNIST using **pure NumPy** — "
        f"no ML frameworks. Test accuracy after 1000 epochs: **{_test_acc*100:.1f}%**."
    )

    gallery = gr.Gallery(
        value=[(img, lbl) for img, lbl in zip(gallery_images, gallery_labels)],
        label="100 random test images  (true label shown on hover)",
        columns=10,
        rows=10,
        height="auto",
        allow_preview=False,
        show_label=True,
    )
    predict_btn = gr.Button("Predict selected image", variant="primary")
    with gr.Row():
        pred_plot  = gr.Plot(label="Class probabilities")
        pred_label = gr.Markdown()

    def on_select(evt: gr.SelectData):
        return evt.index

    gallery.select(fn=on_select, outputs=selected_pos)

    predict_btn.click(
        fn=predict_digit,
        inputs=[selected_pos],
        outputs=[pred_plot, pred_label],
    )

demo.launch()
