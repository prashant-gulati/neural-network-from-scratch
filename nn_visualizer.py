import asyncio
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import uvicorn

# ---- Data ----
data = pd.read_csv('train.csv')
label = data['label']
features = data.drop(columns=['label']) / 255.0
n_features = features.shape[1]

X_train, _, Y_train, _ = train_test_split(features, label, test_size=0.2)
X_train = torch.tensor(X_train.values, dtype=torch.float32)
Y_train = torch.tensor(Y_train.values, dtype=torch.long)

train_dataset = torch.utils.data.TensorDataset(X_train, Y_train)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=True)

# ---- Model ----
class MNISTNet(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.fc1 = nn.Linear(n_features, 10)
        self.fc2 = nn.Linear(10, 10)
        self.fc3 = nn.Linear(10, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        # applies a second RELU to the output before applying softmax to it
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

model = MNISTNet(n_features)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ---- Training state ----
training_mode = False
batch_iterator = iter(train_loader)
batch_count = 0
correct = 0
VIS = 20        # neurons to display per input/hidden layer
STEP_DELAY = 0.6  # seconds between training steps (increase to slow down)

# pytorch softmax - more numerically stable
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return (e_x / e_x.sum()).tolist()

#
def train_step():
    global batch_iterator, batch_count, correct
    try:
        batch_data, target = next(batch_iterator)
    except StopIteration:
        batch_iterator = iter(train_loader)
        batch_data, target = next(batch_iterator)

    optimizer.zero_grad()
    output = model(batch_data)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()

    pixels = batch_data[0].tolist()

    # weights[li][fi][ti] = weight from neuron fi in layer li to neuron ti in layer li+1
    # tanh-normalize so values are in (-1, 1) for consistent color mapping
    def vis_weights(layer_weight, max_from, max_to):
        w = layer_weight.detach().cpu().numpy()   # shape: (to, from)
        w = w[:max_to, :max_from].T               # shape: (from, to)
        return np.tanh(w).tolist()

    v = VIS
    weights_vis = [
        vis_weights(model.fc1.weight, v, v),    # (v, v)
        vis_weights(model.fc2.weight, v, v),    # (v, v)
        vis_weights(model.fc3.weight, v, 10),   # (v, 10)
    ]

    predictions = softmax(output.detach().cpu().numpy()[0])
    highest_n = int(np.argmax(predictions))

    batch_count += 1
    if int(target[0].item()) == highest_n:
        correct += 1

    return {
        "pixels": pixels,
        "weights": weights_vis,
        "predictions": predictions,
        "highest_n": highest_n,
        "target": int(target[0].item()),
        "batch_count": batch_count,
        "correct": correct,
        "accuracy": round((correct / batch_count) * 100, 2),
    }

# ---- HTML + p5.js frontend ----
HTML = r"""<!DOCTYPE html>
<html>
<head>
  <title>Neural Network Visualizer</title>
  <style>
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body { background: #000; overflow: hidden; }
    #status { position: fixed; bottom: 6px; right: 10px; color: #555;
              font-family: monospace; font-size: 12px; z-index: 10; }
  </style>
</head>
<body>
<div id="status">Connecting...</div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/p5.js/1.9.0/p5.min.js"></script>

<script>
const VIS = 20;
const INPUT_SHOW = 5;  // neurons shown at top and bottom of input layer
const LAYERS = [INPUT_SHOW * 2, 10, 10, 10];  // input: 5 top + 5 bottom
const W = 1200, H = 800;
const GRID_SIZE = 28;
// Match original: layer_spacing = WIDTH // (len(NEURONS_PER_LAYER) + 1) = 240
const LAYER_SPACING = Math.floor(W / (LAYERS.length + 1));

let state = null;
let mode = "stopped";
let ws;

function connectWS() {
  ws = new WebSocket("ws://localhost:8000/ws");
  ws.onopen = () => {
    document.getElementById("status").textContent = "Connected";
  };
  ws.onmessage = (e) => { state = JSON.parse(e.data); };
  ws.onclose = () => {
    document.getElementById("status").textContent = "Disconnected — reconnecting...";
    setTimeout(connectWS, 2000);
  };
}
connectWS();

new p5(function(p) {
  p.setup = function() {
    p.createCanvas(W, H).parent(document.body);
    p.frameRate(30);
    p.textFont("monospace");
  };

  let stepDelay = 0.1;

  p.keyPressed = function() {
    if (p.key === 't' || p.key === 'T') {
      mode = (mode === "train") ? "stopped" : "train";
      if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ action: "toggle" }));
      }
    } else if (p.key === '+' || p.key === '=') {
      if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ action: "speed", delta: 0.05 }));
      }
    } else if (p.key === '-') {
      if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ action: "speed", delta: -0.05 }));
      }
    }
  };

  // Input layer: 5 neurons in top 45%, gap with dots, 5 neurons in bottom 45%
  const INPUT_TOP_Y    = [0.06, 0.15, 0.24, 0.33, 0.42].map(f => f * H);
  const INPUT_BOTTOM_Y = [0.58, 0.67, 0.76, 0.85, 0.94].map(f => f * H);
  const INPUT_X = W / 8 + LAYER_SPACING;

  function neuronPos(layerIdx, neuronIdx) {
    if (layerIdx === 0) {
      const y = neuronIdx < INPUT_SHOW ? INPUT_TOP_Y[neuronIdx] : INPUT_BOTTOM_Y[neuronIdx - INPUT_SHOW];
      return { x: INPUT_X, y };
    }
    const numNeurons = LAYERS[layerIdx];
    return {
      x: W / 8 + LAYER_SPACING * (layerIdx + 1),
      y: (H / (numNeurons + 1)) * (neuronIdx + 1),
    };
  }

  p.draw = function() {
    p.background(0);

    // ---- 28x28 pixel grid (bottom-left, matching BOARD_POS = (0, HEIGHT*0.4)) ----
    const GRID_W = W / 4;
    const BOX_SPACING = Math.pow(GRID_SIZE, 0.2);
    const BOX_LENGTH = (GRID_W - GRID_SIZE * BOX_SPACING) / GRID_SIZE;
    const BOARD_Y = H * 0.4;

    if (state && state.pixels) {
      const px = state.pixels;
      const minV = Math.min(...px), maxV = Math.max(...px);
      for (let row = 0; row < GRID_SIZE; row++) {
        for (let col = 0; col < GRID_SIZE; col++) {
          const v = px[row * GRID_SIZE + col] || 0;
          const b = p.map(v, minV, maxV, 0, 255);
          p.noStroke();
          p.fill(p.constrain(b, 0, 255), 0, 0);
          p.rect(col * (BOX_LENGTH + BOX_SPACING),
                 BOARD_Y + row * (BOX_LENGTH + BOX_SPACING),
                 BOX_LENGTH, BOX_LENGTH);
        }
      }
    } else {
      // Empty placeholder grid
      for (let row = 0; row < GRID_SIZE; row++) {
        for (let col = 0; col < GRID_SIZE; col++) {
          p.noStroke(); p.fill(18);
          p.rect(col * (BOX_LENGTH + BOX_SPACING),
                 BOARD_Y + row * (BOX_LENGTH + BOX_SPACING),
                 BOX_LENGTH, BOX_LENGTH);
        }
      }
    }

    // ---- Connections ----
    p.strokeWeight(0.5);
    if (state && state.weights) {
      for (let li = 0; li < state.weights.length; li++) {
        const wMatrix = state.weights[li];
        for (let fi = 0; fi < wMatrix.length; fi++) {
          for (let ti = 0; ti < wMatrix[fi].length; ti++) {
            const w = wMatrix[fi][ti]; // tanh-normalized: (-1, 1)
            const alpha = p.map(Math.abs(w), 0, 1, 0, 150);
            const from = neuronPos(li, fi);
            const to   = neuronPos(li + 1, ti);
            p.stroke(w > 0 ? p.color(0, 200, 80, alpha) : p.color(220, 50, 50, alpha));
            p.line(from.x, from.y, to.x, to.y);
          }
        }
      }
    } else {
      // Dim placeholder connections before training starts
      p.stroke(50, 50, 50, 60);
      for (let li = 0; li < LAYERS.length - 1; li++) {
        for (let fi = 0; fi < LAYERS[li]; fi++) {
          for (let ti = 0; ti < LAYERS[li + 1]; ti++) {
            const from = neuronPos(li, fi);
            const to   = neuronPos(li + 1, ti);
            p.line(from.x, from.y, to.x, to.y);
          }
        }
      }
    }

    // ---- Neurons ----
    for (let li = 0; li < LAYERS.length; li++) {
      const isOutput = (li === LAYERS.length - 1);
      for (let ni = 0; ni < LAYERS[li]; ni++) {
        const pos = neuronPos(li, ni);
        if (isOutput && state && state.predictions) {
          const isTop = (ni === state.highest_n);
          const pred  = state.predictions[ni];
          p.fill(isTop ? p.color(255, 0, 0) : p.map(pred, 0, 1, 30, 200));
          p.stroke(isTop ? p.color(255, 80, 80) : p.color(120));
          p.strokeWeight(isTop ? 2 : 1);
          p.circle(pos.x, pos.y, 20);
          // prediction label to the right
          p.noStroke();
          p.fill(isTop ? p.color(255, 80, 80) : p.color(180));
          p.textSize(11); p.textAlign(p.LEFT, p.CENTER);
          p.text(pred.toFixed(2), pos.x + 13, pos.y);
        } else {
          p.fill(0, 180, 60);
          p.stroke(0, 255, 80);
          p.strokeWeight(1.5);
          p.circle(pos.x, pos.y, 20);
        }
      }
    }

    // Ellipsis dots for input layer
    p.noStroke(); p.fill(120);
    [0.46, 0.50, 0.54].forEach(f => p.circle(INPUT_X, f * H, 5));

    // Digit labels (0-9) to the left of output neurons
    p.noStroke(); p.textSize(11); p.textAlign(p.RIGHT, p.CENTER); p.fill(100);
    for (let ni = 0; ni < 10; ni++) {
      const pos = neuronPos(3, ni);
      p.text(ni, pos.x - 13, pos.y);
    }

    // Layer headers
    const layerNames = ["Input (784)", "Hidden 1 (10)", "Hidden 2 (10)", "Output"];
    p.textSize(11); p.fill(100); p.textAlign(p.CENTER, p.BASELINE);
    for (let li = 0; li < LAYERS.length; li++) {
      p.text(layerNames[li], W / 8 + LAYER_SPACING * (li + 1), H - 10);
    }

    // ---- HUD ----
    p.noStroke(); p.textAlign(p.LEFT, p.BASELINE); p.fill(255); p.textSize(16);
    p.text("Mode: " + mode.toUpperCase() + "  — Controls:  (t): pause/continue  |  (+/-): delay", 10, 20);
    if (state) {
      if (state.step_delay !== undefined) stepDelay = state.step_delay;
      p.textSize(14); p.fill(200);
      p.text("Target: " + state.target,          10, 45);
      p.text("Batch:  " + state.batch_count,      10, 65);
      p.text("Accuracy: " + state.accuracy.toFixed(2) + "%  (" + state.correct + "/" + state.batch_count + ")", 10, 85);
      p.text("Delay: " + stepDelay.toFixed(2) + "s / step", 10, 105);
      if (state.highest_n !== undefined) {
        p.fill(state.highest_n === state.target ? p.color(100, 255, 100) : p.color(255, 100, 100));
        p.text("Prediction: " + state.highest_n, 10, 125);
      }
    }
  };
});
</script>
</body>
</html>
"""

# ---- FastAPI server ----
app = FastAPI()

@app.get("/")
async def get():
    return HTMLResponse(HTML)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    global training_mode
    await websocket.accept()
    loop = asyncio.get_event_loop()

    async def receive_loop():
        global training_mode, STEP_DELAY
        try:
            async for msg in websocket.iter_text():
                data = json.loads(msg)
                if data.get("action") == "toggle":
                    training_mode = not training_mode
                elif data.get("action") == "speed":
                    STEP_DELAY = max(0.0, min(2.0, STEP_DELAY + data["delta"]))
        except (WebSocketDisconnect, Exception):
            pass

    async def send_loop():
        try:
            while True:
                if training_mode:
                    step_data = await loop.run_in_executor(None, train_step)
                    step_data["step_delay"] = STEP_DELAY
                    await websocket.send_json(step_data)
                    await asyncio.sleep(STEP_DELAY)
                else:
                    await asyncio.sleep(0.05)
        except (WebSocketDisconnect, Exception):
            pass

    await asyncio.gather(receive_loop(), send_loop())

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
