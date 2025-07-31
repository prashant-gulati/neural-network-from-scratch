**venv and package setup**

```bash
python3 -m venv /Users/prashantgulati/Documents/dev/python/nnfs/.venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Colab version**

https://colab.research.google.com/drive/17XhZ4SCnEYh0EbYKBdzUbpiVUZ9hRmaW

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
