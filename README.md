# FlowCF-Dockerized: Flow Matching for Collaborative Filtering (Optimized)

> ** Note:** This is an **unofficial, modified implementation** of the paper *"Flow Matching for Collaborative Filtering (KDD 2025)"*.
> This repository was created to improve reproducibility, fix Linux/Docker compatibility issues, and optimize training performance.

---

## Original Paper & Source
All credits for the model architecture and core algorithms belong to the original authors.

* **Paper Title:** Flow Matching for Collaborative Filtering (KDD 2025)
* **Authors:** Chengkai Liu, Yangtian Zhang, Jianling Wang, Rex Ying, James Caverlee
* **Original Paper:** [https://arxiv.org/abs/2502.07303](https://arxiv.org/abs/2502.07303)
* **Original Repository:** [Insert Link to Original GitHub Repo Here]

---

## 🛠 Modifications & Improvements

This project includes the following engineering improvements over the original implementation:

### 1. Full Dockerization
* Added `Dockerfile` and `docker-compose.yml` for a consistent, reproducible environment.
* Solved dependency conflicts between **RecBole** and recent **Pandas** versions.
* Fixed pathing issues (`utils.py`, `inference.py`) to ensure smooth execution on Linux servers.

### 2. Code Hotfixes
* **RecBole Compatibility:** Patched `AttributeError` related to `DataFrame.shuffle` and `Series.numpy` methods in the library.
* **Config Loading:** Fixed an issue where custom parameters (e.g., `s_steps`) in `.yaml` files were being ignored or overridden by defaults.

### 3. Performance Tuning
* **High-Performance Configuration:** Optimized `flowcf.yaml` for high-end GPUs (Increased Batch Size, Tuned MLP dimensions).
* **Cold-Start Simulation:** Added custom scripts (`inference.py`, `evaluate.py`) to simulate and evaluate **Item Cold-Start** scenarios (predicting potential users for new items).

---

## Quick Start

### 1. Prerequisites
* Docker & Docker Compose
* NVIDIA Drivers (for GPU support)

### 2. Build the Environment
```bash
docker compose build
```

### 3. Train the Model
This will start training using the optimized configuration in flowcf.yaml.

```bash
docker compose up
```
The trained model checkpoint will be saved in the saved/ directory (e.g., saved/FlowCF-Jan-12-xxxx.pth).


### 4. Inference (Simulation) & Evaluate Performance
Simulate a "New Movie Release" scenario using specific seed users.
Test the model's accuracy on Item Cold-Start scenarios (Recall@K, NDCG@K).
```bash
# Replace the filename with your actual saved model
docker compose run --rm flowcf python inference.py --checkpoint saved/YOUR_MODEL_FILE.pth
```

## Configuration (flowcf.yaml)
Key hyperparameters have been tuned for better convergence and speed:

- train_batch_size: Increased to 1024 (Balanced for detail & speed).
- eval_batch_size: set to 40960 for rapid evaluation.
- s_steps: Set to 10 for more precise ODE solver inference.
- dims_mlp: Tuned to [1000, 1000, 1000] for optimal capacity.
