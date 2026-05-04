# UCPO: Uniform Correct Policy Optimization

Official implementation of [Uniform Correct Policy Optimization](https://arxiv.org/abs/2605.00365).

UCPO is a reinforcement learning framework for training LLMs on mathematical reasoning. It introduces an inverse Q-learning (IQ) based advantage estimator that promotes **uniform coverage over correct responses**, encouraging the model to explore diverse solution paths rather than collapsing to a single high-reward mode.

Built on top of [verl](https://github.com/volcengine/verl) and [E3](https://github.com/LiaoMengqi/E3-RL4LLMs/tree/main/e3) .

---

## Method

Standard GRPO-style algorithms weight updates by reward, causing the policy to concentrate probability mass on a few high-reward responses. UCPO instead redistributes gradient such that spreads mass uniformly across all correct responses.

Key hyperparameters:
- `tau` — diversity scaling; controls the uniformity of the correct-response distribution

---

## Setup

```bash
# Install verl and dependencies
cd verl
pip install -e .
```

---

## Training

**DeepSeek-R1-Distill-Qwen-1.5B** (4 GPUs):
```bash
bash scripts_c/run_ucpo_1_5b.sh
```

**DeepSeek-R1-Distill-Qwen-7B** (8 GPUs):
```bash
bash scripts_c/run_ucpo_7b.sh
```

Key training settings:



Checkpoints are saved under `../checkpoint_ds/`.

---

## Evaluation

The evaluation pipeline has three stages:

**1. Inference** — generate rollouts from a checkpoint:
```bash
bash scripts_c/infer.sh
```

**2. Scoring** — compute per-response accuracy:
```bash
bash scripts_c/eval_all.sh
```

**3. Metrics** — aggregate Pass@K results:
```bash
bash scripts_c/metric_all.sh
```


---

## Dataset

Training uses a 10K math reasoning dataset (`dataset/train_data_10k.parquet`) with a held-out validation split (`dataset/valid_data.parquet`). Each example contains a prompt and a ground-truth answer in `\boxed{}` format.

---

## Citation

```bibtex
@article{ucpo2026,
  title={Uniform Correct Policy Optimization},
  author={Anamika Lochab, Bolian Li, Ruqi Zhang},
  journal={arXiv preprint arXiv:2605.00365},
  year={2026}
}
```
