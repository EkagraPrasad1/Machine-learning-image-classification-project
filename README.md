# Machine-learning-image-classification-project

CIFAR-10 Image Classification — compact CNNs that generalize on CPU

Wrapped up a focused ML project for COMP3032 where I compared a classic baseline to small CNNs under CPU constraints, with honest validation and a strictly held-out test set.

What I built

Reproducible pipeline: stratified train/val/test, fixed seeds, saved indices & best-epoch checkpoints

Baseline: multinomial Logistic Regression on flattened pixels

CNNs: 2-block and 3-block (Conv-BN-ReLU-Pool + GAP) with early stopping

Results (Test Accuracy)

Logistic Regression: 0.281

2-block CNN: 0.361

3-block + flip/crop aug + SGD(m=0.9): 0.529 ✅ (+24.8 pp over LR)

Why it matters

CNNs preserve spatial structure, improving class separability (especially vehicles vs animals)

Depth helped more than width in this short schedule

The training recipe (augmentation + momentum) delivered the biggest generalization lift

Stack & skills
Python, PyTorch/torchvision, scikit-learn, NumPy, Matplotlib • validation design • early stopping • confusion-matrix & per-class P/R/F1 analysis • experiment tracking

Next steps
Longer schedules, cosine LR, stronger aug (Cutout/Mixup), lightweight residuals.

#machinelearning #deeplearning #computervision #pytorch #datascience #cifar10 #studentprojects
