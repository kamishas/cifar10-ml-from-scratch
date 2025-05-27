# Deeplearning


---
## 🧠 CIFAR-10: Building Machine Learning Models From Scratch (No Libraries!)

Hi there! I’m Shasank — and this repo is a personal deep dive into understanding how core machine learning algorithms work *without relying on frameworks like TensorFlow or PyTorch*.

The idea was simple: **take the CIFAR-10 image dataset and try to classify images using different ML models written completely from scratch using just Python + NumPy**. No black boxes — I wanted to learn what happens under the hood, line by line.

---

### 🧪 What I Built

Over several iterations, I implemented and compared the following classifiers:

| Model          | Loss Function   | Key Takeaways                                                 | Test Accuracy |
| -------------- | --------------- | ------------------------------------------------------------- | ------------- |
| **Perceptron** | Perceptron Loss | Linear model, struggled with non-linear patterns in images    | \~48.91%      |
| **SVM**        | Hinge Loss      | Margin-based optimization, requires regularization tuning     | \~35.92%      |
| **Softmax**    | Cross-Entropy   | Probabilistic output, better for multi-class classification   | \~39.71%      |
| **MLP**        | Cross-Entropy   | Added hidden layers + ReLU → significant boost in performance | Best overall  |

---

### 🔬 Technical Highlights

#### 📦 Dataset

* CIFAR-10: 60,000 color images (32x32), 10 classes (airplanes, cats, cars, etc.)
* Preprocessed the data manually into batches for training.

#### 🔧 Hyperparameter Tuning

Each model had custom-tuned:

* Learning rate
* Epochs
* Regularization constant (for SVM & Softmax)
* Batch size (64 worked best)
* Tried adding **learning rate decay** for stability (did help a bit).

#### 📊 Accuracy Metrics (rounded)

| Metric            | Perceptron | SVM    | Softmax | MLP     |
| ----------------- | ---------- | ------ | ------- | ------- |
| Training Accuracy | \~52.18%   | 35.28% | 40.06%  | Higher  |
| Validation        | \~48.41%   | 34.84% | 39.14%  | Higher  |
| Test Accuracy     | \~48.91%   | 35.92% | 39.71%  | Highest |

---

### 🤖 What I Learned

* **Perceptrons** are great starting points, but fall short with complex datasets like CIFAR-10.
* **SVMs** benefit a lot from hyperparameter tuning, but require careful balancing to avoid underfitting.
* **Softmax + Cross-Entropy** is a great fit for multi-class tasks, and gives smoother gradients during backprop.
* **MLPs (Multi-Layer Perceptrons)** unlock the real power — stacking layers and using ReLU activations helped my model actually *understand* the image patterns.

This project gave me a *hands-on* understanding of:

* Loss function behaviors
* Gradient computation and optimization
* Overfitting vs underfitting trade-offs
* The importance of model architecture and feature representations


---

### 🔮 What’s Next?

I want to:

* Add CNNs (convolutional neural networks) to boost image understanding.
* Implement dropout and batch normalization.
* Switch to GPU-based acceleration for faster training.

---

### 👋 Final Thoughts



Feel free to check it out, leave feedback, or fork it and try tweaking things yourself!

---


