# DSA 8401 — Training Neural Networks on MNIST

A systematic study of neural network design, training, optimisation, regularisation, and image classification using the **MNIST handwritten digits dataset**.

This project was completed as part of **DSA 8401: Applied Machine Learning** in the **Master of Data Science and Analytics** programme.

The assignment focuses on understanding how neural networks behave under different architectural and training configurations before introducing a classical Convolutional Neural Network (CNN) as a benchmark.

---

## Project Overview

The objective of this project was to design, train, evaluate, and improve neural networks for handwritten digit classification.

Rather than starting with a pre-built CNN, the experiments began with **fully connected neural networks (FCNNs)** designed specifically for the MNIST problem. The models were progressively evaluated under different architectural, activation, initialisation, optimisation, and regularisation settings.

A classical CNN was introduced only after the fully connected models had been developed and evaluated, allowing the performance difference between spatially-aware convolutional models and fully connected models to be investigated.

The project therefore follows the progression:

```text
MNIST Data
    │
    ▼
Data Preparation
    │
    ▼
Fully Connected Neural Networks
    │
    ├── Architecture: Depth vs Width
    │
    ├── Activation Functions
    │
    ├── Gradient Behaviour
    │
    ├── Weight Initialisation
    │
    ├── Optimisers
    │
    ├── Learning Rate & Batch Size
    │
    └── Regularisation
    │
    ▼
Best Fully Connected Model
    │
    ▼
Final Test Evaluation
    │
    ├── Accuracy
    ├── Confusion Matrix
    └── Misclassified Digits
    │
    ▼
Classical CNN Benchmark
    │
    ▼
FCNN vs CNN Comparison
```

---

## Dataset

The project uses the **MNIST handwritten digits dataset**, consisting of:

* 70,000 grayscale images
* 60,000 training images
* 10,000 test images
* Image dimensions: **28 × 28 pixels**
* 10 digit classes: **0–9**

The dataset was loaded directly using:

```python
keras.datasets.mnist.load_data()
```

The training data was further divided into training and validation subsets, while the official 10,000-image test set was kept untouched until final evaluation.

### Data representation

Two representations of the images were maintained:

* **784-dimensional flattened vectors** for the fully connected neural networks.
* **28 × 28 × 1 image tensors** for the CNN benchmark.

Pixel values were scaled from the original `[0, 255]` range to `[0, 1]` to improve numerical stability and gradient-based optimisation.

---

## Experiments

The project systematically investigates the major factors affecting neural network training.

### 1. Network Architecture

Fully connected networks with different depths and widths were designed and compared.

The experiments investigate:

* Shallow versus deep networks
* Narrow versus wide networks
* Networks with 1–5 hidden layers
* Parameter counts
* Training time
* Validation accuracy
* Validation loss
* Evidence of overfitting

A parameter-matched comparison between a wider shallow network and a deeper narrower network was also performed to separate the effect of **architecture** from simply increasing the number of parameters.

---

### 2. Activation Functions

Different activation functions were evaluated while keeping the remaining training configuration as consistent as possible.

The experiments included:

* ReLU
* Sigmoid
* Tanh

The comparison focuses on:

* Convergence speed
* Training and validation behaviour
* Final validation performance
* Gradient flow
* The effect of activation functions in deeper networks

A deliberately deep sigmoid network was also investigated to demonstrate the practical effects associated with **vanishing gradients**.

---

### 3. Weight Initialisation

The project investigates the importance of how neural network weights are initialised.

The experiments include:

* He initialisation
* Glorot/Xavier initialisation
* Zero initialisation

The zero-initialisation experiment demonstrates why assigning identical initial weights to all neurons prevents effective symmetry breaking and leads to poor learning behaviour.

The results are compared in terms of:

* Convergence
* Training behaviour
* Validation performance
* Gradient behaviour

---

### 4. Optimisation

Multiple optimisation algorithms were compared using a controlled architecture.

The optimisers included:

* Adam
* RMSprop
* SGD with Momentum

The experiments examine how optimisation strategy affects:

* Training accuracy
* Validation accuracy
* Validation loss
* Convergence
* Training stability

The optimiser experiments showed that different optimisers can produce noticeably different validation behaviour even when the network architecture and other major training settings are held constant.

---

### 5. Learning Rate and Batch Size

Different learning rates and batch sizes were investigated to understand their effect on neural network training.

The experiments consider the relationship between:

```text
Learning Rate
      +
Batch Size
      ↓
Gradient Updates
      ↓
Convergence Behaviour
      ↓
Final Model Performance
```

The results demonstrate that training performance depends not only on the architecture but also on how the optimisation process is configured.

---

### 6. Regularisation

Regularisation techniques were investigated to evaluate their effect on generalisation and overfitting.

The experiments considered techniques such as:

* Dropout
* L2 regularisation
* Early stopping where appropriate

Training and validation curves were used to assess whether the models were beginning to overfit and whether regularisation improved generalisation.

---

### 7. Final Fully Connected Model

After comparing the different configurations, the strongest fully connected architecture was selected based on the experimental results.

The final model was evaluated on the previously untouched MNIST test set.

Evaluation included:

* Test accuracy
* Test loss
* Confusion matrix
* Misclassified examples
* Model saving
* Model restoration
* Predictions using the restored model

This stage represents the final performance of the project's own fully connected neural network before introducing convolution.

---

### 8. CNN Benchmark

Only after completing the fully connected experiments was a classical CNN introduced.

The CNN was trained using the original:

```text
28 × 28 × 1
```

image representation.

The comparison between the best FCNN and CNN considers:

| Measure         | Fully Connected Network |                           CNN |
| --------------- | ----------------------: | ----------------------------: |
| Architecture    |            Dense layers | Convolution + Pooling + Dense |
| Input           |  784-dimensional vector |             28 × 28 × 1 image |
| Parameter count | Reported experimentally |       Reported experimentally |
| Training time   | Reported experimentally |       Reported experimentally |
| Test accuracy   | Reported experimentally |       Reported experimentally |
| Error patterns  |                Analysed |                      Analysed |

The CNN comparison demonstrates why convolutional architectures are particularly suitable for image data.

---

## Key Concepts Investigated

The project provides practical experiments covering:

* Neural network architecture design
* Depth versus width
* Parameter count
* Activation functions
* Vanishing gradients
* Exploding gradients
* Weight initialisation
* Symmetry breaking
* Gradient-based optimisation
* Adam
* RMSprop
* SGD with Momentum
* Learning-rate selection
* Batch-size effects
* Dropout
* L2 regularisation
* Overfitting and generalisation
* Model evaluation
* Model persistence
* Convolutional neural networks
* Spatial information in images
* Weight sharing
* Local receptive fields
* Pooling
* Translation robustness

---

## Reproducibility

Random seeds were set during the experiments to improve reproducibility.

The project environment was based on Python and TensorFlow/Keras.

The notebook records the relevant library versions used during experimentation.

A `requirements.txt` file is included where applicable to document the Python dependencies required to reproduce the analysis.

---

## Repository Structure

```text
dsa8401-neural-network-mnist/
│
├── README.md
│
├── notebooks/
│   └── DSA8401_Training_Neural_Networks_MNIST.ipynb
│
├── models/
│   └── README.md
│
├── results/
│   ├── figures/
│   ├── tables/
│   └── model_comparison.csv
│
├── report/
│   └── assignment_report.pdf
│
├── requirements.txt
│
└── .gitignore
```

The exact contents may vary depending on which experimental outputs and trained model files are retained.

---

## Main Notebook

The main notebook contains the complete experimental workflow, including:

1. Environment and reproducibility setup
2. MNIST data loading
3. Exploratory visualisation
4. Data preprocessing
5. Train/validation/test splitting
6. Fully connected architecture experiments
7. Depth versus width comparison
8. Activation function experiments
9. Gradient analysis
10. Weight initialisation experiments
11. Optimiser comparison
12. Learning-rate experiments
13. Batch-size experiments
14. Regularisation experiments
15. Selection of the final FCNN
16. Final test evaluation
17. Model saving and restoration
18. CNN implementation
19. FCNN versus CNN comparison
20. Conclusions

---

## Results and Analysis

Each major experiment follows the same analytical structure:

> **What did I change?**
> **What happened?**
> **Why did it happen?**
> **What did I learn?**

This approach is used throughout the notebook rather than reporting model accuracy alone.

Results are supported using:

* Training and validation curves
* Accuracy comparisons
* Loss comparisons
* Parameter counts
* Training times
* Gradient measurements
* Confusion matrices
* Misclassified examples
* Model comparison tables

---

## Main Findings

The experiments demonstrate several important principles of neural network training:

1. **Architecture matters.**
   Increasing depth or width does not automatically guarantee better generalisation.

2. **Activation functions affect gradient flow.**
   ReLU generally provides more favourable gradient behaviour than sigmoid in deeper networks, while sigmoid networks can suffer from vanishing gradients.

3. **Initialisation is critical.**
   Appropriate initialisation helps maintain useful signal and gradient magnitudes during training, while zero initialisation prevents effective symmetry breaking.

4. **Optimisers behave differently.**
   Adam, RMSprop, and SGD with Momentum produced different convergence and validation behaviour despite being applied to the same general modelling problem.

5. **Hyperparameters matter.**
   Learning rate and batch size influence convergence speed, training stability, and final performance.

6. **Regularisation affects generalisation.**
   Dropout and other regularisation strategies can alter the balance between fitting the training data and generalising to unseen data.

7. **CNNs are better suited to image structure.**
   Flattening an image into 784 independent inputs removes explicit spatial relationships. Convolutional layers preserve and exploit local spatial structure through local receptive fields and shared weights.

8. **A strong fully connected network can still perform very well on MNIST.**
   The CNN benchmark therefore provides a useful demonstration of the additional benefit gained from architectures designed specifically for image data.

---

## Technologies

* Python
* TensorFlow
* Keras
* NumPy
* Pandas
* Matplotlib
* Seaborn
* Scikit-learn
* Jupyter Notebook

---

## Course

**DSA 8401 — Applied Machine Learning**

**Programme:** Master of Data Science and Analytics

**Project:** Training Neural Networks

**Dataset:** MNIST Handwritten Digits

---

## Author

**Simion Mike Mbumbu**

Master of Data Science and Analytics
Applied Statistics & Data Science

---

## License

This repository is primarily intended for academic and educational purposes as part of the DSA 8401 Applied Machine Learning coursework.
