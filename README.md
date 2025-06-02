# Face Recognition

This project implements a complete face recognition pipeline using both linear and non-linear dimensionality reduction techniques, followed by unsupervised clustering. It includes custom implementations of **PCA**, **Autoencoder**, **K-Means**, and evaluation using **Gaussian Mixture Models (GMM)**, aiming to cluster face images into subject identities without supervision.

---

## Dataset

- **Name:** ORL (AT&T) Face Database
- **Source:** [Kaggle](https://www.kaggle.com/kasikrit/attdatabase-of-faces/)
- **Details:** 40 subjects, 10 grayscale images per subject (92x112 pixels)

---

## Tools & Libraries

- Python
- NumPy, Pandas, Matplotlib
- Scikit-learn
- TensorFlow / Keras
- SciPy (Hungarian algorithm for clustering accuracy)

---

## Project Structure

```bash

├── 📁 data/                      # Contains the ORL face dataset (s1/, s2/, ..., s40/)
│   ├── s1/
│   ├── s2/
│   └── ...
├── face_recognition.ipynb
├── README.md # You are here
```

---

## Pipeline Overview

1. **Preprocessing**  
   - Load grayscale face images and flatten into 10,304-dimensional vectors  
   - Normalize or standardize data  

2. **Dimensionality Reduction**
   - **PCA**: Reduce to retain {80%, 85%, 90%, 95%} variance
   - **Autoencoder**: Deep model compressing input to 128 latent dimensions

3. **Clustering**
   - **K-Means**: Custom implementation and sklearn version
   - **GMM**: Applied on both PCA and Autoencoder-reduced features

4. **Evaluation**
   - Clustering accuracy using Hungarian algorithm
   - Optional F1-score and confusion matrix

---


> *Note on PCA Eigenvectors File*
>
> In the face_recognition.ipynb file, the line that computes the eigenvalues and eigenvectors directly from the covariance matrix is currently commented out, and instead, the code uses np.load(...) to load them from a precomputed file.
> However, the precomputed .npy file containing the full set of eigenvectors is not uploaded to this repository because its size exceeds GitHub's 100MB file limit.
> If you wish to run the PCA code from scratch, simply uncomment the eigen decomposition line and compute the eigenvalues/vectors directly. Be aware this may take time and memory depending on your system.
---

## How to Run

```bash
# Clone the repo
git clone https://github.com/meissa20/face-recognition.git
# Install dependencies
pip install -r requirements.txt

# Run the main notebook
jupyter notebook face_recognition.ipynb

