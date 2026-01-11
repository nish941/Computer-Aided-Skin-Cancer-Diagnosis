Computer-Aided Skin Cancer Diagnosis (ResNet + Web App) 

---
 
## 📌 About the Project

This project began as a part of our **Deep Learning elective coursework** in my 4th semester (Jan–May 2025). I’ve always been fascinated by the intersection of AI and healthcare, so I took up **automated skin cancer classification** — a topic I quickly realized was much more complex than it looked on the surface.

I initially started with standard **CNN models**, but the results were unstable, especially for rare classes like vascular lesions. After a lot of experimentation, I shifted to **ResNet-50** — a deeper model with residual blocks that made training much more stable and gave significantly better results, especially under image noise or poor contrast.

This was more than just a model training exercise — from data cleaning, preprocessing, class imbalance correction, to late-night debugging and waiting for all 50 epochs to finally run — it was a real deep dive into applied deep learning.

To complete the pipeline, I wrapped the model in a **Flask web app**, designed a simple frontend, and deployed it — so anyone could upload an image and get predictions along with Grad-CAM heatmaps to interpret the results.

---

## 🎯 What I Wanted to Achieve

- Train a CNN to classify **7 different types of skin lesions**
- Improve performance on underrepresented classes (e.g. AKIEC, DF)
- Implement **Grad-CAM** for visual interpretability
- Deploy a working **diagnostic demo** via Flask

---

## ✅ What I Actually Achieved

- 🧠 Trained and fine-tuned **ResNet-50** with class-weighted loss and preprocessing
- ⚙️ Achieved **84.3% accuracy** and **91% sensitivity** on test set
- 📈 Compared with **MobileNetV2** and ensemble versions
- 🔬 Used **Grad-CAM** to explain model predictions
- 🌐 Deployed a fully working **Flask web app** with image upload and live results
- 🧪 Experimented with dropout tuning, early stopping, and noise robustness tests

---

## 🔍 Technical Deep Dive

- **Model Choice**: Started with a vanilla CNN but it overfit quickly. ResNet-50 offered skip connections that helped avoid vanishing gradients and trained better on medical images.
- **Preprocessing**:
  - All images converted to **grayscale**
  - Histogram equalization to improve lesion visibility
  - Resized to **224x224**
- **Training Details**:
  - Framework: `TensorFlow + Keras`
  - Optimizer: `Adam`, LR = 1e-4 with ReduceLROnPlateau
  - Loss: `CategoricalCrossentropy` with class reweighting
  - Regularization: Dropout (0.3), EarlyStopping
- **Data Augmentation**:
  - Horizontal/vertical flips
  - Random zoom & brightness
- **Validation Strategy**:
  - 80/10/10 split (train/val/test)
  - Used stratified sampling to preserve class ratios

---

## ⚙️ Tech Stack

| Category         | Tools Used                                      |
|------------------|-------------------------------------------------|
| 💻 Languages      | Python, HTML, CSS, JavaScript                   |
| 🔍 Deep Learning  | TensorFlow, Keras, tf-keras-vis                 |
| 🧰 Tools/Libraries| NumPy, Matplotlib, OpenCV, Flask                |
| 📊 Visualization  | Grad-CAM for interpretability                   |
| 🖥 Deployment     | Flask + GitHub Pages (for static frontend)      |

---

## 📁 File Structure

```bash
📦skin-cancer-diagnosis
├── app/
│   ├── app.py              # Flask app main server
│   ├── model_loader.py     # Load model and Grad-CAM logic
│   ├── utils.py            # Image preprocessing
│   ├── templates/
│   │   └── index.html      # Frontend UI
│   └── static/
│       └── style.css       # CSS styles
├── models/
│   └── resnet50_model.h5   # Trained model weights
├── notebooks/
│   └── training_resnet.ipynb  # Full training notebook
├── requirements.txt
├── README.md
└── run.sh
````

---

## 🧪 Dataset Info

* **Source**: [ISIC 2018 Challenge Dataset](https://challenge.isic-archive.com/data/)
* **Classes** (7 total):

  * Melanoma (MEL)
  * Nevus (NV)
  * Basal Cell Carcinoma (BCC)
  * Actinic Keratoses (AKIEC)
  * Benign Keratosis (BKL)
  * Dermatofibroma (DF)
  * Vascular lesions (VASC)
* **Challenge**: Very imbalanced dataset — classes like VASC and DF had <5% representation.
* **Solution**: Applied **class-weighting** in the loss function and oversampling in training batches.

---

## 📈 Results Summary

| Model          | Accuracy | Sensitivity | Recall (Rare) | Test Robustness |
| -------------- | -------- | ----------- | ------------- | --------------- |
| ResNet-50      | 84.3%    | 91.0%       | 83.2%         | ✅ High          |
| MobileNetV2    | 75.1%    | 81.2%       | 88.4%         | ⚠️ Unstable     |
| Ensemble (Mob) | 79.5%    | 85.0%       | 91.0%         | ⚠️ Mixed        |

---

## 🧠 Grad-CAM Visualizations

* Implemented Grad-CAM to visualize which parts of the lesion image influenced the model’s prediction.
* Made it **part of the live web app**, not just the notebook!
* Used `tf-keras-vis` to extract final conv layer activations and overlaid on input image.

---

## 🌐 Try It Live

> 💡 Try the live frontend:
> [🔗 Skin Cancer Diagnosis Web UI](https://yourusername.github.io/skin-cancer-diagnosis)

*(Frontend only — model hosted locally for now. Full deployment coming soon!)*

---

## 🚀 How to Run Locally

```bash
git clone https://github.com/yourusername/skin-cancer-diagnosis.git
cd skin-cancer-diagnosis
pip install -r requirements.txt
python app/app.py
```

Open `http://localhost:5000` in your browser to use the diagnostic app.

---

## 🧰 Deployment Notes

To deploy:

1. Host `app/` on **Render** or **Railway** (recommended)
2. Use `run.sh` as startup script
3. Static UI deployed via GitHub Pages:

   * `templates/index.html` and `static/style.css` in `gh-pages` branch

---

## 🔮 Future Work

* Convert into **Streamlit / React-based frontend** for faster loading
* Add **patient database and login feature**
* Dockerize the entire project for deployment on any platform
* Add clinical validation with real feedback from medical professionals

---

## 🙏 Acknowledgements

* Thanks to [ISIC Archive](https://www.isic-archive.com/) for the open-access dataset
* TensorFlow and Keras documentation for making model tuning bearable
* `tf-keras-vis` for easy Grad-CAM implementation
* Everyone in our Deep Learning class who helped debug weird tensor shapes at 2am 🙃

---
