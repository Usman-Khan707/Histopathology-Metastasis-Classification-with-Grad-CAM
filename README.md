# Metastasis Detection in Lymph Node Histopathology via CNN Classification and Grad-CAM Interpretability

Deployed demo — [🔗 Live app](https://histopathology-image-classification-ml.streamlit.app)

*"That which is measured, improves." — Karl Pearson*

## Abstract

Metastasis in regional lymph nodes is a key determinant of breast cancer staging, and its manual detection from whole-slide histopathology images is labor-intensive and subject to inter-observer variability. This project implements and evaluates a convolutional neural network (CNN) for binary classification of lymph node histopathology patches as **normal** or **metastatic (tumor)** tissue, using the **PatchCamelyon (PCam)** benchmark derived from the CAMELYON16 challenge dataset. A ResNet50 architecture is trained end-to-end (no ImageNet pretraining) on a stratified subset of PCam and evaluated using standard classification metrics. To address the interpretability requirements of clinical decision-support tools, we integrate **Gradient-weighted Class Activation Mapping (Grad-CAM)** to visualize the spatial regions driving each prediction. The trained model is packaged into an interactive Streamlit application supporting batch inference and heatmap overlay. We report the model's quantitative performance, characterize its class-wise error behavior, and discuss the specific factors — reduced-scale training, limited epochs, and no pretrained initialization — that bound current performance and motivate directions for future work.

---

## 1. Introduction

Histopathological examination of lymph node sections is the clinical standard for detecting metastatic spread in breast cancer, but manual review of gigapixel whole-slide images is time-consuming and shows measurable variability between pathologists. Deep learning approaches to digital pathology have demonstrated the potential to assist this workflow by flagging regions of concern for expert review. This project studies that setting at a tractable scale: rather than gigapixel whole-slide images, it operates on the pre-extracted patch-level formulation provided by **PatchCamelyon**, in which the diagnostic question reduces to a binary decision — *does this 96×96 px patch contain metastatic tissue in its center region?*

The project has two intertwined goals:

1. **Predictive performance** — train a CNN to classify PCam patches as tumor vs. normal, and characterize its accuracy, precision, recall, and F1-score.
2. **Interpretability** — since a "black box" classifier has limited value in a diagnostic-support context, use Grad-CAM to visualize *which regions of a patch* the network attends to when making a prediction, and inspect whether these regions are pathologically plausible.

---

## 2. Dataset: PatchCamelyon (PCam)

PCam is a benchmark derived from the [CAMELYON16](https://camelyon16.grand-challenge.org/) challenge dataset of H&E-stained lymph node whole-slide images. It reformulates whole-slide metastasis detection as a tractable patch-level binary classification problem — described by its authors as **"bigger than CIFAR-10, smaller than ImageNet, and trainable on a single GPU."**

| Property | Value |
|---|---|
| Total images | 327,680 color patches |
| Patch size | 96 × 96 px, RGB |
| Label rule | Positive (1) iff the **center 32×32 px region** contains ≥1 pixel of tumor tissue; tumor tissue in the outer region does not affect the label |
| Official split — Train | 262,144 patches (131,072 / 131,072 — perfectly class-balanced) |
| Official split — Validation | 32,768 patches (16,399 normal / 16,369 tumor) |
| Official split — Test | 32,768 patches (16,391 normal / 16,377 tumor) |

Green boxes in the figure below mark tumor tissue in the center region, which determines the positive label:

![pcam](https://github.com/user-attachments/assets/b8ef1762-de38-4747-a648-914b075b1b35)

**Dataset repository:** [github.com/basveeling/pcam](https://github.com/basveeling/pcam)

> Because the center-region labeling rule decouples the label from the *entire* visual content of the patch, a portion of the classification difficulty is attributable to the dataset design itself, not only to model capacity — informative context can sit just outside the labeled region.

**Scale used in this study.** Loading the full 262,144-patch training set at float32 precision exceeds the memory of a typical single-machine setup. To keep experimentation tractable, this project trains and validates on a **stratified subset**: the first 20,000 training patches and 5,000 validation patches from the official HDF5 arrays. Section 8 discusses how this reduction, rather than the model architecture per se, is the primary limiting factor on the reported results.

---

## 3. Model Architecture

The classifier (`model.py`) wraps a **ResNet50** backbone with a lightweight classification head:

```
Input (96×96×3)
   → ResNet50 backbone (weights = None — trained from scratch, NOT ImageNet-pretrained)
   → GlobalAveragePooling2D
   → Dense(128, activation="relu")
   → Dense(1, activation="sigmoid")
```

| Property | Value |
|---|---|
| Total parameters | 23,850,113 |
| Trainable parameters | 23,850,113 (100% — no layers frozen) |
| Optimizer | Adam |
| Loss | Binary cross-entropy |
| Output | Sigmoid probability of "tumor" class |

**A note on transfer learning.** The training notebook (`train_pcam.ipynb`) explores an alternative configuration with `weights="imagenet"` and the backbone frozen (`base_model.trainable = False`), which reduces the trainable parameter count to 262,401. This configuration is instructive but was **not** the one used to produce the reported results: the actual training run rebuilds the model via `build_model()` from `model.py`, which initializes ResNet50 with `weights=None` and leaves every layer trainable. In other words, the reported model learns all of its convolutional features directly from PCam rather than reusing ImageNet-derived features. This is a meaningful methodological detail — it means the current results likely *understate* what the same architecture could achieve with pretrained initialization, since the frozen-backbone configuration was set up in the notebook but not the one that was trained end-to-end (see Section 8, Future Work).

---

## 4. Training Setup

| Hyperparameter | Value |
|---|---|
| Training samples | 20,000 (stratified subset of official train split) |
| Validation samples | 5,000 (stratified subset of official validation split) |
| Epochs | 3 |
| Batch size | 64 |
| Input normalization | Pixel values scaled to [0, 1] |

**Training and validation curves (as logged):**

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc |
|---|---|---|---|---|
| 1 | 0.4934 | 0.7913 | 1.5277 | 0.6554 |
| 2 | 0.3933 | 0.8277 | 0.5589 | 0.7220 |
| 3 | 0.3658 | 0.8447 | 0.6838 | 0.7088 |

Training accuracy increases monotonically (79.1% → 84.4%) and training loss decreases steadily, indicating the network is fitting the training distribution. Validation accuracy improves from epoch 1 to 2 but plateaus (and loss fluctuates) through epoch 3 — a signature consistent with the onset of mild overfitting under limited data and short training schedules, rather than a fundamental architectural failure.

---

## 5. Evaluation

Model performance was assessed on the 5,000-sample validation subset using `sklearn.metrics.classification_report`:

| Metric | Class 0 (Normal) | Class 1 (Tumor) | Macro Avg | Weighted Avg |
|---|---|---|---|---|
| Precision | 0.65 | 0.82 | 0.74 | 0.74 |
| Recall | 0.88 | 0.54 | 0.71 | 0.71 |
| F1-score | 0.75 | 0.65 | 0.70 | 0.70 |
| **Overall Accuracy** | | | **0.71** | |

**Interpretation.**

- The model exhibits a pronounced **asymmetry between classes**: it is highly sensitive to normal tissue (recall = 0.88) but substantially less sensitive to tumor tissue (recall = 0.54). Equivalently, when the model does flag a patch as tumor, it is usually correct (precision = 0.82), but it **misses roughly 46% of true tumor patches**.
- This behavior pattern — high specificity, moderate-to-low sensitivity for the positive (disease) class — is the opposite of what is typically desired in a diagnostic-support setting, where **false negatives (missed metastases) are the more clinically costly error type** relative to false positives (which a pathologist can rule out on review).
- The gap between training accuracy (84.4%) and validation accuracy (~71%) at the final epoch, combined with the fluctuating validation loss in Section 4, supports the interpretation that the model is data- and schedule-limited rather than architecture-limited: a larger and more varied training sample, together with regularization and a longer schedule, would be expected to narrow this gap and specifically improve tumor recall.

---

## 6. Interpretability via Grad-CAM

To make the classifier's decisions inspectable, this project implements **Grad-CAM** (`gradcam.py`) targeting the final convolutional block of the backbone (`conv5_block3_out`). For a given input patch, Grad-CAM computes the gradient of the predicted class score with respect to the feature maps of this layer, weights each channel by its gradient-averaged importance, and produces a coarse localization map over the input.

Reading the heatmaps:
- 🔴 **Red / hot regions** — areas the network relied on most heavily for its prediction (e.g., regions of dense or irregular nuclei in tumor patches).
- 🔵 **Blue / cold regions** — areas that contributed little to the decision.

Example grid of original patches with their Grad-CAM overlays (validation samples, ground-truth and predicted class shown per patch):

<img width="1288" height="357" alt="histology_white_bg" src="https://github.com/user-attachments/assets/2083a373-f5fe-48c4-ad24-4828df2303f8" />

**Observations from qualitative inspection:**
- On confidently and correctly classified patches, activation tends to concentrate on regions of visibly abnormal cell morphology, which is consistent with what a pathologist would attend to.
- On misclassified patches — particularly false negatives — activation is often diffuse or weak, suggesting the network fails to localize a clear discriminative region rather than confidently attending to the wrong one. This is a useful diagnostic signal: it points toward *representation* limitations (the network hasn't learned a strong enough tumor-morphology feature) rather than a systematic attention bug.

Grad-CAM is implemented twice in this repository with minor variations: once as a standalone module (`gradcam.py`) used in the training notebook, and once inline within `app.py` for the deployed application (using a mathematically equivalent matrix-multiplication formulation of the weighted channel sum).

---

## 7. Interactive Application

The trained model is served through a Streamlit application (`app.py`, deployed at the link above) that allows a user to:

- Upload one or more tissue patch images (PNG/JPG) for batch inference.
- Toggle a Grad-CAM overlay and adjust its opacity via a slider.
- View, per image, the predicted class (Malignant/Benign), confidence score, and inference latency.
- Inspect batch-level results in a summary table.
- Read short in-app explanations of Grad-CAM and the PCam dataset, aimed at non-specialist users.

The app explicitly labels itself **"Research Use Only"** in the sidebar, reflecting that this system is a technical demonstration and not a validated clinical tool.

---

## 8. Discussion: Why Performance Is Moderate, and What Would Change It

The results above (71% accuracy, 0.54 tumor recall) should be read in the context of the deliberate scope reductions made for this project, rather than as a ceiling on what the architecture can achieve:

1. **Data scale.** Training used ~7.6% of the official PCam training set (20,000 of 262,144 patches). PCam was explicitly designed as a dataset large enough to benefit from full-scale training; sub-sampling it removes much of the pattern diversity the benchmark was built to provide.
2. **Training schedule.** Three epochs is sufficient to observe a clear learning trend but is well short of convergence for a 23.85M-parameter network trained from random initialization.
3. **No pretrained initialization.** As discussed in Section 3, the trained model does not benefit from ImageNet-pretrained weights, despite the notebook containing a scaffold for this configuration. Transfer learning from natural images has repeatedly been shown to accelerate convergence and improve data efficiency even for out-of-domain targets like histopathology.
4. **No explicit class-imbalance handling or augmentation** was applied in the training loop, despite the observed recall asymmetry — geometric augmentation (rotation, flipping) is a natural fit for histopathology, since tissue morphology is orientation-invariant.

### Future Work

- **Use the full dataset with batched/streamed loading** (e.g., `tf.data` pipelines or Keras `HDF5Matrix`/generators) to avoid materializing all patches in memory at once.
- **Enable pretrained initialization** (`weights="imagenet"`) with a staged fine-tuning schedule (frozen backbone warmup → gradual unfreezing), which the notebook already scaffolds but does not carry through to the trained model.
- **Train for more epochs with early stopping and learning-rate scheduling**, monitoring validation tumor-recall specifically rather than overall accuracy.
- **Add data augmentation** (rotation, flipping, color jitter) appropriate to the rotation-invariance properties of histopathology, as explored in the original PCam paper.
- **Address class-wise recall asymmetry directly** — e.g., class-weighted loss, threshold tuning on the ROC curve, or focal loss — given that false negatives are the more clinically significant error mode.
- **Benchmark alternative backbones** (EfficientNet, DenseNet) and rotation-equivariant architectures, which the original PCam paper introduces as particularly well-suited to this task's symmetries.

---

## 9. Installation

```bash
# Clone the repository
git clone https://github.com/BleeGleeWee/Histopathology-Image-Classification.git
cd Histopathology-Image-Classification

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

---

## 10. Usage

1. Download the PCam `.h5` files from [github.com/basveeling/pcam](https://github.com/basveeling/pcam) (see `data/dataset.md` for the specific Google Drive link and required files — only the six `.h5` files, ~26 GB uncompressed, are needed, not the `.gz` archives).
2. Place them inside the `data/` folder, matching the naming convention `camelyonpatch_level_2_split_{train,valid,test}_{x,y}.h5`.
3. Open `train_pcam.ipynb` and run it top-to-bottom to:
   - Load and normalize the data
   - Build and train the CNN (`model.py`)
   - Evaluate on the held-out subset
   - Generate Grad-CAM visualizations (`gradcam.py`)
4. To run the interactive demo locally: `streamlit run app.py` (requires `histopath_model.h5` — see the note below).

### A note on `.h5` files and memory

Loading the full-resolution image arrays (`*_x.h5`) into memory at once can raise a `MemoryError` on machines with limited RAM, since the full training array alone is several gigabytes. Prefer `h5py` in streaming mode or Keras' `HDF5Matrix`/generator-based loading for the full dataset; the notebook's slicing pattern (`f["x"][:20000]`) is a simple way to work with a bounded subset during development.

### A note on the packaged model weights

This repository tracks `histopath_model.h5` via **Git LFS** (see `.gitattributes`). If you obtain the repository as a plain `.zip` download (rather than via `git clone` with LFS support), `histopath_model.h5` will be a small Git LFS *pointer file* rather than the actual ~287 MB weight file, and `app.py` will fail to load it. Clone with Git LFS enabled, or retrain using `train_pcam.ipynb` and save your own weights, to run the app locally.

---

## 11. Repository Structure

```
Histopathology-Image-Classification/
│
├── data/
│   ├── camelyonpatch_level_2_split_train_y.h5   # official train labels (image arrays not included — see dataset.md)
│   ├── camelyonpatch_level_2_split_valid_y.h5   # official validation labels
│   ├── camelyonpatch_level_2_split_test_y.h5    # official test labels
│   └── dataset.md                               # dataset download instructions and citations
│
├── train_pcam.ipynb        # end-to-end notebook: load data → build model → train → evaluate → Grad-CAM
├── model.py                # build_model(): ResNet50 backbone + classification head
├── data_loader.py          # HDF5 loading and normalization utilities
├── gradcam.py               # Grad-CAM heatmap generation and overlay utilities
├── app.py                   # Streamlit inference application with Grad-CAM overlay
├── histopath_model.h5       # trained model weights (Git LFS — see note in Section 10)
│
├── requirements.txt
├── LICENSE                  # MIT
├── .gitattributes            # Git LFS config for .h5 files
├── .gitignore
└── README.md
```

*Note: this structure reflects the actual, flat layout of the repository at the time of writing. Image assets referenced in this README are hosted via GitHub's user-attachments CDN rather than stored in a local `images/` directory.*

---

## 12. Contributing

This repository is maintained for **educational and research-demonstration purposes**. Contributions are welcome, particularly:

- Full-dataset training runs and updated benchmark numbers
- Pretrained-backbone / fine-tuning experiments
- Additional interpretability methods (e.g., Integrated Gradients, SHAP) for comparison against Grad-CAM
- Improved data pipelines for memory-constrained environments

Please open an issue or pull request with proposed changes.

---

## 13. References

1. Veeling, B. S., Linmans, J., Winkens, J., Cohen, T., & Welling, M. (2018). *Rotation Equivariant CNNs for Digital Pathology*. arXiv:1806.03962. [https://arxiv.org/abs/1806.03962](https://arxiv.org/abs/1806.03962)

2. Ehteshami Bejnordi, B., et al. (2017). *Diagnostic Assessment of Deep Learning Algorithms for Detection of Lymph Node Metastases in Women With Breast Cancer*. JAMA, 318(22), 2199–2210. doi:10.1001/jama.2017.14585

3. Selvaraju, R. R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., & Batra, D. (2017). *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization*. Proceedings of the IEEE International Conference on Computer Vision (ICCV), 618–626.

4. He, K., Zhang, X., Ren, S., & Sun, J. (2016). *Deep Residual Learning for Image Recognition*. Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 770–778.

---

## License

Released under the [MIT License](LICENSE).
