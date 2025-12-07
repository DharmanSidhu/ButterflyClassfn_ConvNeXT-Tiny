# 🦋 Butterfly Species Classification using ConvNeXt

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange?logo=tensorflow)
![Status](https://img.shields.io/badge/Status-Completed-success)
![Accuracy](https://img.shields.io/badge/Accuracy-96%25-brightgreen)

A Fine-Grained Visual Classification (FGVC) project capable of identifying **100 distinct butterfly species** with state-of-the-art accuracy. This project utilizes **ConvNeXtTiny**, a modernized CNN architecture that integrates Vision Transformer (ViT) design principles to outperform traditional models like ResNet50 and VGG19.

## 📂 Repository Contents

| File | Description |
| :--- | :--- |
| `flutterby.ipynb` | **The Core Code.** End-to-end Jupyter Notebook covering data loading, training, and evaluation. |
| `butterfly_report_convNeXT.pdf` | Full research report detailing methodology, literature survey, and experimental analysis. |
| `butterfly_report_convNeXT.docx` | Editable version of the research report. |
| `flow.drawio` | [Editable Workflow Diagram](https://drive.google.com/file/d/17nj8DO-HxERcUy4TLfgCE5bALLhNpPCb/view) |
| `README.md` | Project documentation (this file). |

---

## 🚀 Key Features

* **Model:** ConvNeXtTiny (~28M Parameters) - Pre-trained on ImageNet.
* **Dataset:** 13,594 images covering 100 species (Train/Val/Test split).
* **Technique:** Two-Phase Transfer Learning (Feature Extraction $\rightarrow$ Fine-Tuning).
* **Optimization:** Implements a memory-safe `tf.data` pipeline using disk streaming (prefetch) to prevent RAM overflows on standard GPUs (e.g., Colab T4).
* **Performance:** Achieves **~96% Test Accuracy**, surpassing traditional baselines.

---

## 📊 Results & Comparison

We compared our ConvNeXt implementation against published benchmarks on the same dataset.

| Model Architecture | Parameters | Published Accuracy | Our Result |
| :--- | :--- | :--- | :--- |
| **VGG19** | 143M | 92.80% | -- |
| **ResNet50** | 25M | 95.00% | -- |
| **ConvNeXtTiny** | **28M** | -- | **96.00%** |

> **Insight:** ConvNeXt achieves higher accuracy than VGG19 while using **80% fewer parameters**, demonstrating the efficiency of modern "Transformer-style" CNN blocks.

---

## 🛠️ How to Run

### Option 1: Google Colab (Recommended)
This project is optimized for the **T4 GPU** environment on Google Colab.

1.  Open `flutterby.ipynb` in Google Colab.
2.  Ensure Runtime is set to **T4 GPU** (*Runtime $\rightarrow$ Change runtime type*).
3.  Run all cells. The script automatically handles:
    * Dataset download (via KaggleHub).
    * Library installation.
    * Training & Evaluation.

### Option 2: Local Execution
If running locally, you need a GPU with at least 4GB VRAM.

```bash
# 1. Clone the repo
git clone https://github.com/DharmanSidhu/ButterflyClassfn_ConvNeXT-Tiny.git

# 2. Enter the directory
cd ButterflyClassfn_ConvNeXT-Tiny

# 3. Install dependencies
pip install tensorflow matplotlib seaborn scikit-learn kagglehub

# 4. Run the notebook
jupyter notebook flutterby.ipynb
