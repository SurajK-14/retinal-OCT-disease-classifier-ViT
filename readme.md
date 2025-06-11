# Retinal Disease Classifier using OCT Imaging and Vision Transformer (ViT)

This project is a preliminary implementation of a retinal disease classifier using Optical Coherence Tomography (OCT) images and a pretrained Vision Transformer (ViT) model.

## Overview

- **Goal:** Classify retinal diseases from OCT images using deep learning.
- **Model:** [ViT (Vision Transformer)](https://huggingface.co/docs/transformers/model_doc/vit?usage=Pipeline) (`vit_tiny_patch16_224`).
- **Dataset:** [Kermany2018 OCT Dataset](https://www.kaggle.com/datasets/paultimothymooney/kermany2018)
*Note: For this project, a **subset** of the original OCT2017 dataset was used due to limited time and GPU resources.*
- **Frameworks:** PyTorch, timm, scikit-learn

## Features

- Data loading and preprocessing with `torchvision`
- Transfer learning with pretrained ViT
- Training and evaluation loops
- ROC and Precision-Recall curve plotting
- Classification report and metrics
- (Planned) Class Activation Maps (CAM) for interpretability

## Usage

1. **Clone the repository:**
    ```bash
    git clone https://github.com/SurajK-14/retinal-OCT-disease-classifier-ViT.git
    cd retinal-OCT-disease-classifier-ViT
    ```

2. **Download the dataset:**
    - Download [OCT2017.zip](https://www.kaggle.com/datasets/paultimothymooney/kermany2018) from Kaggle and place it in the `dataset/` directory.

3. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Or install manually: torch, torchvision, timm, scikit-learn, matplotlib, seaborn, pandas, tqdm, pytorch-grad-cam)*

4. **Run the notebook:**
    - Open `vit1.ipynb` in Jupyter or VS Code and run all cells.

## Results

- The notebook will print training/validation metrics and save ROC/PR curves in the `results/` folder.

## Roadmap

- [ ] Add more advanced data augmentation
- [ ] Hyperparameter tuning
- [ ] Add Class Activation Maps (CAM) visualization
- [ ] Experiment with other transformer architectures
- [ ] Improve documentation and code structure

## Acknowledgements

- [Kermany et al., 2018 OCT Dataset](https://www.kaggle.com/datasets/paultimothymooney/kermany2018)
- [timm library](https://github.com/huggingface/pytorch-image-models)
- [PyTorch](https://pytorch.org/)

---

*This is a work in progress. Contributions and suggestions are welcome!*