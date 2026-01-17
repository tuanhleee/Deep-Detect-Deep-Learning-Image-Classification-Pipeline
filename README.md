# Deep-Detect-Deep-Learning-Image-Classification-Pipeline

##  Dataset, Model Training and Results

###  Dataset: https://www.kaggle.com/datasets/ayushmandatta1/deepdetect-2025

The dataset is organized using a directory-based structure compatible with TensorFlow and Keras utilities.  
Each class is stored in a separate folder, allowing automatic label assignment during data loading.

Example structure:

dataset/
├── train/
│   ├── class_1/
│   ├── class_2/
│   └── class_n/
├── val/
│   ├── class_1/
│   ├── class_2/
│   └── class_n/
└── test/
    ├── class_1/
    ├── class_2/
    └── class_n/

Images are resized and normalized during preprocessing to match the input requirements of the selected models.

---

### Model Architecture

Two EfficientNet architectures were evaluated during training:

- **EfficientNet-B0**
- **EfficientNet-B4**

Both models were used as backbone networks and fine-tuned for the classification task.  
EfficientNet was chosen for its strong trade-off between accuracy and computational efficiency.

---

###  Training Configuration

The models were trained with the following configuration:

- **Number of epochs:** 50  
- **Optimizer:** Adam  
- **Loss function:** Categorical / Sparse Categorical Crossentropy  
- **Input size:** Adapted to EfficientNet requirements  
- **Training strategy:** Fine-tuning with frozen and unfrozen layers

Training and validation datasets were clearly separated to ensure reliable performance evaluation.

---

###  Results

After 50 training epochs, the models achieved the following performance on the test dataset:

- **Test accuracy:** ~ **86%**

This result demonstrates good generalization capabilities on unseen data and confirms the effectiveness of EfficientNet architectures for this classification task.

To better illustrate the performance, **sample prediction images and accuracy curves** are included in the project to visualize:
- Model convergence during training
- Correct and incorrect predictions on test images

---

###  Evaluation and Testing

Model evaluation was conducted using the dedicated Jupyter notebook:

```bash
test_model.ipynb
