# ML Assignment - Pixel Coordinate Prediction

## Problem Statement
Using Deep Learning techniques, predict the coordinates (x, y) of a pixel which has a value of 255 in a 50×50 grayscale image where all other pixels are 0. The bright pixel is randomly assigned.

## Solution Approach

### Mathematical Formulation
- **Input**: Grayscale image of shape (1, 50, 50)
- **Output**: Coordinates [x, y] where x, y ∈ [0, 49]
- **Model**: Convolutional Neural Network (CNN)
- **Loss Function**: Mean Squared Error (MSE)

### Architecture
- **Conv2D Block 1**: 32 filters, 3×3 kernel, ReLU, MaxPool
- **Conv2D Block 2**: 64 filters, 3×3 kernel, ReLU, MaxPool
- **Fully Connected**: 128 → 64 → 2 neurons (Sigmoid output)
- **Dropout**: 0.3 for regularization

## Installation Instructions

### Prerequisites
- Python 3.10 or higher
- pip package manager

### Step 1: Clone or Download the Repository
```bash
cd "ML Assignment"
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

This will install:
- PyTorch 2.10.0 (Deep learning framework)
- NumPy 2.4.2 (Numerical computing)
- Matplotlib 3.10.8 (Visualization)
- Scikit-learn 1.8.0 (ML utilities)
- OpenCV 4.13.0.90 (Image processing)
- Pandas 3.0.0 (Data analysis)
- Jupyter 1.1.1 (Notebook environment)

### Step 3: Launch Jupyter Notebook
```bash
jupyter notebook ml_assignment.ipynb
```
Or use JupyterLab:
```bash
jupyter lab
```

### Step 4: Run All Cells
In Jupyter, select "Run All" from the Cell menu or press `Shift+Enter` to execute cells sequentially.

## Dataset Rationale

### Why Synthetic Data Generation?
1. **Perfect Ground Truth**: We know the exact pixel coordinates
2. **No Memorization**: Random generation prevents the model from memorizing fixed patterns
3. **Scalability**: Can generate unlimited training samples
4. **Uniform Distribution**: Ensures no positional bias (all locations equally likely)
5. **Controlled Complexity**: Clean signal (single bright pixel) simplifies learning

### Dataset Specifications
- **Training Set**: 8,000 images
- **Validation Set**: 2,000 images
- **Image Size**: 50×50 pixels
- **Normalization**: Images [0, 1], Coordinates [0, 1]

## Project Structure
```
ML Assignment/
├── ml_assignment.ipynb    # Main notebook with training and evaluation
├── requirements.txt       # Python dependencies
├── README.md             # This file
└── best_model.pt         # Saved model checkpoint (after training)
```

## Running the Project

### Sequential Execution
All cells in the notebook run sequentially:
1. **Import Libraries**: Load all required dependencies
2. **Data Generation**: Create synthetic dataset
3. **Data Preprocessing**: Normalize and split data
4. **Model Architecture**: Define CNN model
5. **Training Configuration**: Setup loss function and optimizer
6. **Training Loop**: Train for 50 epochs with checkpointing
7. **Evaluation**: Load best model and calculate metrics
8. **Visualization**: Plot predictions vs ground truth

### Expected Output
- Training logs showing decreasing loss
- Model checkpoint saved to `best_model.pt`
- Mean Absolute Error (MAE) in pixels
- Visual comparison of predicted vs true coordinates
- Training/validation loss curves

## Evaluation Criteria

### Functionality ✓
- Model successfully predicts pixel coordinates
- Loss decreases during training
- Approach is sound and well-documented

### Code Quality ✓
- PEP8 compliant (snake_case, proper indentation)
- Comprehensive docstrings
- Well-organized and readable
- Error handling and validation

### Performance
- **Target**: Low MAE (< 2 pixels on average)
- **Metric**: Mean Absolute Error in pixel coordinates
- **Validation**: Visual inspection of predictions

## Key Results
After training, the model achieves:
- Convergent training (loss decreases)
- Accurate coordinate prediction (low MAE)
- Good generalization (train/val loss similar)

## Author
ML Assignment - Supervised Regression

## License
Educational project for ML assignment submission.
