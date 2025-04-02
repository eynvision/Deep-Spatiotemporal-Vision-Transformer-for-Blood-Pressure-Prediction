Deep Spatiotemporal Vision Transformer for Blood Pressure Prediction
The proposed project aims to develop a novel image-based blood pressure (BP) prediction framework by
transforming ECG, PPG, and Bioimpedance (EBI) signals into spatiotemporal images and leveraging a
Vision Transformer (ViT) and Spatiotemporal CNN for feature extraction and fusion.
1. Data Collection and Preprocessing
- Data Collection:
- Collect synchronized ECG, PPG, and EBI signals along with corresponding BP measurements (systolic
and diastolic) from a reliable dataset or clinical trials.
- Ensure the dataset is diverse, covering different age groups, genders, and health conditions for robust
model generalization.
- Preprocessing:
- Remove noise and artifacts from raw signals using filtering techniques (e.g., bandpass filters for ECG
and PPG, low-pass filters for EBI).
- Normalize the signals to ensure consistent scaling across different modalities.
- Segment the signals into fixed-length windows (e.g., 10-second segments) for consistent input size.
2. Signal-to-Image Transformation
Transform the preprocessed time-series signals into image representations to leverage image-processing
techniques.
2.1 Scalogram Representation (Time-Frequency Analysis)
- Apply Continuous Wavelet Transform (CWT) or Short-Time Fourier Transform (STFT) to convert
ECG, PPG, and EBI signals into spectrograms.
- Generate 2D time-frequency heatmaps for each signal.
- Combine the spectrograms into a 3-channel image (ECG, PPG, EBI) per sample.
2.2 Recurrence Plots (RP)
- Convert ECG, PPG, and EBI time-series into recurrence images to capture dynamic system behavior.
- Use RP to detect nonlinear patterns and phase-space evolution in cardiovascular dynamics.
2.3 Gramian Angular Fields (GAF) & Markov Transition Fields (MTF)
- Transform signals into angular images (GAF) and transition probability images (MTF) to preserve
temporal dependencies.
- Combine these representations into multi-channel images for each signal.
A 3-channel physiological image dataset (ECG, PPG, and EBI images) per sample.
3. Multi-Modal Feature Extraction via Vision Transformer (ViT)
- Use a Vision Transformer (ViT) to extract high-level spatial and frequency features from the ECG, PPG,
and EBI images.
- ViT learns:
- Morphological patterns in PPG and ECG spectrograms.
- Vascular dynamics in EBI images.
- Temporal variations in Recurrence Plots and Gramian Angular Fields.
- Fine-tune the ViT architecture to optimize feature extraction for cardiovascular dynamics.
Deep representations of cardiovascular dynamics for BP estimation.
4. Spatiotemporal CNN for Feature Fusion
- Use a 3D Convolutional Neural Network (3D-CNN) to fuse the extracted features from ECG, PPG, and
EBI images.
- The 3D-CNN captures:
- Time-dependent BP variations.
- Multi-modal interactions across ECG, PPG, and EBI.
- Microvascular vs. macrovascular contributions to BP.
- Design the 3D-CNN architecture to handle spatiotemporal data effectively.
A fused model that captures real-time cardiovascular physiology with high fidelity.
5. Model Training and Optimization
- Loss Function:
- Use a combination of Mean Absolute Error (MAE) and Mean Squared Error (MSE) for BP prediction.
- Incorporate uncertainty estimation using techniques like Monte Carlo Dropout or Bayesian Neural
Networks.
- Training:
- Split the dataset into training, validation, and test sets (e.g., 70% training, 15% validation, 15%
testing).
- Train the ViT and 3D-CNN jointly using backpropagation and gradient descent optimization.
- Use data augmentation techniques (e.g., rotation, flipping, noise addition) to improve model
robustness.
- Hyperparameter Tuning:
- Optimize hyperparameters such as learning rate, batch size, and number of layers using grid search or
Bayesian optimization.
6. Model Evaluation
- Evaluate the model using standard metrics for regression tasks:
- Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE) for BP prediction.
- Correlation coefficient (R²) between predicted and actual BP values.
- Perform cross-validation to ensure model generalizability.
- Compare the proposed framework with baseline models (e.g., traditional machine learning models,
standalone CNNs, or RNNs).
