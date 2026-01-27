# Lecture Agenda: Physics-Informed Neural Networks for Vehicle Dynamics

## Presentation Outline

### I. Introduction & Problem Setup (Cells 1-5)
- Overview of the project: Processing .mat files for vehicle dynamics modeling
- Dataset description: Revs Vehicle Dynamics Database
  - Training: 2013 Targa Sixty-Six event (Car A)
  - Testing: 2014 Targa Sixty-Six event (Car B)
- Key variables explained:
  - Velocities: `vxCG`, `vyCG`
  - Accelerations: `axCG`, `ayCG`
  - Yaw rate: `yawRate`
  - Side slip angle: `sideSlip`

---

### II. Data Preprocessing Pipeline (Cells 6-20)
1. **Data Loading & Extraction**
   - Converting MATLAB .mat files to pandas DataFrames
   - Feature selection and time series extraction

2. **Test Data Preparation**
   - Temporal windowing (11.5-minute segment)
   - One-lap extraction for validation
   - Visualization of vehicle path

3. **Training Data Preparation**
   - NaN value handling via linear interpolation
   - Signal smoothing using Savitzky-Golay filter
   - Visualization: Raw vs. filtered signals

---

### III. Data Engineering for Deep Learning (Cells 21-30)
1. **Dataset Construction**
   - One-step-ahead prediction formulation
   - Input: `[vx_k, vy_k, r_k, ax_k, ay_k]`
   - Target: `[vx_{k+1}, vy_{k+1}]`

2. **Scaling Strategy**
   - MinMaxScaler for longitudinal velocity (vx)
   - StandardScaler for other features
   - Train/validation split (80/20)

3. **External Test Dataset (Car B)**
   - Cross-vehicle generalization assessment
   - Same preprocessing pipeline applied
   - Domain shift evaluation

---

### IV. Baseline Neural Network Model (Cells 31-35)
1. **Architecture**
   - Feed-forward network: 5 → 64 → 64 → 2
   - Activation: ReLU
   - Loss function: RMSE

2. **Training Results**
   - 50 epochs with Adam optimizer
   - Loss curves analysis
   - Prediction quality assessment

3. **Performance Analysis**
   - Residual analysis (vx, vy)
   - Statistical metrics (mean, std, RMSE)

---

### V. Physics-Informed Neural Networks (PINNs) (Cells 36-42)
1. **Kinematic Vehicle Model (KVM)**
   - Discrete-time equations:
     - `vx_{k+1} = vx_k + dt * (ax_k - r_k * vy_k)`
     - `vy_{k+1} = vy_k + dt * (ay_k + r_k * vx_k)`
   - Implementation and validation

2. **Physics Collocation Points**
   - Latin Hypercube Sampling (LHS)
   - Physically meaningful bounds
   - Random sampling strategy

3. **Physics Loss Computation**
   - Residual formulation: NN prediction vs. physics prediction
   - Combined loss: `L_total = w_data * L_data + w_phys * L_phys`
   - Weight decay mechanism

---

### VI. PINN Training Framework (Cells 43-52)
1. **Hyperparameter Configuration**
   - Learning rate, physics weight, decay rate
   - Batch size and collocation points (N_phys)

2. **Training Procedure**
   - Dual-loss optimization
   - Exponential decay for physics weight
   - Epoch-wise monitoring

3. **Model Evaluation**
   - Validation on Car A (20% holdout)
   - Testing on Car B (external dataset)
   - Metrics: Data loss, physics loss, total loss

---

### VII. Experimental Analysis (Cells 53-60)
1. **Effect of Physics Sampling Size (N_phys)**
   - Comparison: 64 vs. 512 samples
   - Impact on training stability and generalization

2. **Effect of Decay Rate**
   - Experiments: 0.0, 0.03, 0.1
   - Balance between data-driven and physics-informed learning

3. **Performance Visualization**
   - Loss curves over epochs
   - Prediction vs. ground truth plots
   - Residual distributions

---

### VIII. Hyperparameter Optimization (Cells 61-75)
1. **Ray Tune Framework**
   - MLflow integration for experiment tracking
   - ASHA scheduler for early stopping

2. **Search Algorithms Compared**
   - HyperOpt (Tree-structured Parzen Estimator)
   - Optuna (Bayesian optimization)
   - Random Search (baseline)

3. **Search Space**
   - Learning rate: [5e-4, 1e-2]
   - Physics weight: [0.05, 0.5]
   - Decay rate: [0.0, 0.1]
   - N_phys: [32, 512]
   - Hidden dimensions: [32, 64, 128, 256]

---

### IX. Interpretability & Feature Importance (Cells 76)
1. **SHAP (SHapley Additive exPlanations)**
   - Feature importance ranking
   - Understanding hyperparameter contributions

2. **Visualization**
   - Bar plots for global importance
   - Beeswarm plots for instance-level analysis

---

### X. Final Model Validation (Cells 77-82)
1. **Best Configuration Selection**
   - Parameter extraction from tuning experiments
   - Model retraining with optimal hyperparameters

2. **Test Set Evaluation (Car B)**
   - Complete dataset predictions
   - RMSE calculation for vx and vy
   - Comparison across optimization algorithms

3. **Results Export**
   - CSV files with predictions
   - Visualization of full-dataset performance

---

### XI. State Estimation for Unknown Variables

#### A. Problem Formulation and Motivation
1. **Real-World Challenge**
   - Scenario: Both velocities (vx and vy) are unmeasured/unavailable
   - Causes: Sensor failure, cost constraints, GPS-only systems without velocity sensors
   - Available sensors: Accelerometers (ax, ay) and gyroscope (yawRate)
   - Impact: Cannot directly use standard PINN for prediction without velocity measurements
   - Objective: Estimate both velocity states using only acceleration and yaw rate

2. **Mathematical Formulation**
   - Available measurements: ax, ay, yawRate (r)
   - Unknown variables: vx(k) and vy(k) for all timesteps
   - Initial conditions: vx(0) and vy(0) must be provided or estimated
   - Goal: Estimate complete velocity trajectory [vx(k), vy(k)] for k = 0, 1, 2, ..., N
   - Constraint: Maintain physical consistency with kinematic vehicle model
   - Challenge: Double integration of accelerations leads to drift without correction

#### B. PINN Self-Loop Architecture
1. **Conceptual Design**
   - Standard PINN input: `[vx_k, vy_k, r_k, ax_k, ay_k]` → output: `[vx_{k+1}, vy_{k+1}]`
   - Self-loop modification: Replace unknown vx_k and vy_k with estimated values
   - Recursive propagation: Both vx_est(k+1) and vy_est(k+1) become inputs for next iteration
   - Initial conditions: Use true vx(0), vy(0) or physics-based initialization
   - Bootstrap process: First prediction uses initial conditions, subsequent steps use PINN outputs

2. **Implementation Details**
   - Input construction: `[vx_est_k, vy_est_k, r_k, ax_k, ay_k]` where both velocities are estimated
   - Forward pass: PINN predicts `[vx_{k+1}, vy_{k+1}]`
   - State update: 
     - vx_est_{k+1} = PINN_output[0]
     - vy_est_{k+1} = PINN_output[1]
   - Loop continuation: Feed both estimated velocities as input for step k+2
   - Error propagation concern: Estimation errors in both states compound over time

3. **Physics-Informed Regularization**
   - Dual KVM constraint enforcement for both velocity components:
     - vx residual: `vx_pred - (vx_k + dt * (ax_k - r_k * vy_k))`
     - vy residual: `vy_pred - (vy_k + dt * (ay_k + r_k * vx_k))`
   - Cross-coupling: vx estimation affects vy and vice versa through yaw rate term
   - Prevents drift from physically feasible trajectories
   - Regularization weight balances estimation accuracy and physics compliance

#### C. Self-Loop Training Strategy
1. **Model Preparation**
   - Start with optimized PINN model from Section VIII
   - Option 1: Freeze PINN weights (use as fixed estimator)
   - Option 2: Fine-tune with self-loop loss (end-to-end training)
   - Transfer learning: Leverage knowledge from supervised prediction task

2. **Training Methodology**
   - Synthetic masked data: Remove both vx and vy from training sequences
   - Self-loop simulation during training: Propagate velocity estimates through time
   - Loss components:
     - L_estimation_vx = RMSE(vx_true, vx_estimated)
     - L_estimation_vy = RMSE(vy_true, vy_estimated)
     - L_physics_vx = RMSE(physics_vx_prediction, PINN_vx_prediction)
     - L_physics_vy = RMSE(physics_vy_prediction, PINN_vy_prediction)
     - L_total = w_est * (L_estimation_vx + L_estimation_vy) + w_phys * (L_physics_vx + L_physics_vy)

3. **Preventing Error Accumulation**
   - Teacher forcing: Periodically reset with true vx and vy values
   - Curriculum learning: Start with short sequences, increase length
   - Clipping: Constrain estimated velocities to physical bounds
   - Validation on held-out sequences
   - Special challenge: Coupled error propagation between vx and vy

#### D. RNN-Based Corrector Module
1. **Motivation for Hybrid Architecture**
   - PINN self-loop limitation: Accumulated drift over long sequences
   - Observation: Errors follow temporal patterns
   - Solution: Learn correction model for systematic biases
   - Hybrid benefit: Physics consistency (PINN) + temporal learning (RNN)

2. **Corrector Architecture**
   - Input: PINN self-loop estimates for both vx and vy + measurement residuals
   - RNN Type: LSTM (Long Short-Term Memory) or GRU (Gated Recurrent Unit)
   - Hidden layers: 32-128 units with dropout for regularization
   - Output: Correction terms [Δvx, Δvy] to add to PINN estimates
   - Final estimates: 
     - vx_final(k) = vx_PINN(k) + Δvx_RNN(k)
     - vy_final(k) = vy_PINN(k) + Δvy_RNN(k)

3. **Training the Corrector**
   - Data preparation: Compute residuals from PINN self-loop on training data
   - Targets: 
     - Correction for vx: vx_true - vx_PINN
     - Correction for vy: vy_true - vy_PINN
   - Sequence length: Variable (10-100 timesteps)
   - Loss function: MSE on correction predictions for both velocity components
   - Optimizer: Adam with learning rate scheduling
   - Regularization: Dropout, early stopping, L2 penalty

4. **Two-Stage Training Pipeline**
   - Stage 1: Train PINN with full supervision (Sections VI-VIII)
   - Stage 2: Freeze PINN, run self-loop to generate estimates
   - Stage 3: Train RNN corrector on PINN residuals
   - Optional Stage 4: Joint fine-tuning of PINN + RNN

#### E. Performance Evaluation and Results
1. **Evaluation Metrics**
   - RMSE for vx estimation: `sqrt(mean((vx_true - vx_est)^2))`
   - RMSE for vy estimation: `sqrt(mean((vy_true - vy_est)^2))`
   - RMSE for side-slip angle β: Converted from velocities using β = arctan(vy/vx)
   - Temporal error growth: RMSE vs. sequence length
   - Comparison baselines: Pure RNN, pure physics integration, standard PINN

2. **Quantitative Results**
   - PINN self-loop alone: RMSE(vy) ≈ 0.82 m/s, RMSE(β) ≈ 2.36 deg
   - PINN + RNN corrector: RMSE(vy) ≈ 0.29 m/s, RMSE(β) ≈ 0.78 deg
   - Improvement: ~65% reduction in vy estimation error
   - vx estimation performance and its impact on overall accuracy
   - Generalization: Performance on Car B test data

3. **Qualitative Analysis**
   - Visualization: True vs. estimated vx and vy over time
   - Error distribution: Histogram of estimation residuals for both velocities
   - Physics consistency: Check if estimates satisfy KVM equations
   - Critical maneuvers: Performance during high lateral and longitudinal acceleration
   - Coupling effects: How vx errors propagate to vy and vice versa

4. **Ablation Studies**
   - Impact of RNN architecture choice (LSTM vs. GRU)
   - Effect of sequence length on accuracy
   - Sensitivity to initial condition errors in both vx(0) and vy(0)
   - Robustness to measurement noise in accelerations
   - Contribution of each velocity component to overall estimation quality

---

### XII. Key Takeaways & Lessons Learned
1. **Progression Journey**
   - Data preprocessing → Baseline NN → PINN → Optimized PINN → State Estimation
   - Importance of domain knowledge integration
   - From prediction to estimation: handling missing information

2. **Challenges Encountered**
   - NaN handling strategies
   - Scaling across different vehicles
   - Balancing physics and data losses
   - Error accumulation in self-loop estimation

3. **Best Practices**
   - Feature-wise scaling
   - Physics loss weight scheduling
   - Systematic hyperparameter tuning
   - Cross-vehicle validation
   - Hybrid physics-data approaches for robustness

---

### XIII. Questions & Discussion
- Open floor for questions
- Future improvements discussion
- Applications to other domains
- Extension to multi-sensor fusion scenarios