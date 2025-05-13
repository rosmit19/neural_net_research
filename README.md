# � Discovery Project: BP vs. FF 

## 📌 Overview
This project compares **Backpropagation (BP) and Forward-Forward (FF)** training algorithms using a Convolutional Neural Network (CNN) on the MNIST/CIFAR-10 datasets.  
The goal is to analyze **power consumption, accuracy, GPU memory usage, and training time** for both methods.

---

## 📂 Repository Structure

**discovery_project/**

│── **.gitignore**   # Files to ignore in Git

│── **README.md**    # Project documentation

│── **req.txt**      # Python dependencies

│── **scripts/**     # Python training scripts
  - ├── **mnist_bp_cnn.py**   # BP training script
  - ├── **mnist_ff_cnn.py**    # FF training script

│── **data/**    # MNIST dataset (stored locally)

│── **models/**  # Trained model checkpoints

│── **logs/**    # Training logs (ignored in .gitignore)

│── **slurm/** # SLURM job scripts for running on HPC
  - ├── **bp_train.sbatch**   # SLURM script for BP
  - ├── **ff_train.sbatch**   # SLURM script for FF

│── **results/**  # CSV & visualizations for comparison

---

## 🔧 Setup Instructions
### **1️⃣ Clone the Repository**

```sh
git clone https://github.com/YOUR_USERNAME/discovery_project.git
cd discovery_project
```

### **2️⃣ Set Up Virtual Environment**

```sh
python -m venv venv
source venv/bin/activate  # For Linux/macOS
venv\Scripts\activate     # For Windows
```
### **3️⃣ Install Dependencies**
```sh
pip install -r req.text
```
### **4️⃣ Submit SLURM Jobs on HPC**
```sh
sbatch slurm/bp_train.sbatch  # Runs Backpropagation Training
sbatch slurm/ff_train.sbatch  # Runs Forward-Forward Training
```
### **5️⃣ View Logs & Results**

- Training logs are stored in the logs/ directory.
- Performance results (accuracy, power, memory usage) are saved in results/.

---

## 📊 Evaluation Metrics
We compare BP vs. FF based on:

- Training Time (seconds)
- GPU Memory Usage (MB)
- Accuracy (%)
- Power Consumption (W)

---

## 📜 Research Findings (To Be Updated)

---

## 🛠 Troubleshooting
- Ensure SSH is set up
- Ampere 40 & 80 GPUs do not allow direct power tracking

