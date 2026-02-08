# Cross-Platform Installation Guide

**This project works on Windows, macOS, and Linux!**

---

## Quick Start (All Platforms)

### 1. Navigate to Project Directory

**Windows (PowerShell or CMD):**
```powershell
cd path\to\UAV_Bird_Classification
```

**macOS/Linux (Terminal):**
```bash
cd path/to/UAV_Bird_Classification
```

### 2. Install Dependencies

**Windows (PowerShell or CMD):**
```powershell
python -m pip install -r requirements.txt
```

**macOS/Linux (Terminal):**
```bash
python3 -m pip install -r requirements.txt
# OR (with pip)
pip install -r requirements.txt
```

### 3. Run the System

**Windows (PowerShell or CMD):**
```powershell
python quickstart.py
```

**macOS/Linux (Terminal):**
```bash
python3 quickstart.py
# OR
python quickstart.py
```

---

## Detailed Setup by OS

### Windows Setup

#### Step 1: Install Python

1. Download Python 3.8+ from https://www.python.org/downloads/
2. **IMPORTANT**: Check "Add Python to PATH" during installation
3. Verify installation:
   ```powershell
   python --version
   pip --version
   ```

#### Step 2: Clone/Download Project

```powershell
# If using Git
git clone https://github.com/shivanipeesari/UAV-micro-doppler-spectrogram.git
cd UAV-micro-doppler-spectrogram\src\UAV_Bird_Classification

# OR manually download and extract, then navigate to folder
cd C:\path\to\UAV_Bird_Classification
```

#### Step 3: Create Virtual Environment (Recommended)

```powershell
# Create virtual environment
python -m venv venv

# Activate it
venv\Scripts\activate

# You should see (venv) in your prompt
```

#### Step 4: Install Dependencies

```powershell
# With virtual environment activated
python -m pip install -r requirements.txt

# Or without virtual environment
pip install -r requirements.txt
```

#### Step 5: Run Project

```powershell
python quickstart.py
```

#### Troubleshooting Windows

**Problem**: "python is not recognized"
- **Solution**: Add Python to PATH or use `py` command:
  ```powershell
  py quickstart.py
  ```

**Problem**: "Cannot find module cv2"
- **Solution**: Reinstall OpenCV:
  ```powershell
  pip install --upgrade opencv-python
  ```

**Problem**: Permission denied on virtual environment
- **Solution**: Run PowerShell as Administrator

---

### macOS Setup

#### Step 1: Install Python

**Option A: Using Homebrew (Recommended)**
```bash
# Install Homebrew if not already installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python
brew install python@3.11
```

**Option B: Using Python.org**
1. Download from https://www.python.org/downloads/macos/
2. Run the installer

**Option C: Using MacPorts**
```bash
port install python311
```

#### Step 2: Verify Installation

```bash
python3 --version
pip3 --version
```

#### Step 3: Clone/Download Project

```bash
# Using Git
git clone https://github.com/shivanipeesari/UAV-micro-doppler-spectrogram.git
cd UAV-micro-doppler-spectrogram/src/UAV_Bird_Classification

# OR manually download and extract
cd ~/Downloads/UAV_Bird_Classification
```

#### Step 4: Create Virtual Environment (Recommended)

```bash
python3 -m venv venv
source venv/bin/activate

# You should see (venv) in your prompt
```

#### Step 5: Install Dependencies

```bash
# With virtual environment activated
pip install -r requirements.txt

# Or directly
pip3 install -r requirements.txt
```

#### Step 6: Run Project

```bash
python3 quickstart.py
```

#### Troubleshooting macOS

**Problem**: "ModuleNotFoundError: No module named 'tensorflow'"
- **Solution**:
  ```bash
  pip install --upgrade tensorflow
  ```

**Problem**: "Port 5900 already in use" (Jupyter)
- **Solution**: This is fine, just use the command line

**Problem**: Homebrew-related issues
- **Solution**: Run:
  ```bash
  brew reinstall python@3.11
  ```

---

### Linux Setup

#### Step 1: Install Python and pip

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install python3 python3-pip python3-venv
```

**Fedora/RHEL:**
```bash
sudo dnf install python3 python3-pip
```

**Arch:**
```bash
sudo pacman -S python python-pip
```

**Verify:**
```bash
python3 --version
pip3 --version
```

#### Step 2: Clone/Download Project

```bash
git clone https://github.com/shivanipeesari/UAV-micro-doppler-spectrogram.git
cd UAV-micro-doppler-spectrogram/src/UAV_Bird_Classification

# OR download manually
cd ~/Downloads/UAV_Bird_Classification
```

#### Step 3: Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate

# You should see (venv) in your prompt
```

#### Step 4: Install Dependencies

```bash
pip install -r requirements.txt
```

#### Step 5: Run Project

```bash
python3 quickstart.py
```

#### Troubleshooting Linux

**Problem**: "Permission denied" when installing
- **Solution**: Don't use `sudo pip`. Use virtual environment instead
- **If really needed**:
  ```bash
  sudo -H pip3 install -r requirements.txt
  ```

**Problem**: Missing system dependencies for cv2
- **Solution (Ubuntu/Debian)**:
  ```bash
  sudo apt install libatlas-base-dev libjasper-dev libtiff5-dev libjasper-dev libjpeg-dev zlib1g-dev
  pip install opencv-python
  ```

**Problem**: "No module named 'tensorflow'"
- **Solution**:
  ```bash
  pip install tensorflow
  # If you have GPU:
  pip install tensorflow[and-cuda]
  ```

---

## Using Python Command Instead of python3

If `python` doesn't work and you need to use `python3`:

### On Windows
```powershell
# Use 'py' instead
py quickstart.py
py generate_data.py
```

### On macOS/Linux
```bash
# Create an alias (permanent)
echo "alias python=python3" >> ~/.bashrc
source ~/.bashrc

# Then use normally
python quickstart.py
```

---

## Virtual Environment Management

### Activate (All Platforms)

**Windows:**
```powershell
venv\Scripts\activate
```

**macOS/Linux:**
```bash
source venv/bin/activate
```

### Deactivate (All Platforms)

```bash
deactivate
```

### Remove Virtual Environment

**Windows:**
```powershell
rmdir /s venv
```

**macOS/Linux:**
```bash
rm -rf venv
```

---

## Verify Installation

Run this to check all dependencies:

```bash
# All platforms (use python or python3)
python -c "
import tensorflow, cv2, numpy, sklearn, pandas, matplotlib
print('✓ TensorFlow:', tensorflow.__version__)
print('✓ OpenCV:', cv2.__version__)
print('✓ NumPy:', numpy.__version__)
print('✓ Scikit-learn:', sklearn.__version__)
print('✓ Pandas:', pandas.__version__)
print('✓ Matplotlib:', matplotlib.__version__)
print('\n✅ All dependencies installed!')
"
```

---

## Execution Commands (All Platforms)

### Generate Data
```bash
python quickstart.py        # Automatic everything
# OR
python generate_data.py     # Just data generation
```

### Run Pipeline
```bash
python run_pipeline.py
```

### Make Predictions
```bash
python predict_new.py dataset/UAV/uav_001.png
```

### View Results
```bash
# Windows
type reports\summary_report.txt

# macOS/Linux
cat reports/summary_report.txt
```

---

## GPU Support (Optional)

### NVIDIA GPU (CUDA)

```bash
# Install CUDA-enabled TensorFlow
pip install tensorflow[and-cuda]

# Verify GPU is detected
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

### Apple Silicon (M1/M2 Mac)

```bash
# TensorFlow with Metal acceleration
pip install tensorflow-macos

# And Metal plugin
pip install tensorflow-metal
```

---

## Still Having Issues?

1. **Check Python version**: Must be 3.8+
   ```bash
   python --version
   ```

2. **Check pip works**:
   ```bash
   pip list
   ```

3. **Reinstall everything**:
   ```bash
   pip install --upgrade --force-reinstall -r requirements.txt
   ```

4. **Check GitHub issues**: https://github.com/shivanipeesari/UAV-micro-doppler-spectrogram/issues

---

## Quick Reference Table

| Task | Windows | macOS | Linux |
|------|---------|-------|-------|
| Navigate | `cd path\to\` | `cd path/to/` | `cd path/to/` |
| Python check | `python --version` | `python3 --version` | `python3 --version` |
| Install deps | `pip install -r requirements.txt` | `pip install -r requirements.txt` | `pip install -r requirements.txt` |
| Create venv | `python -m venv venv` | `python3 -m venv venv` | `python3 -m venv venv` |
| Activate venv | `venv\Scripts\activate` | `source venv/bin/activate` | `source venv/bin/activate` |
| Run project | `python quickstart.py` | `python3 quickstart.py` | `python3 quickstart.py` |
| View results | `type reports\summary...` | `cat reports/summary...` | `cat reports/summary...` |

---

**You're ready to go! Choose your OS above and follow the steps. The system is 100% cross-platform compatible.** 🚀
