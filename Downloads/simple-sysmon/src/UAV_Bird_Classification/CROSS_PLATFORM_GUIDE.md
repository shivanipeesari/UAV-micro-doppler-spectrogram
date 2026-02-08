# 🚀 Cross-Platform Quick Start Guide

**Your project now works on Windows, macOS, and Linux!**

---

## ✅ What Changed

- Fixed `quickstart.py` to use cross-platform Python commands
- Updated help messages to use universal commands
- Created detailed installation guides for each OS
- All code uses `pathlib.Path` for cross-platform file paths

---

## ⚡ Universal Commands (All Systems)

### 1. Navigate to Project
```bash
cd UAV_Bird_Classification
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Everything
```bash
python quickstart.py
```

### 4. View Results
```bash
cat reports/summary_report.txt
```

---

## 🪟 Windows Users

### Setup Steps

1. **Install Python**: https://www.python.org/downloads/
   - ✅ **Check "Add Python to PATH"**

2. **Open PowerShell or CMD** and run:
   ```powershell
   cd path\to\UAV_Bird_Classification
   pip install -r requirements.txt
   python quickstart.py
   ```

3. **View Results**:
   ```powershell
   type reports\summary_report.txt
   ```

### Windows Troubleshooting

| Problem | Solution |
|---------|----------|
| `python not found` | Use `py` instead: `py quickstart.py` |
| `pip not found` | Use: `python -m pip install -r requirements.txt` |
| Permission errors | Run PowerShell as Administrator |

**Full Windows guide**: See `INSTALL_WINDOWS_MAC_LINUX.md`

---

## 🍎 macOS Users

### Setup Steps

1. **Install Python** (if not already installed):
   ```bash
   brew install python@3.11
   # OR download from https://www.python.org/downloads/
   ```

2. **Open Terminal** and run:
   ```bash
   cd path/to/UAV_Bird_Classification
   pip install -r requirements.txt
   python3 quickstart.py
   ```

3. **View Results**:
   ```bash
   cat reports/summary_report.txt
   ```

### macOS Troubleshooting

| Problem | Solution |
|---------|----------|
| `python3 not found` | Use Homebrew: `brew install python@3.11` |
| `TensorFlow issues` | Use: `pip install --upgrade tensorflow` |
| `OpenCV issues` | Use: `pip install --upgrade opencv-python` |

**Full macOS guide**: See `INSTALL_WINDOWS_MAC_LINUX.md`

---

## 🐧 Linux Users

### Setup Steps

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install python3 python3-pip
cd path/to/UAV_Bird_Classification
pip install -r requirements.txt
python3 quickstart.py
```

**Fedora/RHEL:**
```bash
sudo dnf install python3 python3-pip
cd path/to/UAV_Bird_Classification
pip install -r requirements.txt
python3 quickstart.py
```

### View Results:
```bash
cat reports/summary_report.txt
```

### Linux Troubleshooting

| Problem | Solution |
|---------|----------|
| `permission denied` | Use virtual environment (no `sudo pip`) |
| Missing cv2 deps | `sudo apt install libsm6 libxext6` |
| TensorFlow issues | `pip install --upgrade tensorflow` |

**Full Linux guide**: See `INSTALL_WINDOWS_MAC_LINUX.md`

---

## 🌍 Universal Command Reference

These commands work on **all three platforms**:

| Action | Command |
|--------|---------|
| Check Python | `python --version` |
| Install deps | `pip install -r requirements.txt` |
| Run project | `python quickstart.py` |
| Generate data | `python generate_data.py` |
| Run pipeline | `python run_pipeline.py` |
| Predict | `python predict_new.py dataset/UAV/uav_001.png` |
| View results | `cat reports/summary_report.txt` |
| View CSV | `cat reports/evaluation_report.csv` |

---

## 🔧 System Requirements (All Platforms)

- Python 3.8 or higher
- pip (Python package manager)
- 4 GB RAM minimum (8 GB recommended)
- Internet connection for first install

---

## 📚 Detailed Guides

For complete setup instructions, see:

- **Windows**: `INSTALL_WINDOWS_MAC_LINUX.md` → Windows Section
- **macOS**: `INSTALL_WINDOWS_MAC_LINUX.md` → macOS Section  
- **Linux**: `INSTALL_WINDOWS_MAC_LINUX.md` → Linux Section

---

## ✨ Key Features

✅ Works on Windows, macOS, Linux
✅ No hardcoded paths
✅ Uses standard Python commands
✅ Cross-platform Path handling
✅ Clear error messages

---

## 🚨 Quick Fix for Any System

If you encounter issues:

```bash
# 1. Ensure you're in the project directory
cd UAV_Bird_Classification

# 2. Use system Python
python -m pip install --upgrade pip

# 3. Reinstall dependencies
python -m pip install -r requirements.txt

# 4. Run project
python quickstart.py
```

---

## 📱 Platform-Specific Notes

### Windows
- Use `python` instead of `python3`
- Use PowerShell or CMD, not Bash
- Use `\` for paths or forward slashes `/`
- Virtual env: `venv\Scripts\activate`

### macOS
- Use `python3` if `python` gives error
- Most commands same as Linux
- Virtual env: `source venv/bin/activate`
- May need to accept security prompts

### Linux
- Use `python3` for Python 3.x
- Use `sudo` carefully (preferably with venv)
- Virtual env: `source venv/bin/activate`
- Most standard Python conventions apply

---

## 🎯 Next Steps

1. **First time?** → Follow the quick start above
2. **Need detailed help?** → Check `INSTALL_WINDOWS_MAC_LINUX.md`
3. **System not working?** → See Troubleshooting section above
4. **Ready for viva?** → Check `VIVA_QA_DOCUMENT.md`

---

## 💡 Pro Tips

- **Use virtual environments** to keep Python clean
- **Update pip first**: `python -m pip install --upgrade pip`
- **For macOS M1/M2 Macs**: TensorFlow works great with native acceleration
- **For Windows**: PowerShell is better than CMD

---

**Your system is now truly cross-platform! 🎉**

```
┌─────────────────────────────────────┐
│  Works on:                          │
│  ✅ Windows 10/11                   │
│  ✅ macOS (Intel & Apple Silicon)   │
│  ✅ Linux (Ubuntu, Fedora, etc.)    │
└─────────────────────────────────────┘
```
