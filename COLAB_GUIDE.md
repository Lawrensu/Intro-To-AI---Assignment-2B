# Google Colab Training Guide for RNN/LSTM Models

## Quick Start for Collaborators

This guide helps your team train the RNN/LSTM models on Google Colab with free GPU access.

---

## Prerequisites

- Google account (for Google Colab access)
- GitHub repository link
- Internet connection

---

## Step-by-Step Instructions

### 1. Open Google Colab

1. Go to [https://colab.research.google.com/](https://colab.research.google.com/)
2. Sign in with your Google account

### 2. Upload the Notebook

**Option A: From GitHub (Recommended)**
1. In Colab, click **File** → **Open Notebook**
2. Click the **GitHub** tab
3. Enter your repository URL: `https://github.com/Lawrensu/Intro-To-AI---Assignment-2B`
4. Select `RNN_LSTM_Training_Colab.ipynb`

**Option B: Upload File**
1. Download `RNN_LSTM_Training_Colab.ipynb` from the repository
2. In Colab, click **File** → **Upload Notebook**
3. Select the downloaded file

### 3. Enable GPU (Important!)

1. In Colab, click **Runtime** → **Change runtime type**
2. Under **Hardware accelerator**, select **T4 GPU**
3. Click **Save**

This gives you free GPU access for faster training!

### 4. Run the Notebook

Execute cells in order by:
- Clicking the play button (▶) on each cell, OR
- Press `Shift + Enter` to run current cell and move to next

**Important cells to run:**
1. ✅ Check GPU availability
2. ✅ Install packages (if needed)
3. ✅ Clone repository
4. ✅ Import models
5. ✅ Generate training data
6. ✅ Train RNN model (~5-10 minutes)
7. ✅ Train LSTM model (~5-10 minutes)
8. ✅ Download models

### 5. Download Trained Models

The last cell will download these files:
- `rnn_pattern_model.pth` - Traffic Pattern RNN
- `lstm_travel_time_model.pth` - Travel Time LSTM
- `rnn_pattern_training_history.png` - Training curves
- `lstm_travel_time_training_history.png` - Training curves

**Save these files in your local project's `models/` directory!**

---

## Training Configuration

### Default Settings (Good for Most Cases)

```python
# Data
n_samples = 2000       # Number of training samples
seq_length = 30        # Sequence length
batch_size = 32        # Batch size

# Model
hidden_size = 64       # Hidden units
num_layers = 2         # RNN/LSTM layers
dropout = 0.3          # Dropout rate

# Training
epochs = 30            # Training epochs
learning_rate = 0.001  # Learning rate
```

### Adjusting Parameters

If training is too slow:
- Reduce `n_samples` to 1000
- Reduce `epochs` to 20
- Increase `batch_size` to 64

If you want better accuracy:
- Increase `n_samples` to 3000
- Increase `epochs` to 50
- Increase `hidden_size` to 128

---

## Expected Training Time

On Google Colab with T4 GPU:
- **RNN Model**: ~5-10 minutes (30 epochs)
- **LSTM Model**: ~5-10 minutes (30 epochs)
- **Total**: ~15-20 minutes

On CPU (not recommended):
- **RNN Model**: ~20-30 minutes
- **LSTM Model**: ~20-30 minutes
- **Total**: ~45-60 minutes

---

## Expected Performance

### Traffic Pattern RNN
- **Training Accuracy**: 85-95%
- **Validation Accuracy**: 80-90%
- **Test Accuracy**: 80-90%

### Travel Time LSTM
- **Training MAE**: 5-10 minutes
- **Validation MAE**: 8-12 minutes
- **Test MAE**: 8-12 minutes

*Note: These are for synthetic data. Real data may vary.*

---

## Common Issues and Solutions

### Issue 1: "CUDA out of memory"
**Solution:**
- Reduce `batch_size` to 16
- Reduce `hidden_size` to 32
- Restart runtime: **Runtime** → **Restart runtime**

### Issue 2: "Module not found"
**Solution:**
- Make sure you ran the "Clone repository" cell
- Check that the current directory is correct: `%cd Intro-To-AI---Assignment-2B`

### Issue 3: "No GPU available"
**Solution:**
- Change runtime type to GPU (see Step 3)
- If GPU quota exceeded, wait or use CPU (it will be slower)

### Issue 4: Training is very slow
**Solution:**
- Check GPU is enabled: Run `print(torch.cuda.is_available())` should be `True`
- Reduce number of samples or epochs
- Use a smaller batch size

### Issue 5: Colab disconnects
**Solution:**
- Colab has 90-minute idle timeout
- Keep the tab active
- If disconnected, re-run from beginning (training progress is lost)

---

## Alternative: Training Locally

If you prefer to train locally instead of Colab:

### Windows (PowerShell)
```powershell
# Navigate to project
cd C:\Project\AI\Intro-To-AI---Assignment-2B

# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Train both models
python src/train_lstm.py --model both --epochs 30

# Train only RNN
python src/train_lstm.py --model rnn --epochs 30

# Train only LSTM
python src/train_lstm.py --model lstm --epochs 30
```

### Mac/Linux
```bash
# Navigate to project
cd ~/path/to/Intro-To-AI---Assignment-2B

# Activate virtual environment
source venv/bin/activate

# Train both models
python src/train_lstm.py --model both --epochs 30
```

### Custom Parameters
```bash
# Example: More samples, longer training
python src/train_lstm.py \
  --model both \
  --n_samples 3000 \
  --epochs 50 \
  --hidden_size 128 \
  --batch_size 64
```

---

## After Training

### 1. Verify Models Work
```bash
# Run demo script
python demo_rnn_lstm.py
```

Expected output:
- Traffic pattern predictions for different scenarios
- Travel time predictions for different routes
- Route comparison with ML predictions

### 2. Check Model Files

Ensure these files exist in `models/`:
- ✅ `rnn_pattern_model.pth`
- ✅ `lstm_travel_time_model.pth`
- ✅ `rnn_pattern_training_history.png`
- ✅ `lstm_travel_time_training_history.png`
- ✅ `rnn_pattern_info.json`
- ✅ `lstm_travel_time_info.json`

### 3. Review Training Plots

Open the PNG files to check:
- Loss should decrease over epochs
- Accuracy (RNN) should increase
- MAE (LSTM) should decrease
- No large gaps between train/val curves (indicates good generalization)

---

## Integration with Main Project

### Using the Models in Code

```python
from models.rnn_model import (
    TrafficPatternRNN,
    TravelTimeLSTM,
    load_model,
    predict_travel_time,
    analyze_traffic_pattern
)

# Load models
rnn_model = TrafficPatternRNN(...)
rnn_model = load_model(rnn_model, 'models/rnn_pattern_model.pth')

lstm_model = TravelTimeLSTM(...)
lstm_model = load_model(lstm_model, 'models/lstm_travel_time_model.pth')

# Analyze traffic pattern
traffic_data = np.array([...])  # Shape: (30, 10)
pattern_probs = analyze_traffic_pattern(rnn_model, traffic_data)

# Predict travel time
path_data = np.array([...])  # Shape: (30, 15)
travel_time = predict_travel_time(lstm_model, path_data)
```

### Integration with Pathfinding

Add this to your pathfinding algorithm:

```python
def calculate_adjusted_time(base_time, path_features, lstm_model):
    """
    Adjust path time using LSTM prediction
    """
    predicted_time = predict_travel_time(lstm_model, path_features)
    return predicted_time

# In your pathfinding algorithm
for path in candidate_paths:
    path_features = extract_path_features(path)  # Your function
    adjusted_time = calculate_adjusted_time(
        base_time=path.base_time,
        path_features=path_features,
        lstm_model=lstm_model
    )
    path.total_time = adjusted_time
```

---

## Team Workflow

### For the Team Leader
1. Train models on Colab (or locally)
2. Commit trained models to shared drive (not Git - files too large)
3. Share download link with team
4. Update README with model performance metrics

### For Team Members
1. Download trained models from shared drive
2. Place in local `models/` directory
3. Run `demo_rnn_lstm.py` to verify
4. Use models in your assigned components

### For Everyone
- **Don't commit .pth files to Git** (too large)
- **Do commit** code, notebooks, and documentation
- **Do share** training plots and metrics
- **Keep** model info JSON files in Git

---

## Additional Resources

### Documentation
- `src/models/rnn_model.py` - Full model implementation
- `src/models/MODELS.md` - Model specifications
- `src/train_lstm.py` - Training script
- `demo_rnn_lstm.py` - Usage examples

### Tutorials
- [PyTorch RNN Tutorial](https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html)
- [LSTM Time Series Forecasting](https://machinelearningmastery.com/time-series-forecasting-long-short-term-memory-network-python/)

### Help
- Check issues on GitHub repository
- Ask in team group chat
- Refer to PyTorch documentation

---

## Success Checklist

Before considering RNN/LSTM training complete:

- [ ] Google Colab notebook runs without errors
- [ ] GPU is enabled and used for training
- [ ] Both models trained successfully
- [ ] Training plots show good learning curves
- [ ] Models downloaded and saved locally
- [ ] Demo script runs and shows predictions
- [ ] Model files in correct directory
- [ ] Team members can access trained models
- [ ] Integration plan discussed with team

---

## Questions?

If you encounter issues not covered here:

1. Check the error message carefully
2. Review Common Issues section above
3. Check Google Colab's documentation
4. Ask team members
5. Create an issue on GitHub repository

---

**Good luck with training! 🚀**
