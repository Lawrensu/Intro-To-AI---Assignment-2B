# 🚀 Quick Start - RNN/LSTM Models

## For Collaborators: 3-Step Training Process

### Option 1: Google Colab (Recommended - Free GPU!)

```
1. Open: https://colab.research.google.com/
2. Upload: RNN_LSTM_Training_Colab.ipynb
3. Runtime → Change runtime type → T4 GPU
4. Run all cells (takes ~15-20 minutes)
5. Download trained models from last cell
```

**✅ Advantages:** Free GPU, no setup, fast training
**❌ Disadvantages:** 90-min timeout, requires re-upload

---

### Option 2: Local Training (Windows)

```powershell
# Setup (once)
cd C:\Project\AI\Intro-To-AI---Assignment-2B
.\venv\Scripts\Activate.ps1

# Train models
python src/train_lstm.py --model both --epochs 30

# Test models
python demo_rnn_lstm.py
```

**✅ Advantages:** No timeouts, save progress
**❌ Disadvantages:** Slower without GPU, uses local resources

---

## 📁 Files You Need

### Before Training:
- `src/models/rnn_model.py` - Model definitions
- `src/train_lstm.py` - Training script
- `RNN_LSTM_Training_Colab.ipynb` - Colab notebook

### After Training:
- `models/rnn_pattern_model.pth` ⬅️ Download this!
- `models/lstm_travel_time_model.pth` ⬅️ Download this!
- `models/rnn_pattern_training_history.png`
- `models/lstm_travel_time_training_history.png`

---

## ⚙️ Training Parameters

| Parameter | Default | Fast Training | Better Accuracy |
|-----------|---------|---------------|-----------------|
| n_samples | 2000 | 1000 | 3000 |
| epochs | 30 | 20 | 50 |
| batch_size | 32 | 64 | 32 |
| hidden_size | 64 | 32 | 128 |

### Example: Fast Training
```bash
python src/train_lstm.py --model both --n_samples 1000 --epochs 20
```

### Example: High Accuracy
```bash
python src/train_lstm.py --model both --n_samples 3000 --epochs 50 --hidden_size 128
```

---

## 📊 Expected Results

### Traffic Pattern RNN
- **Task:** Classify traffic (low/medium/high)
- **Expected Accuracy:** 80-90%
- **Training Time:** ~5-10 minutes (GPU)

### Travel Time LSTM
- **Task:** Predict journey time (minutes)
- **Expected MAE:** 8-12 minutes
- **Training Time:** ~5-10 minutes (GPU)

---

## 🔧 Common Issues

| Issue | Solution |
|-------|----------|
| "CUDA out of memory" | Reduce batch_size to 16 |
| "Module not found" | Run `%cd Intro-To-AI---Assignment-2B` |
| No GPU in Colab | Runtime → Change runtime type → GPU |
| Training too slow | Enable GPU or reduce epochs |
| Colab disconnects | Keep tab active, re-run if needed |

---

## ✅ Verification Checklist

After training, verify:
- [ ] Both .pth model files exist in `models/`
- [ ] Training plots show decreasing loss
- [ ] `python demo_rnn_lstm.py` runs successfully
- [ ] Models make reasonable predictions

---

## 🎯 Quick Test

```python
# Test if models work
python demo_rnn_lstm.py
```

You should see:
- ✅ Traffic pattern analysis (3 scenarios)
- ✅ Travel time predictions (3 scenarios)
- ✅ Route comparison with recommendations

---

## 📖 Detailed Documentation

- **Full Guide:** `COLAB_GUIDE.md`
- **Model Code:** `src/models/rnn_model.py`
- **Training Script:** `src/train_lstm.py`
- **Demo Script:** `demo_rnn_lstm.py`

---

## 🤝 Team Workflow

1. **Leader** trains models → shares .pth files via Drive
2. **Members** download models → place in `models/` folder
3. **Everyone** runs `demo_rnn_lstm.py` to verify
4. **Integrate** models with pathfinding algorithm

**Note:** Don't commit .pth files to Git (too large)

---

## 💡 Pro Tips

- **Colab GPU quota:** Limited per day, use wisely
- **Save checkpoints:** Models auto-save during training
- **Monitor training:** Watch loss/accuracy curves
- **Test quickly:** Use demo script before integrating
- **Share models:** Use Google Drive/Dropbox (not Git)

---

## 🆘 Need Help?

1. Check `COLAB_GUIDE.md` for detailed instructions
2. Review error messages carefully
3. Try reducing parameters (samples/epochs)
4. Ask in team chat
5. Check PyTorch documentation

---

## 📞 Support

- **Documentation:** See `COLAB_GUIDE.md`
- **Code Issues:** Check `src/models/rnn_model.py`
- **Training Issues:** Review `src/train_lstm.py`
- **Team Help:** Ask in group chat

---

**Ready to start? Open `COLAB_GUIDE.md` for full instructions!**
