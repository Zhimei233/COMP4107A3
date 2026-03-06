# test_q2q3.py
from assignment3 import (
    UWaveGestureLibraryDataset,
    u_wave_gesture_library_cnn_model,
    u_wave_gesture_library_rnn_model
)
import torch

TRAIN = "UWaveGestureLibrary_TRAIN.csv"
TEST  = "UWaveGestureLibrary_TEST.csv"

# 测试 Q1
ds = UWaveGestureLibraryDataset(TRAIN)
x, y = ds[0]
assert x.shape == (3, 315), f"Wrong x shape: {x.shape}"
assert y.shape == (8,),     f"Wrong y shape: {y.shape}"
assert y.sum() == 1.0,      "y is not one-hot"
print(f"Q1 OK — {len(ds)} samples, x:{x.shape}, y:{y.shape}")

# 测试 Q2
cnn_model, cnn_train, cnn_val = u_wave_gesture_library_cnn_model(TRAIN)
assert isinstance(cnn_model, torch.nn.Module), "CNN is not nn.Module"
assert 0 <= cnn_train <= 1, "Invalid train accuracy"
assert 0 <= cnn_val   <= 1, "Invalid val accuracy"
# 测试模型能对测试集跑通
test_ds = UWaveGestureLibraryDataset(TEST)
x_test, _ = test_ds[0]
out = cnn_model(x_test.unsqueeze(0))
assert out.shape == (1, 8), f"Wrong output shape: {out.shape}"
print(f"Q2 OK — train={cnn_train:.4f}  val={cnn_val:.4f}")

# 测试 Q3
rnn_model, rnn_train, rnn_val = u_wave_gesture_library_rnn_model(TRAIN)
assert isinstance(rnn_model, torch.nn.Module), "RNN is not nn.Module"
assert 0 <= rnn_train <= 1, "Invalid train accuracy"
assert 0 <= rnn_val   <= 1, "Invalid val accuracy"
out = rnn_model(x_test.unsqueeze(0))
assert out.shape == (1, 8), f"Wrong output shape: {out.shape}"
print(f"Q3 OK — train={rnn_train:.4f}  val={rnn_val:.4f}")

print("\n所有检查通过！")