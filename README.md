# 🤖 OpenArm ACT Training Pipeline (Action Chunking with Transformers)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Repository chuyên biệt dành riêng cho việc **huấn luyện mô hình AI ACT (Action Chunking with Transformers)** cho hệ thống robot 2 cánh tay **OpenArm (Bimanual 16-DOF)** tích hợp camera **RGB-D**.

---

## 📂 Cấu trúc Repository

```text
OpenArm_ACT_Training/
├── act_pipeline/                      # Core package mã nguồn huấn luyện ACT
│   ├── config.py                      # Cấu hình siêu tham số (ModelConfig, TrainConfig)
│   ├── train.py                       # Pipeline huấn luyện chính
│   ├── eval.py                        # Pipeline đánh giá & benchmark
│   ├── data/                          # Dataset loader & Tiền xử lý
│   │   ├── dataset.py                 # Episodic Dataset loader (HDF5 slicing, padding mask)
│   │   ├── preprocess.py              # Tiền xử lý ảnh RGB-D (ImageNet mean/std + Depth 0.2m-1.2m)
│   │   ├── normalization.py           # Z-score normalization (Mean & Std)
│   │   └── validate_dataset.py        # Kiểm tra tính toàn vẹn format HDF5
│   ├── models/                        # Kiến trúc mạng nơ-ron
│   │   ├── act_model.py               # ACT Policy tổng hợp
│   │   ├── cvae.py                    # CVAE Encoder (phân phối tiềm ẩn z)
│   │   ├── policy.py                  # Transformer Decoder (dự đoán 50 bước tương lai)
│   │   └── backbone.py                # ResNet-18 4 kênh RGB-D Feature Extractor
│   └── utils/                         # Tiện ích logging & checkpoints
│       ├── checkpoint.py              # Quản lý lưu/khôi phục weights
│       ├── logger.py                  # Logging đa kênh (Console, Logfile, TensorBoard, WandB)
│       ├── tb_visualizer.py           # Trực quan hóa MAE từng khớp & histogram trên TensorBoard
│       └── temporal_ensemble.py       # Bộ lọc ghép chuỗi hành động theo hàm mũ
├── train.py                           # [Entrypoint] Khởi chạy huấn luyện
├── eval.py                            # [Entrypoint] Đánh giá checkpoint
├── validate_dataset.py                # [Entrypoint] Kiểm tra dữ liệu demo trước khi train
├── requirements.txt                   # Danh sách thư viện Python cần thiết
├── pyproject.toml                     # Cấu hình cài đặt package
├── DATA_FORMAT_SPECIFICATION.md       # Tài liệu đặc tả chuẩn cấu trúc dữ liệu HDF5
└── README.md                          # Hướng dẫn chi tiết
```

---

## 🛠️ Cài đặt môi trường

### 1. Tạo môi trường ảo Python (Khuyên dùng Python 3.10)
```bash
# Tạo môi trường ảo
python3 -m venv venv
source venv/bin/activate

# Nâng cấp pip
pip install --upgrade pip
```

### 2. Cài đặt PyTorch với CUDA (Dành cho GPU NVIDIA)
Xem lệnh cài đặt tương thích với phiên bản CUDA của máy tại [pytorch.org](https://pytorch.org):
```bash
# Ví dụ với CUDA 11.8 hoặc CUDA 12.1:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 3. Cài đặt các thư viện bổ trợ
```bash
pip install -r requirements.txt

# (Tùy chọn) Cài đặt package ở chế độ editable
pip install -e .
```

---

## 📊 Chuẩn bị dữ liệu (Dataset Format)

Tập dữ liệu phải ở định dạng **HDF5 (`.hdf5`)** theo chuẩn mô tả trong [`DATA_FORMAT_SPECIFICATION.md`](DATA_FORMAT_SPECIFICATION.md):
- **Observations:**
  - `observations/images/chest_rgb`: Shape `[T, H, W, 3]`, kiểu `uint8`
  - `observations/images/chest_depth`: Shape `[T, H, W]`, kiểu `uint16` (mm) hoặc `float32` (mét)
  - `observations/qpos`: Shape `[T, 16]`, kiểu `float32` (14 khớp tay rad + 2 kẹp mét)
- **Actions:**
  - `action`: Shape `[T, 16]`, kiểu `float32`

### Kiểm tra dữ liệu trước khi train
```bash
python validate_dataset.py --dataset_dir path/to/dataset_folder
```

---

## 🚀 Huấn luyện mô hình (Training)

### 1. Lệnh huấn luyện cơ bản
```bash
python train.py --dataset_dir path/to/dataset --output_dir checkpoints/act_openarm
```

### 2. Chọn GPU cụ thể trên Server SSH
Nếu server có nhiều GPU (kiểm tra bằng `nvidia-smi`), dùng `CUDA_VISIBLE_DEVICES` để chỉ định GPU còn trống:

```bash
# Chạy trên GPU 0:
CUDA_VISIBLE_DEVICES=0 python train.py --dataset_dir path/to/dataset --output_dir checkpoints/act_openarm

# Chạy trên GPU 1:
CUDA_VISIBLE_DEVICES=1 python train.py --dataset_dir path/to/dataset --output_dir checkpoints/act_openarm
```

### 3. Các siêu tham số quan trọng

| Tham số | Mặc định | Ý nghĩa |
| :--- | :--- | :--- |
| `--dataset_dir` | `dataset/` | Thư mục chứa các file `.hdf5` |
| `--output_dir` | `checkpoints/act_openarm` | Thư mục lưu checkpoint và TensorBoard logs |
| `--epochs` | `500` | Tổng số epochs huấn luyện |
| `--batch_size` | `16` | Kích thước batch (GPU 16GB tối ưu ở 16, 24GB có thể dùng 32) |
| `--chunk_size` | `50` | Độ dài cửa sổ hành động tương lai ($k = 50$ bước $\approx 1.0$ giây @ 50Hz) |
| `--lr` | `1e-4` | Learning rate cho Transformer Decoder và CVAE Encoder |
| `--lr_backbone` | `1e-5` | Learning rate nhỏ cho Conv1 4-Channel Adapter |
| `--kl_weight` | `10.0` | Hệ số phạt $\beta$ cho hàm mất mát KL Divergence |
| `--val_split` | `0.15` | Tỷ lệ phân chia tập validation (15% số episode) |
| `--resume` | `None` | Khôi phục huấn luyện (`auto` để tìm checkpoint mới nhất) |
| `--no_amp` | `False` | Tắt Automatic Mixed Precision FP16 nếu GPU không hỗ trợ |

---

## 📈 Giám sát quá trình huấn luyện với TensorBoard

Mở tab terminal mới và chạy:
```bash
tensorboard --logdir checkpoints/act_openarm/tensorboard --port 6006
```
Truy cập trình duyệt: `http://localhost:6006`

**Các đồ thị chính theo dõi:**
- `loss/total_train` & `loss/total_val`: Tổng độ mất mát.
- `loss/recon_l1_val`: Sai số phục dựng hành động thực tế.
- `loss/kl_val`: Mức độ đa dạng của phân phối tiềm ẩn $z$.
- `per_joint_mae/val_joint_*`: Sai số chi tiết trên từng khớp trong 16 khớp robot.

---

## 🎯 Đánh giá mô hình (Evaluation)

Chạy benchmark trên tập dữ liệu kiểm thử (hỗ trợ cả checkpoint PyTorch `.pth` và ONNX `.onnx`):
```bash
# Đánh giá với file PyTorch checkpoint:
python eval.py \
  --checkpoint_path checkpoints/act_openarm/best_checkpoint.pth \
  --dataset_dir path/to/dataset \
  --output_dir eval_results/

# Hoặc đánh giá trực tiếp với file ONNX:
python eval.py \
  --checkpoint_path checkpoints/act_openarm/best_checkpoint.onnx \
  --dataset_dir path/to/dataset \
  --output_dir eval_results/
```

---

## 🤖 Xuất Checkpoint để Deploy sang Robot / Dashboard (ONNX & PyTorch)

Khi huấn luyện, pipeline **tự động lưu cả hai định dạng** PyTorch (`.pth`) và ONNX (`.onnx`) trong `--output_dir`:
1. `best_checkpoint.pth`: Checkpoint PyTorch tối ưu nhất (model + optimizer + scheduler + scaler) dùng để tiếp tục huấn luyện (`--resume`).
2. `best_checkpoint.onnx`: **Mô hình định dạng ONNX tối ưu nhất**, tối ưu hóa tính toán đồ thị suy luận (Inference Graph) cho edge computing, ROS2, C++, TensorRT hoặc ONNX Runtime.
3. `best_checkpoint_metadata.json`: Chứa đầy đủ thông số chuẩn hóa Z-score ($Mean$, $Std$) và cấu hình kiến trúc model đi kèm.
4. `act_deploy_model.onnx`: Bản sao ONNX sẵn sàng nạp trực tiếp vào chương trình điều khiển robot.
5. `act_deploy_weights.pth`: Trọng số PyTorch tinh gọn chỉ chứa model weights và normalization stats.
6. `dataset_stats.pkl`: File Pickle lưu thống kê $Mean$ và $Std$ của góc khớp và hành động.

### Xuất thủ công file checkpoint .pth sang .onnx bất kỳ lúc nào:
```bash
python export_onnx.py \
  --checkpoint checkpoints/act_openarm/best_checkpoint.pth \
  --output checkpoints/act_openarm/best_checkpoint.onnx \
  --opset 18
```

