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

### 1. Lệnh chạy huấn luyện tiêu chuẩn (Recommended Training Command)
Chạy huấn luyện mô hình ACT trên GPU 0 với cấu hình tối ưu 100 epochs, batch size 32, đánh giá và lưu checkpoint mỗi 10 epochs:

```bash
CUDA_VISIBLE_DEVICES=0 python3 -m act_pipeline.train \
    --dataset_dir data_set/data_set \
    --output_dir checkpoints/act_openarm \
    --epochs 100 \
    --batch_size 32 \
    --lr 1e-4 \
    --lr_backbone 1e-5 \
    --kl_weight 10.0 \
    --warmup_epochs 15 \
    --val_split 0.15 \
    --eval_every 10 \
    --save_every 10 \
    --device cuda
```

### 2. Lệnh huấn luyện cơ bản hoặc chạy qua script entrypoint:
```bash
# Chạy trực tiếp qua train.py:
python train.py --dataset_dir data_set/data_set --output_dir checkpoints/act_openarm

# Chọn GPU cụ thể (ví dụ GPU 1):
CUDA_VISIBLE_DEVICES=1 python train.py --dataset_dir data_set/data_set --output_dir checkpoints/act_openarm
```

### 3. Các siêu tham số quan trọng

| Tham số | Mặc định | Ý nghĩa |
| :--- | :--- | :--- |
| `--dataset_dir` | `dataset/` | Thư mục chứa các file `.hdf5` |
| `--output_dir` | `checkpoints/act_openarm` | Thư mục lưu checkpoint ONNX và normalization stats |
| `--epochs` | `500` | Tổng số epochs huấn luyện |
| `--batch_size` | `16` | Kích thước batch (GPU 16GB tối ưu ở 16, 24GB có thể dùng 32) |
| `--chunk_size` | `50` | Độ dài cửa sổ hành động tương lai ($k = 50$ bước $\approx 1.0$ giây @ 50Hz) |
| `--lr` | `1e-4` | Learning rate cho Transformer Decoder và CVAE Encoder |
| `--lr_backbone` | `1e-5` | Learning rate nhỏ cho Conv1 4-Channel Adapter |
| `--kl_weight` | `10.0` | Hệ số phạt $\beta$ cho hàm mất mát KL Divergence |
| `--val_split` | `0.15` | Tỷ lệ phân chia tập validation (15% số episode) |
| `--eval_every` | `10` | Chu kỳ đánh giá tập validation (mặc định 10 epochs) |
| `--save_every` | `10` | Chu kỳ kiểm tra và lưu checkpoint (mặc định 10 epochs) |
| `--save_pth` | `False` | (Tùy chọn) Lưu thêm file checkpoint PyTorch `.pth` |
| `--save_logs` | `False` | (Tùy chọn) Lưu thêm file text `train.log`, `metrics.jsonl` |
| `--use_tensorboard` | `False` | (Tùy chọn) Bật ghi nhật ký đồ thị TensorBoard |

---

## 📈 Giám sát quá trình huấn luyện với TensorBoard

Nếu bạn muốn ghi nhật ký đồ thị TensorBoard, hãy thêm cờ `--use_tensorboard` khi chạy lệnh huấn luyện:
```bash
CUDA_VISIBLE_DEVICES=0 python3 -m act_pipeline.train \
    --dataset_dir data_set/data_set \
    --output_dir checkpoints/act_openarm \
    --epochs 100 \
    --batch_size 32 \
    --use_tensorboard \
    --device cuda
```

### 1. Khởi chạy TensorBoard:
Mở một terminal mới và chạy:
```bash
tensorboard --logdir checkpoints/act_openarm/tensorboard --port 6006
```
Truy cập trình duyệt tại: `http://localhost:6006`

### 2. Mở giao diện TensorBoard từ máy cá nhân (khi train trên Server SSH):
Chạy lệnh chuyển tiếp cổng (SSH Tunnel) trên terminal máy cá nhân của bạn:
```bash
# Cú pháp: ssh -N -L 6006:localhost:6006 <user>@<ip_server>
ssh -N -L 6006:localhost:6006 quan@192.168.1.100
```
Sau đó mở trình duyệt máy tính cá nhân và truy cập: `http://localhost:6006`

---

## 🎯 Đánh giá mô hình (Evaluation)

### 1. Lệnh chạy đánh giá tiêu chuẩn (Recommended Evaluation Command)
Chạy benchmark định lượng và kiểm tra sai số trên tập dữ liệu kiểm thử với GPU 0:

```bash
CUDA_VISIBLE_DEVICES=0 python3 -m act_pipeline.eval \
    --checkpoint_path checkpoints/act_openarm/best_checkpoint.onnx \
    --dataset_dir data_set/data_set \
    --output_dir evaluation_results \
    --device cuda
```

*(Hệ thống hỗ trợ nạp cả file `.onnx` hoặc tự động tìm `.onnx` nếu truyền đường dẫn `.pth`)*

---

## 🤖 Định dạng Checkpoint sau khi chạy (Chỉ lưu 3 file)

Khi hoàn tất huấn luyện, pipeline được thiết kế tinh gọn để **chỉ lưu đúng 3 file quan trọng nhất** trong `--output_dir`:
1. **`best_checkpoint.onnx`**: Mô hình đồ thị ONNX tốt nhất (Best Checkpoint) được tối ưu hóa cho suy luận thời gian thực trên robot thật (ROS2, C++, Python ONNX Runtime, TensorRT).
2. **`best_checkpoint_metadata.json`**: File JSON chứa đầy đủ thông số chuẩn hóa Z-score ($Mean$, $Std$ của 16 động cơ) và cấu hình model để phía điều khiển robot unnormalize góc khớp mà **không cần cài đặt PyTorch**.
3. **`dataset_stats.pkl`**: File Pickle lưu thống kê $Mean$ và $Std$ của góc khớp và hành động.

Chu kỳ kiểm tra và cập nhật checkpoint là **mỗi 10 epoch** (`--save_every 10`, `--eval_every 10`).

### Xuất thủ công file checkpoint .pth sang .onnx bất kỳ lúc nào:
```bash
python export_onnx.py \
  --checkpoint checkpoints/act_openarm/best_checkpoint.pth \
  --output checkpoints/act_openarm/best_checkpoint.onnx \
  --opset 18
```

---

## 📥 Lệnh tải Best Checkpoint từ Server về Máy cá nhân / Robot

Chạy các lệnh dưới đây trên **terminal của máy cá nhân hoặc máy tính gắn trên robot** để kéo model về:

### Cách 1: Dùng lệnh `scp` (Đơn giản, sẵn có trên Linux / macOS / Windows PowerShell)
```bash
# 1. Tải riêng file mô hình ONNX tốt nhất (khoảng ~108 MB):
scp username@<IP_SERVER>:/home/quan/code/OpenArm_ACT_Training/checkpoints/act_openarm/best_checkpoint.onnx ./

# 2. Tải file metadata JSON đi kèm:
scp username@<IP_SERVER>:/home/quan/code/OpenArm_ACT_Training/checkpoints/act_openarm/best_checkpoint_metadata.json ./

# 3. Tải file thống kê chuẩn hóa dataset_stats.pkl:
scp username@<IP_SERVER>:/home/quan/code/OpenArm_ACT_Training/checkpoints/act_openarm/dataset_stats.pkl ./

# Hoặc tải nhanh toàn bộ 3 file trong thư mục checkpoints về thư mục hiện tại:
scp username@<IP_SERVER>:/home/quan/code/OpenArm_ACT_Training/checkpoints/act_openarm/* ./
```

*(Thay `username` và `<IP_SERVER>` bằng tài khoản và địa chỉ IP của server huấn luyện, ví dụ: `scp quan@192.168.1.100:...`)*

### Cách 2: Dùng lệnh `rsync` (Khuyên dùng: Tốc độ cao, có hiển thị % tiến trình tải)
```bash
# Tải đồng bộ toàn bộ thư mục checkpoints về thư mục local của robot:
rsync -avzP username@<IP_SERVER>:/home/quan/code/OpenArm_ACT_Training/checkpoints/act_openarm/ ./act_robot_models/
```
