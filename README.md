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
│   │   └── backbone.py                # ConvNeXt-Nano (SOTA Pure-CNN) RGB-D Feature Extractor
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

## 📊 Chuẩn bị dữ liệu (Dataset Format & Đồng bộ 50Hz)

Tập dữ liệu ở định dạng **HDF5 (`.hdf5`)** theo chuẩn mô tả trong [`DATA_FORMAT_SPECIFICATION.md`](DATA_FORMAT_SPECIFICATION.md):
- **Observations:**
  - `observations/images/chest_rgb`: Shape `[T, 240, 424, 3]`, kiểu `uint8` (Chuẩn tỷ lệ 16:9 của camera RealSense D435/D405)
  - `observations/images/chest_depth`: Shape `[T, 240, 424]`, kiểu `uint16` (mm)
  - `observations/qpos`: Shape `[T, 16]`, kiểu `float32` (14 khớp tay rad + 2 kẹp)
- **Timestamps:**
  - `timestamp_ns`: Shape `[T]`, kiểu `int64` (thời điểm ghi nhận thực tế từng bước)
- **Actions:**
  - `action`: Shape `[T, 16]`, kiểu `float32`

### 1. Kiểm tra dữ liệu (Validate)
```bash
python validate_dataset.py data_set/data_set
```
Script sẽ tự động phát hiện nếu có hiện tượng rớt khung hình (Frame Drop), giật lag thời gian ($dt$ dao động) hoặc méo tỷ lệ ảnh (Aspect Ratio Distortion).

### 2. Đồng bộ hóa & Khắc phục rớt khung hình (Resample to 50Hz)
Nếu quá trình thu thập telemetry/camera bị nghẽn (tần số thực tế < 50Hz, ví dụ ~39.9Hz hoặc $dt$ giật từ 0.001s đến 0.149s), hãy chạy công cụ đồng bộ hóa để chuyển toàn bộ dataset về tần số chuẩn 50.0 Hz đồng đều:
```bash
python resample_dataset.py --input_dir data_set/data_set --output_dir data_set/data_set_50hz
```
*(Lưu ý: Ngay cả khi bạn chưa chạy resample offline, DataLoader trong pipeline huấn luyện cũng được tích hợp sẵn cơ chế **On-the-fly Resampling**, tự động phát hiện và nội suy quỹ đạo về đúng 50Hz ($dt=0.02s$) để chuỗi chunk 50 bước luôn tương đương đúng 1.0 giây vật lý!)*

### 3. Đẩy dữ liệu từ Máy thu thập / Robot lên Server Huấn luyện (Upload Dataset)
Chạy các lệnh dưới đây trên **Terminal của máy cá nhân hoặc máy tính gắn trên robot** (nơi lưu các file `.hdf5` vừa quay xong):

#### Cách 1: Dùng lệnh `rsync` (Khuyên dùng: Tốc độ cao, có hiển thị %, tự động truyền tiếp nếu đứt mạng)
```bash
# Đẩy toàn bộ thư mục data_set lên server huấn luyện:
rsync -avzP path/to/local/data_set/ toann@192.168.8.19:/home/quan/code/OpenArm_ACT_Training/data_set/data_set/
```
*(Thay `path/to/local/data_set/` bằng đường dẫn thư mục chứa dataset trên máy bạn. Nếu bị rớt mạng giữa chừng, chỉ cần chạy lại lệnh, `rsync` sẽ tự động truyền tiếp các file còn dang dở).*

#### Cách 2: Dùng lệnh `scp` (Đơn giản, sẵn có trên Linux / macOS / Windows PowerShell)
```bash
# Đẩy toàn bộ các file .hdf5 lên server:
scp -r path/to/local/data_set/*.hdf5 toann@192.168.8.19:/home/quan/code/OpenArm_ACT_Training/data_set/data_set/
```

---

## 🚀 Huấn luyện mô hình (Training)

### 0. Khởi tạo phiên làm việc với tmux & Thiết lập môi trường (Khuyên dùng khi chạy trên Server)
Khi huấn luyện trên máy chủ từ xa qua SSH, hãy sử dụng `tmux` để tiến trình huấn luyện liên tục chạy ngầm ổn định ngay cả khi ngắt kết nối SSH, gập laptop hoặc rớt mạng:

#### Bước 0.1: Tạo phiên tmux và môi trường ảo
```bash
# 1. Tạo một phiên tmux mới có tên là 'act_train'
tmux new -s act_train

# 2. Tạo môi trường ảo Python và kích hoạt:
python3 -m venv venv
source venv/bin/activate
```

*(Lưu ý: Nếu hệ thống báo lỗi thiếu gói venv, chạy: `sudo apt update && sudo apt install -y python3-venv python3-pip`)*

#### Bước 0.2: Cài đặt đầy đủ các thư viện cần thiết (Dependencies)
Chạy các lệnh sau bên trong môi trường `(venv)` để tải và cài đặt toàn bộ thư viện:

```bash
# 1. Nâng cấp công cụ cài đặt pip
pip install --upgrade pip

# 2. Cài đặt PyTorch có hỗ trợ GPU CUDA (Ví dụ CUDA 12.1):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 3. Cài đặt toàn bộ thư viện cần thiết (h5py, onnx, onnxruntime, scipy, tensorboard, opencv...)
pip install -r requirements.txt
```

> **💡 Các thao tác và phím tắt tmux cần nhớ:**
> - **Rời khỏi phiên (Detach)** để tiến trình chạy ngầm: Nhấn tổ hợp phím **`Ctrl + B`**, sau đó thả tay ra và nhấn **`D`**.
> - **Quay lại phiên (Re-attach)** để theo dõi log huấn luyện:
>   ```bash
>   tmux attach -t act_train
>   ```
> - **Xem danh sách các phiên đang chạy:**
>   ```bash
>   tmux ls
>   ```
> - **Đóng/Xóa phiên tmux** (khi huấn luyện xong):
>   ```bash
>   tmux kill-session -t act_train
>   # Hoặc gõ lệnh 'exit' trực tiếp bên trong phiên tmux
>   ```

---

### 1. Lệnh chạy huấn luyện tiêu chuẩn (Recommended Training Command)
Chạy huấn luyện mô hình ACT trên GPU 0 với cấu hình tối ưu 100 epochs, batch size 16 (hoặc 32 nếu VRAM lớn), tự động đánh giá và lưu checkpoint mỗi 10 epochs, đồng thời bật theo dõi qua **TensorBoard** và **Weights & Biases (WandB)**:

```bash
CUDA_VISIBLE_DEVICES=0 python3 -m act_pipeline.train \
    --dataset_dir data_set/data_set \
    --output_dir checkpoints/act_openarm \
    --epochs 100 \
    --batch_size 16 \
    --lr 1e-4 \
    --lr_backbone 1e-5 \
    --kl_weight 10.0 \
    --warmup_epochs 15 \
    --val_split 0.15 \
    --eval_every 10 \
    --save_every 10 \
    --use_tensorboard \
    --use_wandb \
    --wandb_project act-openarm \
    --device cuda
```

> **💡 Gợi ý:**
> - Nếu chỉ muốn giám sát cục bộ không dùng internet: Bỏ cờ `--use_wandb` và giữ `--use_tensorboard`.
> - Nếu giám sát từ xa không cần mạng LAN / không cắm cáp: Giữ `--use_wandb` (sau khi chạy `wandb login`).
> - Nếu GPU có VRAM khiêm tốn (<=6GB): Đặt `--batch_size 16`. Nếu GPU 16GB–24GB: Có thể tăng `--batch_size 32`.

### 2. Lệnh huấn luyện cơ bản hoặc chạy qua script entrypoint:
```bash
# Chạy trực tiếp qua train.py (với TensorBoard):
python train.py --dataset_dir data_set/data_set --output_dir checkpoints/act_openarm --use_tensorboard

# Chọn GPU cụ thể (ví dụ GPU 1):
CUDA_VISIBLE_DEVICES=1 python train.py --dataset_dir data_set/data_set --output_dir checkpoints/act_openarm --use_tensorboard
```

### 3. Các siêu tham số quan trọng

| Tham số | Mặc định | Ý nghĩa |
| :--- | :--- | :--- |
| `--dataset_dir` | `dataset/` | Thư mục chứa các file `.hdf5` |
| `--output_dir` | `checkpoints/act_openarm` | Thư mục lưu checkpoint ONNX và normalization stats |
| `--epochs` | `500` | Tổng số epochs huấn luyện |
| `--batch_size` | `16` | Kích thước batch (GPU 6GB dùng 16, GPU >=16GB có thể dùng 32) |
| `--chunk_size` | `50` | Độ dài cửa sổ hành động tương lai ($k = 50$ bước $\approx 1.0$ giây @ 50Hz) |
| `--lr` | `1e-4` | Learning rate cho Transformer Decoder và CVAE Encoder |
| `--lr_backbone` | `1e-5` | Learning rate nhỏ cho Conv1 4-Channel Adapter |
| `--kl_weight` | `10.0` | Hệ số phạt $\beta$ cho hàm mất mát KL Divergence |
| `--val_split` | `0.15` | Tỷ lệ phân chia tập validation (15% số episode) |
| `--eval_every` | `10` | Chu kỳ đánh giá tập validation (mặc định 10 epochs) |
| `--save_every` | `10` | Chu kỳ kiểm tra và lưu checkpoint (mặc định 10 epochs) |
| `--save_pth` | `False` | (Tùy chọn) Lưu thêm file checkpoint PyTorch `.pth` |
| `--save_logs` | `False` | (Tùy chọn) Lưu thêm file text `train.log`, `metrics.jsonl` |
| `--backbone_type` | `convnext_nano` | Kiến trúc mạng trích xuất thị giác (độc quyền ConvNeXt-Nano) |
| `--unfreeze_backbone` | `False` | (Tùy chọn) Mở khóa để fine-tune toàn bộ trọng số Backbone |
| `--use_tensorboard` | `False` | (Tùy chọn) Bật ghi nhật ký đồ thị TensorBoard vào `<output_dir>/tensorboard` |
| `--use_wandb` | `False` | (Tùy chọn) Bật đồng bộ theo dõi trực tuyến lên cloud Weights & Biases |
| `--wandb_project` | `act-openarm` | Tên dự án trên Weights & Biases |

---

## 📈 Giám sát quá trình huấn luyện

### Cách 1: Giám sát trực quan với TensorBoard (Mạng nội bộ / SSH Tunnel)

Khi bật cờ `--use_tensorboard`, hệ thống lưu log vào `checkpoints/act_openarm/tensorboard/`.

#### Bước 1: Khởi chạy TensorBoard trên server
Mở một terminal mới (hoặc trong phiên tmux `tmux new -s act_tb`):
```bash
tensorboard --logdir checkpoints/act_openarm/tensorboard --port 6007
```

#### Bước 2: Mở đường hầm SSH từ máy tính cá nhân
Mở terminal trên máy tính của bạn và forward cổng 6007:
```bash
ssh -N -L 6007:localhost:6007 toann@192.168.8.19
```
Sau đó mở trình duyệt truy cập: `http://localhost:6007`

### Cách 2: Giám sát từ xa qua Weights & Biases (Không cần cắm LAN / Wi-Fi)

Khi bật cờ `--use_wandb`, hệ thống tự động đẩy toàn bộ metric (loss, lr, per-joint MAE) lên cloud wandb.ai:

```bash
# 1. Đăng nhập 1 lần duy nhất trên server:
wandb login

# 2. Xem trực tiếp biểu đồ mọi lúc mọi nơi qua điện thoại/laptop tại:
# https://wandb.ai
```

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
scp toann@192.168.8.19:/home/quan/code/OpenArm_ACT_Training/checkpoints/act_openarm/best_checkpoint.onnx ./

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
