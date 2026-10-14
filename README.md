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
  - `observations/images/chest_rgb`: Shape `[T, 480, 640, 3]`, kiểu `uint8` (Chuẩn độ phân giải 640x480 hoặc 16:9 với Letterbox)
  - `observations/images/chest_depth`: Shape `[T, 480, 640]`, kiểu `uint16` (mm)
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
# Đẩy toàn bộ thư mục data_set lên server huấn luyện (thư mục quan_ws):
rsync -avzP path/to/local/data_set/ toann@VR-CUONGNM88-D:~/quan_ws/data_set/data_set/
# (Hoặc dùng địa chỉ IP: toann@192.168.8.19:~/quan_ws/data_set/data_set/)
```
*(Thay `path/to/local/data_set/` bằng đường dẫn thư mục chứa dataset trên máy bạn. Nếu bị rớt mạng giữa chừng, chỉ cần chạy lại lệnh, `rsync` sẽ tự động truyền tiếp các file còn dang dở).*

#### Cách 2: Dùng lệnh `scp` (Đơn giản, sẵn có trên Linux / macOS / Windows PowerShell)
```bash
# Đẩy toàn bộ các file .hdf5 lên server:
scp -r path/to/local/data_set/*.hdf5 toann@VR-CUONGNM88-D:~/quan_ws/data_set/data_set/
```

---

## 🚀 Quy trình Huấn luyện Mô hình trên Server (Server Training Workflow)

Dưới đây là toàn bộ quy trình chuẩn từng bước để huấn luyện mô hình ACT trên server `VR-CUONGNM88-D` (tài khoản `toann`, thư mục làm việc `~/quan_ws`), bắt đầu từ khâu đẩy mã nguồn từ máy cá nhân lên máy chủ:

---

### Bước 1: Đẩy mã nguồn từ máy cá nhân lên thư mục `quan_ws` trên Server

Bạn có thể lựa chọn 1 trong 2 cách sau tùy theo thói quen làm việc:

#### Cách 1: Đồng bộ trực tiếp siêu tốc qua `rsync` (Khuyên dùng — Nhanh nhất, không phụ thuộc Git commit)
Chạy lệnh này trên **Terminal của máy cá nhân (Local)** tại thư mục mã nguồn:
```bash
# Đồng bộ toàn bộ mã nguồn lên server vào thư mục ~/quan_ws (loại trừ venv, checkpoints, git):
rsync -avzP --exclude='.git' --exclude='venv' --exclude='checkpoints' --exclude='*.onnx*' \
    ./ toann@VR-CUONGNM88-D:~/quan_ws/

# (Hoặc thay VR-CUONGNM88-D bằng IP nếu chưa cấu hình hostname: toann@192.168.8.19:~/quan_ws/)
```

#### Cách 2: Đẩy mã nguồn qua Git (GitHub)
- **Trên máy cá nhân (Local):**
  ```bash
  git add .
  git commit -m "Update ACT pipeline: 640x480 resolution & Visual Auxiliary Losses"
  git push origin master    # Hoặc đẩy nhánh đang phát triển: git push origin quan1
  ```
- **Trên Server (`toann@VR-CUONGNM88-D`):**
  ```bash
  cd ~/quan_ws
  git fetch origin
  git pull origin master    # Hoặc chuyển nhánh: git checkout quan1 && git pull origin quan1
  ```

---

### Bước 2: Kết nối SSH & Khởi tạo phiên làm việc `tmux` trên Server

Tiến trình huấn luyện kéo dài nhiều giờ. Để tránh bị ngắt tiến trình khi đóng laptop, gập máy hoặc mạng SSH chập chờn, **bắt buộc chạy trong phiên `tmux`**:

```bash
# 1. Đăng nhập SSH vào Server:
ssh toann@VR-CUONGNM88-D    # Hoặc: ssh toann@192.168.8.19

# 2. Di chuyển vào thư mục làm việc:
cd ~/quan_ws

# 3. Tạo một phiên tmux mới có tên là 'act_train':
tmux new -s act_train

# (Nếu phiên act_train đã có sẵn từ trước, vào lại phiên bằng lệnh):
# tmux attach -t act_train
```

> **💡 Các phím tắt tmux cốt lõi cần nhớ:**
> - **Rời khỏi phiên (Detach)** để tiến trình tiếp tục chạy ngầm trên server: Nhấn tổ hợp phím **`Ctrl + B`**, sau đó thả tay ra và nhấn **`D`**.
> - **Vào lại phiên (Re-attach)** để theo dõi tiếp: `tmux attach -t act_train`
> - **Xem danh sách các phiên đang chạy:** `tmux ls`
> - **Đóng/Xóa phiên** (khi đã huấn luyện xong): `tmux kill-session -t act_train`

---

### Bước 3: Kích hoạt môi trường ảo Python & Kiểm tra tài nguyên

Bên trong phiên `tmux` trên server, chạy các lệnh sau:

```bash
# 1. Kích hoạt môi trường ảo Python:
source venv/bin/activate

# (Lưu ý: Nếu server mới tinh chưa tạo môi trường ảo, hãy chạy 1 lần):
# python3 -m venv venv
# source venv/bin/activate
# pip install --upgrade pip
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
# pip install -r requirements.txt

# 2. Kiểm tra nhận diện GPU NVIDIA và VRAM:
nvidia-smi

# 3. Kiểm tra tính toàn vẹn và tỷ lệ ảnh (640x480) của tập dữ liệu trước khi train:
python validate_dataset.py --dataset_dir data_set/data_set
```

---

### Bước 4: Lệnh chạy huấn luyện tiêu chuẩn trên Server (Recommended Training Command)

Chạy lệnh huấn luyện ACT tối ưu toàn diện trên GPU 0:
- Camera RGB-D chuẩn **640x480** (H=480, W=640, 300 visual tokens).
- Tự động lưu checkpoint PyTorch (`.pth`), xuất định kỳ mô hình ONNX (`.onnx`), ghi log và TensorBoard.
- Tự động dừng sớm (**Plateau Early Stopping**) nếu validation loss không cải thiện sau 10 lần đánh giá liên tiếp.

```bash
CUDA_VISIBLE_DEVICES=0 python -m act_pipeline.train \
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
    --img_height 480 \
    --img_width 640 \
    --use_tensorboard \
    --save_pth \
    --save_logs \
    --device cuda
```

> **💡 Gợi ý điều chỉnh theo cấu hình phần cứng:**
> - **Batch Size:** Nếu GPU có VRAM khiêm tốn ($\le 8\text{GB}$), giữ `--batch_size 16`. Nếu GPU có VRAM lớn ($16\text{GB} - 24\text{GB}$ như RTX 3090/4090/A5000), có thể tăng `--batch_size 32`.
> - **Rời khỏi terminal an toàn (Detach tmux):** Khi thấy log bắt đầu in Epoch 1, bạn nhấn **`Ctrl + B`** rồi bấm **`D`**. Quá trình huấn luyện vẫn tiếp tục chạy trên server ngay cả khi bạn tắt máy tính cá nhân.

---

### Bước 5: Giám sát quá trình huấn luyện (Monitoring)

#### Cách 1: Theo dõi file log thời gian thực
Mở một terminal mới kết nối vào server và chạy:
```bash
tail -f ~/quan_ws/checkpoints/act_openarm/train.log
```

#### Cách 2: Giám sát trực quan qua TensorBoard (SSH Tunnel)
1. **Khởi chạy TensorBoard trên Server** (chạy trong phiên tmux riêng `tmux new -s act_tb`):
   ```bash
   tensorboard --logdir ~/quan_ws/checkpoints/act_openarm/tensorboard --port 6007
   ```
2. **Mở đường hầm SSH từ máy tính cá nhân (Local):**
   Mở terminal trên máy tính của bạn và forward cổng 6007:
   ```bash
   ssh -N -L 6007:localhost:6007 toann@VR-CUONGNM88-D
   # (Hoặc: ssh -N -L 6007:localhost:6007 toann@192.168.8.19)
   ```
3. Mở trình duyệt web trên máy cá nhân truy cập: **`http://localhost:6007`** để xem biểu đồ loss, learning rate và MAE từng khớp.

---

### Bước 6: Xuất và Kiểm tra mô hình ONNX sau khi hoàn thành

Mặc định pipeline đã tự động lưu và xuất `best_checkpoint.onnx` khi đạt epoch có loss tốt nhất. Bạn cũng có thể xuất thủ công bất kỳ lúc nào:

```bash
cd ~/quan_ws
source venv/bin/activate

python export_onnx.py \
    --checkpoint checkpoints/act_openarm/best_checkpoint.pth \
    --output checkpoints/act_openarm/best_checkpoint.onnx
```

---

### 📋 Bảng các siêu tham số quan trọng

| Tham số | Mặc định | Ý nghĩa |
| :--- | :--- | :--- |
| `--dataset_dir` | `data_set/data_set` | Thư mục chứa các file `.hdf5` |
| `--output_dir` | `checkpoints/act_openarm` | Thư mục lưu checkpoint, model ONNX, metadata và log |
| `--epochs` | `100` | Tổng số epochs huấn luyện |
| `--batch_size` | `16` | Kích thước batch (GPU 6-8GB dùng 16, GPU >=16GB dùng 32) |
| `--chunk_size` | `50` | Độ dài cửa sổ hành động tương lai ($k = 50$ bước $\approx 1.0$ giây @ 50Hz) |
| `--lr` | `1e-4` | Learning rate cho Transformer Decoder và CVAE Encoder |
| `--lr_backbone` | `1e-5` | Learning rate cho ConvNeXt-Nano Backbone |
| `--kl_weight` | `10.0` | Hệ số phạt $\beta$ cho hàm mất mát KL Divergence |
| `--img_height` | `480` | Chiều cao ảnh RGB-D (Chuẩn 640x480) |
| `--img_width` | `640` | Chiều rộng ảnh RGB-D (Chuẩn 640x480) |
| `--val_split` | `0.15` | Tỷ lệ phân chia tập validation (15% số episode) |
| `--eval_every` | `10` | Chu kỳ đánh giá tập validation (mỗi 10 epochs) |
| `--save_every` | `10` | Chu kỳ kiểm tra và lưu checkpoint (mỗi 10 epochs) |
| `--save_pth` | `True` | Tự động lưu file checkpoint PyTorch `.pth` |
| `--save_logs` | `True` | Tự động ghi file text `train.log`, `metrics.jsonl` |
| `--use_tensorboard`| `True` | Tự động ghi log đồ thị TensorBoard vào `<output_dir>/tensorboard` |

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
# 1. Tải file mô hình ONNX tốt nhất:
scp toann@VR-CUONGNM88-D:~/quan_ws/checkpoints/act_openarm/best_checkpoint.onnx ./

# 2. Tải file metadata JSON đi kèm:
scp toann@VR-CUONGNM88-D:~/quan_ws/checkpoints/act_openarm/best_checkpoint_metadata.json ./

# 3. Tải file thống kê chuẩn hóa dataset_stats.pkl:
scp toann@VR-CUONGNM88-D:~/quan_ws/checkpoints/act_openarm/dataset_stats.pkl ./

# Hoặc tải nhanh toàn bộ file trong thư mục checkpoints về thư mục hiện tại:
scp toann@VR-CUONGNM88-D:~/quan_ws/checkpoints/act_openarm/* ./
```

*(Thay `username` và `<IP_SERVER>` bằng tài khoản và địa chỉ IP của server huấn luyện, ví dụ: `scp quan@192.168.1.100:...`)*

### Cách 2: Dùng lệnh `rsync` (Khuyên dùng: Tốc độ cao, có hiển thị % tiến trình tải)
```bash
# Tải đồng bộ toàn bộ thư mục checkpoints về thư mục local của robot:
rsync -avzP toann@VR-CUONGNM88-D:~/quan_ws/checkpoints/act_openarm/ ./act_robot_models/
# (Hoặc thay VR-CUONGNM88-D bằng IP: toann@192.168.8.19:~/quan_ws/checkpoints/act_openarm/)
```
