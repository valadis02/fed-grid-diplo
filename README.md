# Byzantine Fault-Tolerant Federated Learning for Smart Energy Grids

Diploma thesis implementation — School of Electrical and Computer Engineering, NTUA.

This project implements a three-tier (Cloud–Fog–Edge) Federated Learning system for smart energy grids, evaluating Byzantine fault-tolerant aggregation algorithms against adversarial attacks.

---

## Requirements

### Hardware
- **PC** (Windows 10/11): runs Cloud, Edge nodes, and Docker Desktop
- **Raspberry Pi 4 or 5** (optional, for Chapter 7 & 8 experiments): runs the Fog aggregator
- Both devices must be on the **same local network**

### Software
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (latest version) — **must be installed and running before any Docker command**
- Python 3.10
- PowerShell 7+ (for running `.ps1` scripts)
- Git

> **Important:** Before running any experiment, make sure Docker Desktop is open and fully started. You should see the Docker whale icon in your system tray (Windows) with status "Docker Desktop is running". If you get a "cannot connect to Docker daemon" error, open Docker Desktop and wait for it to finish loading.

### Python Libraries

Each component has its own `requirements.txt`:

**Edge node** (`edge/requirements.txt`):
```
uvicorn==0.34.0
fastapi==0.115.6
pydantic==2.10.5
pika==1.3.2
tensorflow==2.20.0
pandas
scikit-learn
scipy
websockets==10.4.0
aio-pika==9.5.4
psutil==7.0.0
paho-mqtt
cryptography>=42.0.0
tensorflow-model-optimization
numpy
tenseal==0.3.16
```

**Fog node** (`fog/requirements.txt`):
```
uvicorn==0.34.0
fastapi==0.115.6
pydantic==2.10.5
pika==1.3.2
tensorflow==2.17.0
scipy==1.15.1
websockets==10.4.0
aio-pika==9.5.4
psutil==7.0.0
paho-mqtt
cryptography>=42.0.0
numpy
tenseal==0.3.15
pandas
```

**Cloud node** (`cloud/requirements.txt`):
```
uvicorn==0.34.0
fastapi==0.115.6
pydantic==2.10.5
pika==1.3.2
tensorflow==2.17.0
websockets==10.4.0
aio-pika==9.5.4
paho-mqtt
pandas==2.2.2
scikit-learn==1.5.2
cryptography>=42.0.0
tensorflow-model-optimization
tf-keras
numpy
tenseal==0.3.15
```

---

## Project Structure

```
fed-grid-diplo/
├── cloud/                      # Cloud node (global model evaluation)
├── edge/                       # Edge node (local training)
├── fog/                        # Fog node (BFT aggregation)
├── shared/                     # Shared utilities and model architectures
├── lcl_data/                   # London Smart Meter dataset (Chapter 8)
├── experiments/                # Docker Compose files per experiment
│   ├── ch7_exp1_fl_vs_centralized/
│   ├── ch7_exp2_node_scaling/
│   ├── ch7_exp3_hardware_limits/
│   ├── ch7_exp4_aes_encryption/
│   ├── ch7_exp4_ckks_encryption/
│   ├── ch7_exp5_pruning/
│   ├── ch8_byzantine_robustness/
│   └── ch8_scalability_rpi/
├── experiments_yml/            # Full Byzantine experiment YMLs (50 scenarios)
│   ├── pc/                     # PC-side YMLs (cloud + edges)
│   ├── pc_replication/         # Replication run YMLs
│   ├── rpi/                    # RPi-side YMLs (fog)
│   └── rpi_scalability/        # Scalability experiment YMLs
├── centralized_training.py     # Baseline centralized training (Chapter 7)
├── benchmark_aggregation.py    # Aggregation scalability benchmark
└── prepare_london_data.py      # LCL dataset preprocessing
```

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/valadis02/fed-grid-diplo.git
cd fed-grid-diplo
```

### 2. Create the Docker network

All containers communicate over a shared Docker network. Create it once:

```bash
docker network create aqtf_shared_network
```

### 3. Build the Docker images

Build the three images from the project root:

```bash
# Cloud node
docker build -f cloud/Dockerfile -t cloud_node_app:latest .

# Edge node
docker build -f edge/Dockerfile -t edge-node:latest .

# Fog node (PC)
docker build -f fog/Dockerfile -t fog_node_app:latest .

# Fog node (Raspberry Pi — run this on the RPi)
docker build -f fog/Dockerfile.rpi -t fog_node_app:arm64 .
```

### 4. Prepare the London Smart Meter dataset (Chapter 8 only)

Download the LCL dataset from [UK Data Service](https://beta.ukdataservice.ac.uk/datacatalogue/studies/study?id=7857) and place the raw CSV files in `lcl_data/`. Then run:

```bash
python prepare_london_data.py
```

This will preprocess the data into the format expected by the edge nodes.

### 5. Prepare the FLTrust root dataset (Chapter 8, FLTrust experiments only)

```bash
python create_fltrust_root_dataset.py
python create_server_dataset.py
```

---

## Running the Experiments

### Chapter 7 — FL vs Centralized, Hardware, Encryption, Pruning

Each experiment has its own folder under `experiments/`. Navigate to the relevant folder and run the corresponding YML files.

**Example — Experiment 1 (FL vs Centralized):**
```bash
docker compose -f experiments/ch7_exp1_fl_vs_centralized/experiment1.yml up
```

**Example — Experiment 3 (Hardware limits, requires RPi as Fog):**

On the PC:
```bash
docker compose -f experiments/ch7_exp3_hardware_limits/experiment10nodes_pc.yml up
```

On the RPi:
```bash
docker compose -f experiments/ch7_exp3_hardware_limits/experiment10nodes_rpi.yml up
```

**Example — Experiment 4.1 (AES Encryption, requires RPi as Fog):**

On the PC:
```bash
docker compose -f experiments/ch7_exp4_aes_encryption/exp4_aes_pc.yml up
```

On the RPi:
```bash
docker compose -f experiments/ch7_exp4_aes_encryption/exp4_aes_rpi.yml up
```

**Example — Experiment 4.2 (CKKS, set CKKS_POLY_MOD to 8192, 16384, or 32768):**

On the PC:
```bash
docker compose -f experiments/ch7_exp4_ckks_encryption/exp4_ckks_pc.yml up
```

On the RPi:
```bash
CKKS_POLY_MOD=8192 docker compose -f experiments/ch7_exp4_ckks_encryption/exp4_ckks_rpi.yml up
```

**Example — Experiment 5 (Pruning, set QUANTIZATION_MODE):**

On the PC:
```bash
QUANTIZATION_MODE=pruned70 docker compose -f experiments/ch7_exp5_pruning/experiment6.yml up
```

On the RPi:
```bash
PC_IP=<your_pc_ip> QUANTIZATION_MODE=pruned70 docker compose -f experiments/ch7_exp5_pruning/experiment6_edge.yml up
```

---

### Chapter 8 — Byzantine Robustness (50 scenarios × 5 algorithms × 3 attacks × 3 attacker ratios)

The Byzantine experiments are fully automated via PowerShell. The YMLs are pre-generated in `experiments_yml/`.

**Step 1** — Start the Fog on the RPi for each experiment. The script will prompt you when to do this.

**Step 2** — On the PC, run the orchestration script:

```powershell
.\experiments\ch8_byzantine_robustness\run_all_experiments.ps1
```

To resume from a specific experiment (e.g. experiment 15):
```powershell
.\experiments\ch8_byzantine_robustness\run_all_experiments.ps1 -StartFrom 15
```

Results are saved automatically to `results/replication_run/`.

---

### Chapter 8 — Scalability Benchmarks (RPi 4 & 5, 5–50 nodes)

**On the RPi**, run the scalability orchestration script:

```bash
bash experiments/ch8_scalability_rpi/run_scalability_benchmarks.sh
```

The script will prompt you to start the corresponding PC-side containers from `experiments_yml/pc_scalability/` for each topology.

Results are saved to `~/fed-grid-diplo/results/scalability/`.

---

## Dataset

The London Smart Meter dataset (LCL) used in Chapter 8 is provided by UK Power Networks under a [Creative Commons Attribution 4.0](https://creativecommons.org/licenses/by/4.0/) license. Please cite the original source if you use this data.

---

## Citation

If you use this work, please cite:

> Tsirindanis, C. (2026). *Byzantine Fault-Tolerant Federated Learning for Smart Energy Grids*. Diploma Thesis, School of Electrical and Computer Engineering, NTUA.
