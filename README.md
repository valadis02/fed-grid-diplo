# Byzantine Fault-Tolerant Federated Learning for Smart Energy Grids

Diploma thesis implementation — School of Electrical and Computer Engineering, NTUA.

This project implements a three-tier (Cloud–Fog–Edge) Federated Learning system for smart energy grids, evaluating Byzantine fault-tolerant aggregation algorithms against adversarial attacks.

---

## Requirements

### Hardware
- **PC** (Windows 10/11): runs Cloud, Fog, and Edge nodes via Docker Desktop
- **Raspberry Pi 4 or 5** (optional, for Chapter 7 & 8 scalability experiments): runs the Fog aggregator
- Both devices must be on the **same local network** (RPi setup only)

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
├── experiments_yml/            # Full Byzantine experiment YMLs
│   ├── pc/                     # PC-side YMLs (cloud + fog + edges, self-contained)
│   ├── pc_replication/         # Replication run YMLs
│   ├── pc_scalability/         # PC-side YMLs for scalability (cloud + edges)
│   ├── rpi/                    # RPi-side YMLs (fog only, requires RPi)
│   └── rpi_scalability/        # RPi-side YMLs for scalability (fog only)
├── centralized_training.py     # Baseline centralized training (Chapter 7)
├── benchmark_aggregation.py    # Aggregation scalability benchmark
└── prepare_london_data.py      # LCL dataset preprocessing
```

---

## Setup

### 1. Clone the repository

> **Important:** Clone into your home directory or another user-writable location. Do **not** clone into `C:\WINDOWS\system32` or any other system directory — Docker volume mounts will fail with "Access is denied".

**On the PC:**
```bash
git clone https://github.com/valadis02/fed-grid-diplo.git
cd fed-grid-diplo
```

**On the RPi** (for scalability experiments):
```bash
git clone https://github.com/valadis02/fed-grid-diplo.git
cd fed-grid-diplo
```

### 2. Install Docker on the RPi (scalability experiments only)

```bash
curl -sSL https://get.docker.com | sh
sudo usermod -aG docker $USER
```

Log out and back in, then verify:

```bash
docker --version
```

### 3. Create the Docker network

**On the PC:**
```bash
docker network create aqtf_shared_network
```

### 4. Build the Docker images

**On the PC**, build from the project root:

```bash
# Cloud node
docker build -f cloud/Dockerfile -t cloud_node_app:latest .

# Edge node
docker build -f edge/Dockerfile -t edge-node:latest .

# Fog node (PC)
docker build -f fog/Dockerfile -t fog_node_app:latest .
```

**On the RPi**, build the ARM64 fog image:

```bash
docker build --no-cache -f fog/Dockerfile.rpi -t fog_node_app:arm64 .
```

> **Note:** The RPi build takes 10–15 minutes. The Dockerfile.rpi uses `fog/requirements_nontenseal.txt` (no TensorSeal) and `edge/federated_topology.json`. If the build fails on first attempt due to network timeouts, run it again — Docker will resume from cache.

The RPi fog image also requires `rabbitmq:3-management` and `eclipse-mosquitto`. If the RPi cannot pull them due to network issues, transfer them from the PC:

```powershell
# On the PC
docker pull --platform linux/arm64 rabbitmq:3-management
docker save rabbitmq:3-management -o rabbitmq_arm64.tar
docker save eclipse-mosquitto -o mosquitto.tar
scp rabbitmq_arm64.tar <rpi_user>@<rpi_hostname>:~/
scp mosquitto.tar <rpi_user>@<rpi_hostname>:~/
```

```bash
# On the RPi
docker load -i ~/rabbitmq_arm64.tar
docker load -i ~/mosquitto.tar
```

### 5. Prepare the London Smart Meter dataset (Chapter 8 only)

Download the LCL dataset from [UK Data Service](https://beta.ukdataservice.ac.uk/datacatalogue/studies/study?id=7857) and place the raw CSV files in `lcl_data/`. Then run:

```bash
python prepare_london_data.py
```

### 6. Prepare the FLTrust root dataset (Chapter 8, FLTrust experiments only)

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

### Chapter 8 — Byzantine Robustness (50 scenarios)

The Byzantine experiments cover 5 aggregation algorithms × 3 attack types × 3 attacker ratios (10/20/30%). The `experiments_yml/pc/` YMLs are **fully self-contained**: each file includes Cloud, Fog, and all 10 Edge nodes, so no RPi is needed.

#### Experiment numbering

| ID | Algorithm | Attack | Attackers |
|----|-----------|--------|-----------|
| 01 | FedAvg | none | 0% |
| 02 | FedAvg | label flipping | 10% |
| 03 | FedAvg | sign flipping | 10% |
| 04 | FedAvg | gaussian noise | 10% |
| 05 | FedAvg | label flipping | 20% |
| 06 | FedAvg | sign flipping | 20% |
| 07 | FedAvg | gaussian noise | 20% |
| 08 | FedAvg | label flipping | 30% |
| 09 | FedAvg | sign flipping | 30% |
| 10 | FedAvg | gaussian noise | 30% |
| 11–20 | Krum | (same attack progression) | 10/20/30% |
| 21–30 | Multi-Krum | (same attack progression) | 10/20/30% |
| 31–40 | Trimmed Mean | (same attack progression) | 10/20/30% |
| 41–50 | FLTrust | (same attack progression) | 10/20/30% |

#### Run all 50 experiments (automated)

```powershell
.\experiments\ch8_byzantine_robustness\run_all_experiments.ps1
```

To resume from a specific experiment (e.g. experiment 15):

```powershell
.\experiments\ch8_byzantine_robustness\run_all_experiments.ps1 -StartFrom 15
```

Results (logs + metrics) are saved automatically to `results/replication_run/<exp_name>/`.

#### Run a single experiment manually

Each pc YML is self-contained. You can run any individual scenario directly:

```powershell
docker compose -f experiments_yml\pc\exp01_fedavg_no_attack_pc.yml up --abort-on-container-exit --exit-code-from cloud_app
```

When done, tear down:

```powershell
docker compose -f experiments_yml\pc\exp01_fedavg_no_attack_pc.yml down
```

#### Example 1 — FedAvg, no attack (baseline)

Runs FedAvg with 10 honest edge nodes and no attackers. Expected result: R² converges smoothly toward ~0.75–0.85 over 15 rounds.

```powershell
docker compose -f experiments_yml\pc\exp01_fedavg_no_attack_pc.yml up --abort-on-container-exit --exit-code-from cloud_app
```

Monitor the global model performance live:

```powershell
docker logs exp01_cloud_app -f
```

Look for lines like:
```
Cloud eval [Round 6]: MSE=0.0906 | MAE=0.2079 | R²=0.7579
```

#### Example 2 — FedAvg, Gaussian noise, 20% attackers

Runs FedAvg with 2 out of 10 edges sending Gaussian noise updates. Expected result: FedAvg collapses immediately — R² stays deeply negative throughout all 15 rounds, demonstrating its vulnerability to Gaussian noise.

```powershell
docker compose -f experiments_yml\pc\exp07_fedavg_ga_20pct_pc.yml up --abort-on-container-exit --exit-code-from cloud_app
```

Monitor:

```powershell
docker logs exp07_cloud_app -f
```

Expected output:
```
Cloud eval [Round 1]: MSE=43496532.0000 | MAE=5021.2725 | R²=-116257920.0000
```

Tear down between experiments:

```powershell
docker compose -f experiments_yml\pc\exp07_fedavg_ga_20pct_pc.yml down
```

#### Monitoring during any experiment

To watch cloud metrics in real time:

```powershell
docker logs <exp_id>_cloud_app -f
```

To check fog aggregation (confirms all 10 edges participated — look for `n_clients=10`):

```powershell
docker logs <exp_id>_fog_app | Select-String "AGG METRICS"
```

---

### Chapter 8 — Scalability Benchmarks (RPi 4 & 5, 5–20 nodes)

These experiments measure aggregation time, RAM usage, and CPU temperature of the Fog node running on a Raspberry Pi across 5 algorithms and 3 topologies (5, 10, 20 nodes). The Fog runs on the RPi; Cloud and Edge nodes run on the PC.

The same procedure works for both RPi 4 and RPi 5 — the only difference is performance.

#### Architecture

```
PC (Windows)                          RPi (Fog)
─────────────────────────────         ──────────────────────
Cloud node        ←── model ───────→  fog_app (arm64)
Edge nodes 1..N   ──── model ──────→  rabbitmq_fog
                                       mqtt_fog
```

#### Network requirements

Both devices must be on the same local network. Note the IPs of each:

```powershell
# On the PC
ipconfig | Select-String "IPv4"
```

```bash
# On the RPi
hostname -I
```

The `rpi_scalability` YMLs already have the correct IP hardcoded. If your PC IP differs from `192.168.1.5`, update all rpi YMLs:

```bash
# On the RPi
find experiments_yml/rpi_scalability/ -name "*.yml" | xargs sed -i 's/192\.168\.1\.5/<your_pc_ip>/g'
```

#### Run all scalability experiments (automated)

On the **RPi**, run the orchestration script:

```bash
bash experiments/ch8_scalability_rpi/run_scalability_benchmarks.sh
```

To start from a specific algorithm and topology:

```bash
bash experiments/ch8_scalability_rpi/run_scalability_benchmarks.sh --start-algo fltrust --start-topo 10
```

For each experiment the script will:
1. Start the Fog containers on the RPi automatically
2. Print the command to run on the PC
3. Wait for you to press ENTER after starting the edges on the PC
4. Collect 10 rounds of aggregation metrics
5. Save results to `~/fed-grid-diplo/results/scalability/metrics_<algo>_<n>.json`
6. Tear down and move to the next experiment

#### Example 1 — FedAvg, 10 nodes

**Step 1** — Start the script on the RPi (or wait for it to reach this experiment):

```bash
bash experiments/ch8_scalability_rpi/run_scalability_benchmarks.sh --start-algo fedavg --start-topo 10
```

**Step 2** — When the script prints the prompt, run this on the **PC**:

```powershell
docker compose -f experiments_yml\pc_scalability\scale_fedavg_10_pc.yml up
```

**Step 3** — Press ENTER on the RPi. The script collects metrics automatically.

**Step 4** — When done, tear down on the **PC**:

```powershell
docker compose -f experiments_yml\pc_scalability\scale_fedavg_10_pc.yml down
```

Press ENTER on the RPi to continue to the next experiment.

Monitor fog aggregation in real time (separate RPi terminal):

```bash
docker logs scale_fedavg_10_fog_app -f | grep "AGG METRICS"
```

Expected output:
```
[AGG METRICS] strategy=FEDAVG | n_clients=10 | agg_time=0.0009s | total_time=1.11s | peak_ram=612.3MB | cpu_temp_before=52.1C | cpu_temp_after=53.4C
```

#### Example 2 — FLTrust, 10 nodes

**Step 1** — Start from FLTrust 10:

```bash
bash experiments/ch8_scalability_rpi/run_scalability_benchmarks.sh --start-algo fltrust --start-topo 10
```

**Step 2** — On the **PC**:

```powershell
docker compose -f experiments_yml\pc_scalability\scale_fltrust_10_pc.yml up
```

**Step 3** — Press ENTER on the RPi.

**Step 4** — When done, on the **PC**:

```powershell
docker compose -f experiments_yml\pc_scalability\scale_fltrust_10_pc.yml down
```

Monitor:

```bash
docker logs scale_fltrust_10_fog_app -f | grep "AGG METRICS"
```

Expected output (FLTrust has higher agg_time due to cosine similarity computation):
```
[AGG METRICS] strategy=FLTRUST | n_clients=10 | agg_time=0.0820s | total_time=1.12s | peak_ram=631.5MB | cpu_temp_before=54.0C | cpu_temp_after=55.5C
```

#### View saved results

```bash
cat ~/fed-grid-diplo/results/scalability/metrics_fedavg_10.json
cat ~/fed-grid-diplo/results/scalability/metrics_fltrust_10.json
```

---

## Dataset

The London Smart Meter dataset (LCL) used in Chapter 8 is provided by UK Power Networks under a [Creative Commons Attribution 4.0](https://creativecommons.org/licenses/by/4.0/) license. Please cite the original source if you use this data.

---

## Citation

If you use this work, please cite:

> Tsirindanis, Ch. (2026). *Byzantine Fault-Tolerant Federated Learning for Smart Energy Grids*. Diploma Thesis, School of Electrical and Computer Engineering, NTUA.
