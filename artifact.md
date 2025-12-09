# Falcon Artifact Evaluation

This repository contains all the scripts and benchmarks needed to reproduce our results on:

1. **LLM-based code translation** (zero- and few-shot).  
2. **Backend performance profiling** across oneAPI, cuDNN, rocBLAS.  
3. **End-to-end throughput tests** with and without self-debugging / SMT.

---

## Prerequisites

- **Linux** (Ubuntu 20.04+ recommended)  
- **Python 3.10+**, with:
  - `openai`
  - `torch` (with CUDA、Hip、DL Boost support as needed)

- **Bash** shell (for the `.sh` scripts)  
- Credentials:
  ```bash
  export OPENAI_API_KEY="YOUR_OPENAI_KEY"
  ```

---

## 1. LLM-Based Cross-Platform Translation

We provide two modes of prompt engineering:

### 1.1 Zero-Shot

```bash
bash gpt4_zero_shot_test.sh
bash gpto1_zero_shot_test.sh
```

### 1.2 Few-Shot

```bash
bash gpt4_few_shot_test.sh
bash gpto1_few_shot_test.sh
```

Both scripts will:

1. Read source kernels from `benchmark/data/*_code_test/`
2. Call `benchmark/zero_shot/code_transfer_gpt4.py`  
3. Produce translated code under `translated/<src>_to_all/<dst>/`  
4. Invoke the appropriate compile-and-test driver  
   - CUDA → `benchmark/evaluation/cuda_test/compilation.py`  
   - HIP  → `benchmark/evaluation/hip_test/compilation.py`  
   - CPU  → `benchmark/evaluation/dlboost_test/compilation.py`  

---

## 2. Backend Performance Profiling

We measure kernel throughput on various vendor libraries:

```bash
python benchmark/perf/perf_oneAPI.py      # Intel oneAPI
python benchmark/perf/perf_cudnn.py       # NVIDIA cuDNN
python benchmark/perf/perf_rocblas.py     # AMD rocBLAS
```

Each script will load representative operators, run batched inputs, and print normalized throughput results.

---

## 3. End-to-End Performance Tests

These “falcon” scripts run full workloads of our system, with different configurations:

- **With self-debugging loop recovery**  
  ```bash
  bash falcon_with_self_debugging.sh
  ```
- **Without speculative multi-threading (SMT)**  
  ```bash
  bash falcon_without_smt.sh
  ```
- **Default end-to-end pipeline**  
  ```bash
  bash falcon.sh
  ```

Each will:

1. Translate a set of kernels via LLM  
2. Compile them on the target backend  
3. Run correctness/unit tests  
4. Report overall success rates and timings

---

## 4. Directory Layout

```
.
├── benchmark/
│   ├── data/                  # CUDA/HIP/CPU code samples
│   ├── evaluation/            # compile & test drivers per platform
│   ├── perf/                  # vendor-library profiling scripts
│   └── zero_shot/             # LLM translation with zero shot Python script
|   |—— few_shot/              # LLM translation with few shot Python script
|   |—— macro/                 # macro files for kernels
├── falcon*.sh                 # end-to-end pipelines
├── gpt4_zero_shot_test.sh     # gpt4 zero-shot translation driver
├── gpt4_few_shot_test.sh      # gpt4 few-shot translation driver
├── gpto1_zero_shot_test.sh    # gpto1 zero-shot translation driver
├── gpto1_few_shot_test.sh     # gpto1 few-shot translation driver
└── README.md                  # this file
```

---

## 5. Running the Full Evaluation

1. **Set your API key**:
   ```bash
   export OPENAI_API_KEY=…
   ```
2. **Install Python deps**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Run translation tests**:
   ```bash
   bash gpt4_zero_shot_test.sh
   bash gpt4_few_shot_test.sh
   ```
4. **Run vendor-profile benchmarks**:
   ```bash
   python benchmark/perf/perf_oneAPI.py
   python benchmark/perf/perf_cndnn.py
   python benchmark/perf/perf_rocblas.py
   ```
5. **Run end-to-end scripts**:
   ```bash
   bash falcon.sh
   bash falcon_with_self_debugging.sh
   bash falcon_without_smt.sh
   ```

All scripts print progress bars and summary statistics. Refer to each script’s header for platform-specific configuration flags if needed.

# Transcompiling
Our transcompilation times range from 1.2 to 7.8 hours. To better showcase Falcon’s capabilities, we provide the following examples:

**A quick start for transcompiling**. Take `add` operator from cpu to cuda  as an example, you can invoke Falcon like this:
```
python falcon/mcts/transcompile.py --source cpu --target cuda --file_name benchmark/data/cpp_code_test/add_1_15_64.cpp
```
During execution, you’ll see log messages such as:
```
Step: 12	Action: [8, 7]	Reward: 0.8930	Best Reward: 0.9435	Best action: [7]
```
Once complete, the transcompiled code will be available in the ``cpu_cuda`` directory.
You can then evaluate its performance by running:
```
python benchmark/perf/perf_cuda.py  --file_name ./cpu_cuda/add_1_15_64.cpp
```
This will output the performance result, for example:
```
Execution time:  0.0062 ms
```
**Comlplete evaluation**
For complete evaluation, please run the corresponding scripts. For example, to evaluate code from cpu to cuda shown in Figure 7, please use the following commands:
```
cd Figure7
python run.py
```
