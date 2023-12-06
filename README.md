# Gaussian-Flow

## Installation

1. Extensions

```bash
python -m pip install git+https://github.com/graphdeco-inria/diff-gaussian-rasterization.git
python -m pip install git+https://gitlab.inria.fr/bkerbl/simple-knn.git
```

2. Install taichi, the DDDM needs taichi to be installed.

```bash
python -m pip install taichi
```

3. Install the rest of the requirements

```bash
python -m pip install -r requirements.txt
```

## Quickstart

### Data (DNeRF dataset)

Download the data from Google drive [here](https://drive.google.com/file/d/19Na95wk0uikquivC7uKWVqllmTx-mBHt/view?usp=sharing) and extract it. The data should have the following structure:

```
├── data 
│   ├── mutant
│   ├── standup 
│   ├── ...
```

Run the script to train all the DNeRF scenes:

```bash
python scripts/run_dnerf_bench.py
```

>Note that remebmer to change the `data_root` in the script to the path of the extracted data.