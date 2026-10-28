# Harmoning-Minds

Dual EEG interaction in music performance

# CVAE EEG Signal Generation Model

A Conditional VAE-based EEG band-power signal generation system, designed to simulate brainwave responses under different emotional segments for ensemble visualization and interaction.

## Project Structure

```
├── run.py                  # Training / inference entry
├── concat_performance.py    # Concatenate segments into full performance
├── timeline.txt            # Musical segment timeline configuration
├── cvae_eeg/               # Model package
│   ├── config.py           # Hyperparameters
│   ├── dataset.py          # Data loading
│   ├── model.py            # CVAE model
│   ├── train.py            # Training logic
│   └── inference.py        # Inference generation
├── data/                   # Raw collected data (CSV)
├── checkpoints/            # Trained model weights
└── generated_csv/          # Generated simulated signals
```

## Requirements

- Python 3.8+
- Dependencies: `pip install torch pandas numpy`

## Data Format

Raw data is CSV, one row per 0.5 second, containing 5 band-power values:

```csv
timestamp,time_s,Theta,Alpha,LowBeta,HighBeta,Gamma
```

Each segment requires 3 recording sessions, named `{segment}-1.csv`, `{segment}-2.csv`, `{segment}-3.csv`, placed in the `data/` directory.

## Usage

### 1. Preprocessing

Apply artifact removal (ICA filtering) to remove eye blinks, facial muscle artifacts, and other noise from the raw signal. Export cleaned data as standard CSV into the `data/` directory.

### 2. Train Models

Train a CVAE model for each musical segment using the collected signals:

```bash
python run.py --segment playful
python run.py --segment meditative
python run.py --segment intense
python run.py --segment lyrical
```

Trained weights are saved to `checkpoints/cvae_{segment}.pth`.

### 3. Generate Simulated Signals

Use trained models to generate simulated band-power signals of specified duration:

```bash
python run.py --infer --segment playful --duration 120
python run.py --infer --segment meditative --duration 120
python run.py --infer --segment intense --duration 120
python run.py --infer --segment lyrical --duration 120
```

Generated CSV files are saved in the `generated_csv/` directory.

### 4. Concatenate Full Performance

Based on the musical segment timeline annotated in `timeline.txt`, concatenate simulated segments into a complete EEG signal for the full performance:

```bash
python concat_performance.py --prep 10 --transition 5
```

- `--prep`: Opening preparation time in seconds
- `--transition`: Crossfade duration in seconds (split equally before and after each segment switch)

Output: `generated_csv/full_generated.csv`

## Configuration

### timeline.txt Format

```
# Start time (seconds),segment name
0,blank
16,meditative
48,lyrical
80,playful
...
END_TIME=422
```

Available segment names: `lyrical`, `playful`, `intense`, `meditative`, `blank` (rest/preparation).
