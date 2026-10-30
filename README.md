# Harmonizing Minds

### A Brain-Driven Audiovisual Experience for Dual-Person EEG Interaction in Music Performance
#### CVAE EEG Signal Generation Model

> Two performers · One shared particle field · A synchronized mind-to-mind dialogue

| | | |
|---|---|---|
| ![Performer A — α/β/θ bands](docs/img/01a_performer_a.png) | ![Hero particle field](docs/img/01b_hero_particles.png) | ![Title block — Harmonizing Minds](docs/img/01c_title.png) |
| ![Performer B — α/β/θ bands](docs/img/01d_performer_b.png) | ![Synchrony — correlation / resonance / shared experience](docs/img/01e_synchrony.png) |  |

**Harmonizing Minds** is a real-time audiovisual performance system that translates the neural activity of two performers into a shared particle field. The technical core is a **Conditional VAE (CVAE)** that learns band-power distributions per musical mood (Lyrical / Playful / Intense / Meditative) and simulates a partner brainwave stream in real time.

- **MUSIC** · **PARTICLE** · **HUMAN**
- Project 01, 2026 · Li Ruichen, Hunan University

---

## 01 | Background

Traditional performance practice has three persistent gaps:

1. **Pre-written scores** lack real-time state feedback — performers cannot react to each other's moment-to-moment mental state.
2. **Preset stage visuals** are fixed and do not reflect what is actually happening on stage.
3. **Audiences infer emotion** only from body language and facial expression — the inner neural state remains hidden.

| | | |
|---|---|---|
| ![Orchestra / conductor / sheet-music collage](docs/img/02a_orchestra.png) | ![Brain–computer interaction — capturing hidden mental states](docs/img/02b_bci.png) | ![Digital performing art — dynamic imagery & teamwork](docs/img/02c_digital_art.png) |

### Project Structure

```
├── run.py                  # Training / inference entry
├── concat_performance.py   # Concatenate segments into full performance
├── timeline.txt            # Musical segment timeline configuration
├── cvae_eeg/               # Model package
│   ├── config.py           # Hyperparameters
│   ├── dataset.py          # Data loading
│   ├── model.py            # CVAE model
│   ├── train.py            # Training logic
│   └── inference.py         # Inference generation
├── data/                   # Raw collected data (CSV)
├── checkpoints/            # Trained model weights
└── generated_csv/          # Generated simulated signals
```

---

## 02 | Case Study

The system is built on established open-source toolchains and references prior EEG artworks.

| Layer | Tools / Works | Notes |
|---|---|---|
| Hardware | **OpenBCI** | Low-cost multimodal physiological acquisition; suitable for EEG-driven music, neurofeedback, and artistic prototyping |
| Software | **MNE-Python** | Import / filtering / artifact removal / time-frequency analysis / real-time streaming |
| Academic | FractalBrain (CHI-EA '24), BrainiBeats (CHI-EA '23) | Dual-person EEG-generated music; 5 mapped parameters: pitch, duration, interval, intensity, timbre |
| Artistic | *Melting Memories* (Refik Anadol), *Csound's Brain* (NIME '26), *Neurorecital* (Nicolas Namoradze) | Data-driven sculptural visuals; θ/α/β band power mapped to SATB harmony; live piano EEG visualization |

| | | |
|---|---|---|
| ![OpenBCI hardware platform](docs/img/03a_openbci.png) | ![MNE-Python analysis library](docs/img/03b_mne.png) | ![FractalBrain — academic project](docs/img/03c_fractalbrain.png) |
| ![Melting Memories — Refik Anadol](docs/img/03d_melting_memories.png) | ![BrainiBeats — synchronized dual EEG music](docs/img/03e_brainibeats.png) | ![Csound's Brain — EEG as instrument](docs/img/03f_csound.png) |
| ![Neurorecital — concert visualization](docs/img/03g_neurorecital.png) | | |

### Requirements

- Python 3.8+
- Dependencies:

```bash
pip install torch pandas numpy
```

---

## 01 | EEG Feature Layer

Five canonical brainwave bands are extracted from each performer's signal:

| Band | Frequency | Mental State |
|---|---|---|
| **δ** Delta | < 4 Hz | Deep sleep, physical recovery, unconscious states |
| **θ** Theta | 4 – 8 Hz | Drowsiness, meditation, memory processing, inward focus |
| **α** Alpha | 8 – 13 Hz | Relaxed wakefulness, calm attention (stronger with eyes closed) |
| **β** Beta | 13 – 30 Hz | Active thinking, focused attention, decision-making |
| **γ** Gamma | 30 Hz+ | High-level cognition, sensory integration, complex attention |

![EEG band feature cards — Delta / Theta / Alpha / Beta / Gamma](docs/img/04a_eeg_bands.png)

### Data Format

Raw data is CSV, one row per **0.5 second**, containing 5 band-power values:

```csv
timestamp,time_s,Theta,Alpha,LowBeta,HighBeta,Gamma
```

Each segment requires 3 recording sessions, named `{segment}-1.csv`, `{segment}-2.csv`, `{segment}-3.csv`, placed in the `data/` directory.

---

## 02 | Mapping Pipeline

```
Raw EEG  ──▶  Feature Extraction  ──▶  Visual Output
(2 channels)   (band power,           Motion · Coherence ·
               attention, emotion,    Complexity · Attraction
               mutual resonance)      ── control parameters ──▶ particle field
```

| | | |
|---|---|---|
| ![Raw EEG signal & head topology](docs/img/04b_raw_eeg.png) | ![Frequency-band power waveforms](docs/img/04c_band_power.png) | ![Visual output control parameters](docs/img/04d_visual_controls.png) |

### Preprocessing

Apply artifact removal (**ICA filtering**) to remove eye blinks, facial muscle artifacts, and other noise from the raw signal. Export cleaned data as standard CSV into the `data/` directory.

---

## 03 | Visualization

Three derived visual dimensions:

- **Thinking Intensity** — front-central β power → flow width / particle speed
- **Emotion Intensity** — (β + θ) / α → color brightness
- **Attention Immersion** — θ / β → flow curvature
- **Mutual Resonance** — `½(1 + Pearson r)` between the two performers → emitter spacing / shared-field interaction

| | | |
|---|---|---|
| ![Thinking intensity — low vs. high head maps](docs/img/04e_thinking_maps.png) | ![Emotion intensity — low vs. high head maps](docs/img/04f_emotion_maps.png) | ![Attention immersion — divergent vs. focused head maps](docs/img/04g_attention_maps.png) |

### Train Models

Train a CVAE model for each musical segment using the collected signals:

```bash
python run.py --segment playful
python run.py --segment meditative
python run.py --segment intense
python run.py --segment lyrical
```

Trained weights are saved to `checkpoints/cvae_{segment}.pth`.

---

## 04 | Prototype States

![Prototype states — low / high resonance grid](docs/img/04h_prototype_states.png)

![EEG feature → visual behavior formulas](docs/img/04i_formula_mapping.png)

### Generate Simulated Signals

Use trained models to generate simulated band-power signals of specified duration:

```bash
python run.py --infer --segment playful   --duration 120
python run.py --infer --segment meditative --duration 120
python run.py --infer --segment intense   --duration 120
python run.py --infer --segment lyrical  --duration 120
```

Generated CSV files are saved in the `generated_csv/` directory.

---

## 01 | From Neural Signals to Shared Experience (CVAE)

A Conditional Variational Autoencoder learns the joint distribution of EEG features conditioned on music mood. During performance it simulates the "partner" mind state in real time, enabling bidirectional exchange between a live performer and the simulated counterpart.

```
Input (EEG × mood)
   │
   ├── CNN / LSTM encoders per mood ──▶ latent (μ, σ, z) ── reparameterization ──▶
   │
   └──────────────── Decoder (LSTM + Concat) ──────────▶ Output (generated EEG stream)
```

| | |
|---|---|
| ![Autoencoder — 784-400-20-20-400-784](docs/img/05a_autoencoder.png) | ![CVAE architecture — mood-conditioned EEG generation](docs/img/05b_cvae.png) |

![Audience-area stage photo with live particle projection](docs/img/05c_stage_photo.png)

### Concatenate Full Performance

Based on the musical segment timeline annotated in `timeline.txt`, concatenate simulated segments into a complete EEG signal for the full performance:

```bash
python concat_performance.py --prep 10 --transition 5
```

- `--prep`: Opening preparation time in seconds
- `--transition`: Crossfade duration in seconds (split equally before and after each segment switch)

Output: `generated_csv/full_performance.csv`

---

## 02 | User Journey

| | |
|---|---|
| ![Performer side — real-time particle visualization](docs/img/07a_performer_side.png) | ![Audience side — visible particle emotions](docs/img/07b_audience_side.png) |
| ![Training AI — CVAE model on live EEG](docs/img/07c_training_ai.png) | ![On stage — bidirectional mind-to-mind improvisation](docs/img/07d_on_stage.png) |

1. **Training AI** — Acquire EEG data from live performances to train a CVAE model that simulates a human performer's mental state in real time.
2. **Performer Side** — A real-time particle display lets performers see each other's mental state and adjust their playing to achieve mutual neural rapport.
3. **On Stage** — Live EEG and simulated brainwave exchange bidirectionally, driving spontaneous, mind-to-mind improvisation.
4. **Audience Side** — Hidden performer emotions are externalized into visible particle motion, giving the audience an immersive and empathetic window into the performance.

### Configuration — `timeline.txt` Format

```
# Start time (seconds),segment name
0,blank
16,meditative
48,lyrical
80,playful
...
END_TIME=422
```

Available segment names: `prelude`, `lyrical`, `playful`, `intense`, `meditative`, `blank` (rest / preparation).

---

## 01 | Between Two Minds

Two EEG streams (Performer A = magenta, Performer B = cyan) feed a shared particle field. Relative metrics — activation, stability, state difference `D`, and similarity `1 − D` — are exponentially smoothed (τ ≈ 0.45 s) and drive emitter speed, curl noise, emitter separation, and shared-field interaction.

| | | |
|---|---|---|
| ![State bar charts — Feeling Explosion / Concentrated Resonance / Emotional Dislocation / Synchronized Meditation](docs/img/06a_state_bars.png) | ![Similarity line chart over 0–30 s](docs/img/06b_similarity_curve.png) | ![Main live particle field between two minds](docs/img/06c_main_particles.png) |

![Mapping diagram — simulated streams → relational metrics → shared particle field](docs/img/06d_mapping_diagram.png)

Four characteristic dramatic states emerge:

| State | α / β / θ pattern | Visual behavior |
|---|---|---|
| **Feeling Explosion** | Sudden high β on one side | Particles burst outward |
| **Concentrated Resonance** | Both sides α-dominant, high similarity | Monochrome convergence |
| **Emotional Dislocation** | Opposite band power trends | Dual-color separation |
| **Synchronized Meditation** | Both sides θ-dominant, `r ≈ 1` | Multicolor interweaving |

---

## 02 | Visual State Contrast

![Visual state contrast — Monochrome / Dual-Color / Multicolor / Open Dispersion](docs/img/06e_state_contrast.png)

Higher-order states progress from **Monochrome Convergence → Dual-Color Convergence → Multicolor Interweaving → Open Dispersion**, with particles alternating between *Approach* and *Compact / Dispersed* configurations.

![Approach / Compact / Dispersed particle configurations](docs/img/06f_approach_compact.png)

---

## Tech Stack

- **EEG acquisition:** OpenBCI (dual headset setup)
- **Signal processing:** MNE-Python (filtering, artifact removal / ICA, band-power, coherence)
- **Model:** PyTorch CVAE (CNN + LSTM encoder / decoder, reparameterization trick)
- **Real-time bridge:** Lab Streaming Layer (LSL) for EEG → rendering
- **Visualization:** custom GLSL / particle system (WebGL or TouchDesigner-style emitter field)
- **Audio:** Csound / SATB mapping for EEG-derived harmony

---

## Author

**Li Ruichen** — Hunan University
Project 01 · 2026

*Brain · Sound · Particle · Human*

---

## Image Index (for `docs/img/`)

| Placeholder | Source slide | Section |
|---|---|---|
| `01a_performer_a.png` | Slide 1 (left circle) | Hero |
| `01b_hero_particles.png` | Slide 1 (center) | Hero |
| `01c_title.png` | Slide 1 (right) | Hero |
| `01d_performer_b.png` | Slide 1 (bottom-left) | Hero |
| `01e_synchrony.png` | Slide 1 (bottom-right) | Hero |
| `02a_orchestra.png` | Slide 2 | 01 Background |
| `02b_bci.png` | Slide 2 | 01 Background |
| `02c_digital_art.png` | Slide 2 | 01 Background |
| `03a_openbci.png` | Slide 2 | 02 Case Study |
| `03b_mne.png` | Slide 2 | 02 Case Study |
| `03c_fractalbrain.png` | Slide 2 | 02 Case Study |
| `03d_melting_memories.png` | Slide 2 | 02 Case Study |
| `03e_brainibeats.png` | Slide 2 | 02 Case Study |
| `03f_csound.png` | Slide 2 | 02 Case Study |
| `03g_neurorecital.png` | Slide 2 | 02 Case Study |
| `04a_eeg_bands.png` | Slide 3 | 01 EEG Feature Layer |
| `04b_raw_eeg.png` | Slide 3 | 02 Mapping Pipeline |
| `04c_band_power.png` | Slide 3 | 02 Mapping Pipeline |
| `04d_visual_controls.png` | Slide 3 | 02 Mapping Pipeline |
| `04e_thinking_maps.png` | Slide 3 | 03 Visualization |
| `04f_emotion_maps.png` | Slide 3 | 03 Visualization |
| `04g_attention_maps.png` | Slide 3 | 03 Visualization |
| `04h_prototype_states.png` | Slide 3 | 04 Prototype States |
| `04i_formula_mapping.png` | Slide 3 | 04 Prototype States |
| `05a_autoencoder.png` | Slide 4 | 01 CVAE |
| `05b_cvae.png` | Slide 4 | 01 CVAE |
| `05c_stage_photo.png` | Slide 4 | 01 CVAE |
| `07a_performer_side.png` | Slide 4 | 02 User Journey |
| `07b_audience_side.png` | Slide 4 | 02 User Journey |
| `07c_training_ai.png` | Slide 4 | 02 User Journey |
| `07d_on_stage.png` | Slide 4 | 02 User Journey |
| `06a_state_bars.png` | Slide 5 | 01 Between Two Minds |
| `06b_similarity_curve.png` | Slide 5 | 01 Between Two Minds |
| `06c_main_particles.png` | Slide 5 | 01 Between Two Minds |
| `06d_mapping_diagram.png` | Slide 5 | 01 Between Two Minds |
| `06e_state_contrast.png` | Slide 5 | 02 Visual State Contrast |
| `06f_approach_compact.png` | Slide 5 | 02 Visual State Contrast |
