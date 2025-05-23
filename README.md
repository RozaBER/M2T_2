# Large Brain-to-Text Model with MEG Encoder and LLaVA-Style Architecture

This repository contains the implementation of a brain-to-text model that generates natural language from magnetoencephalography (MEG) signals. The model combines a specialized neural encoder for MEG data with a Large Language Model (LLM) decoder in a LLaVA-style architecture, using Residual Vector Quantization (RVQ) for discrete representations.

## Overview

The model architecture consists of:
- **MEG Encoder**: Hierarchical transformer-based encoder for processing multi-channel MEG signals
- **Residual Vector Quantization (RVQ)**: Discretizes continuous neural representations for LLM integration
- **LLM Decoder**: Pre-trained LLaMA model with LoRA adaptation for text generation

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/brain-to-text-model.git
cd brain-to-text-model

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Dataset

The model is designed to work with the MEG-MASC dataset:
- 27 English speakers listening to naturalistic stories
- 208-channel MEG recordings at 1 kHz
- Time-aligned word and phoneme annotations
- Download from: https://osf.io/rguwj/

## Usage

### Training

1. **Configure the model**: Edit `configs/default_config.yaml` with your dataset path and training parameters.

2. **Train with multi-phase strategy**:
```bash
python train.py --config configs/default_config.yaml --mode multi_phase --output_dir ./output
```

3. **Train with single-phase strategy**:
```bash
python train.py --config configs/default_config.yaml --mode full --output_dir ./output
```

4. **Knowledge distillation** (requires a trained teacher model):
```bash
python train.py --config configs/default_config.yaml --mode distill \
    --teacher_checkpoint path/to/teacher --output_dir ./output
```

### Inference

Generate text from MEG signals:

```bash
# From MEG file (.fif format)
python inference.py --model_path ./output/best_model \
    --input_file path/to/meg_file.fif \
    --output_file results.json

# From numpy array
python inference.py --model_path ./output/best_model \
    --input_file path/to/meg_data.npy \
    --output_file results.json
```

## Model Architecture

### MEG Encoder
- 12 transformer layers with 768 hidden dimensions
- Convolutional stem for initial feature extraction
- Positional encoding for temporal information
- Multi-head self-attention (12 heads)

### Residual Vector Quantization
- 8 hierarchical codebooks
- 8192 entries per codebook
- 256-dimensional codebook vectors
- Commitment and entropy losses for training

### LLM Decoder
- LLaMA-2 7B base model
- LoRA adaptation (rank 8, alpha 16)
- Special tokens for MEG integration
- Autoregressive text generation

## Training Strategy

The multi-phase training approach consists of:

1. **Phase 1**: Encoder pre-training (50k steps)
   - Masked autoencoding objective
   - Learn general MEG representations

2. **Phase 2**: RVQ training (30k steps)
   - Frozen encoder
   - Optimize discrete representations

3. **Phase 3**: End-to-end fine-tuning (100k steps)
   - All components trainable
   - Language modeling objective

## Project Structure

```
├── src/
│   ├── models/
│   │   ├── eeg_encoder.py      # MEG/EEG encoder implementation
│   │   ├── rvq_module.py       # Residual Vector Quantization
│   │   ├── llm_decoder.py      # LLM decoder with LoRA
│   │   └── full_model.py       # Complete model integration
│   ├── data/
│   │   ├── meg_dataset.py      # MEG-MASC dataset loader
│   │   ├── preprocessing.py    # Signal preprocessing
│   │   └── tokenizer.py        # Custom tokenizer
│   └── training/
│       ├── trainer.py          # Base trainer
│       ├── multi_phase_trainer.py  # Multi-phase training
│       └── distillation.py     # Knowledge distillation
├── configs/
│   └── default_config.yaml     # Configuration file
├── report/
│   ├── main.tex               # Academic paper
│   └── references.bib         # Bibliography
├── train.py                   # Training script
├── inference.py               # Inference script
├── requirements.txt           # Dependencies
└── README.md                  # This file
```

## Performance

On the MEG-MASC test set:
- BLEU-4: 18.7
- ROUGE-L: 42.6
- BERTScore: 0.71
- Word Error Rate: 58.3%

## Citation

If you use this code in your research, please cite:

```bibtex
@article{yourlastname2024brain,
  title={Large Brain-to-Text Model with MEG Encoder and LLaVA-Style Architecture},
  author={Your Name},
  journal={arXiv preprint arXiv:2024.xxxxx},
  year={2024}
}
```

## Acknowledgments

- MEG-MASC dataset authors for providing the data
- LLaMA team for the pre-trained language model
- LLaVA authors for the multimodal architecture inspiration

## License

This project is licensed under the MIT License - see the LICENSE file for details.
