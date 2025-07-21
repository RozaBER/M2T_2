# Brain-to-Text Model Evaluation Report

**Model Checkpoint**: ./output/final_model
**Evaluation Date**: 2025-07-04 11:21:22

## Encoder Performance

- Average Reconstruction Error: 0.0414
- Average VQ Loss: 0.0570

### Codebook Utilization

- codebook_0: 1/8192 (0.0%)
- codebook_1: 1/8192 (0.0%)
- codebook_2: 1/8192 (0.0%)
- codebook_3: 2/8192 (0.0%)
- codebook_4: 2/8192 (0.0%)
- codebook_5: 1/8192 (0.0%)
- codebook_6: 2/8192 (0.0%)
- codebook_7: 2/8192 (0.0%)

## Generation Quality

- Samples Evaluated: 8
- Average Generation Loss: 5.9487

### ROUGE Scores

- rouge-1: 0.1469
- rouge-2: 0.1195
- rouge-l: 0.1469
- rouge-1-std: 0.1914
- rouge-2-std: 0.1571
- rouge-l-std: 0.1914

### BLEU Scores

- bleu-1: 0.3077
- bleu-2: 0.2500
- bleu-3: 0.1818
- bleu-4: 0.1934

### Example Generations

**Example 1**:
- Prediction:  Initial Lolovsky0000000000000000 Lolovsky0000000000000000 Lolovsky0000000000000000 Lolovsky0000000000000000 Lolovsky0000000000000000 Lolovsky0000000000000000 Emmanuel word Fort by Bill Glover
- Reference: The Black Willow Allan sat down

**Example 2**:
- Prediction:  Initial Lolovsky0000000000000000 Lolovsky0000000000000000 Lolovsky0000000000000000 Lolovsky0000000000000000 Lolovsky0000000000000000 Lolovsky0000000000000000 Emmanuel word Fort by Bill Glover
- Reference: The Cable Fort by Bill Glover

**Example 3**:
- Prediction:  Initial Lolovsky0000000000000000 Lolovsky0000000000000000 Lolovsky0000000000000000 Lolovsky0000000000000000 Lolovsky0000000000000000 Lolovsky0000000000000000 Emmanuel word Fort by Bill Glover
- Reference: The Black Willow Allan sat down

**Example 4**:
- Prediction:  Initial Lolovsky0000000000000000 Lolovsky0000000000000000 Lolovsky0000000000000000 Lolovsky0000000000000000 Lolovsky0000000000000000 Lolovsky0000000000000000 Emmanuel word Fort by Bill Glover
- Reference: The Black Willow Allan sat down

**Example 5**:
- Prediction:  Initial Lolovsky0000000000000000 Lolovsky0000000000000000 Lolovsky0000000000000000 Lolovsky0000000000000000 Lolovsky0000000000000000 Lolovsky0000000000000000 Emmanuel word Fort by Bill Glover
- Reference: The Cable Fort by Bill Glover

