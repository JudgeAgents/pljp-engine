# Precedent-Enhanced Legal Judgment Prediction with LLM and Domain-Model Collaboration

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the implementation of "Precedent-Enhanced Legal Judgment Prediction with LLM and Domain-Model Collaboration", published at EMNLP 2023.

## 📋 Table of Contents

- [Overview](#overview)
- [Methodology](#methodology)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Dataset](#dataset)
- [Usage](#usage)
- [Evaluation](#evaluation)
- [Citation](#citation)
- [License](#license)

## 🔍 Overview

This work presents a novel framework for Legal Judgment Prediction (LJP) that combines Large Language Models (LLMs) with domain-specific models through in-context learning. The system leverages precedent retrieval and fact reorganization to enhance prediction accuracy for legal articles, charges, and penalties.

### Key Features

- **Hybrid Architecture**: Combines LLMs with domain models for improved accuracy
- **Precedent Retrieval**: Uses similarity-based retrieval to find relevant precedents
- **Fact Reorganization**: Splits facts into subjective motivation, objective behavior, and post-facto circumstances
- **Multi-task Support**: Handles article prediction, charge classification, and penalty estimation
- **Few-shot Learning**: Utilizes in-context examples for better performance

## 🎯 Methodology

Our approach consists of three main stages:

1. **Fact Reorganization**: Facts are reorganized by LLMs into three components:
   - **Subjective Motivation (主观动机)**: The defendant's mental state and intent
   - **Objective Behavior (客观行为)**: The actual criminal actions taken
   - **Post-facto Circumstances (事外情节)**: Mitigating or aggravating factors

2. **Domain Model Prediction**: Domain-specific models (CNN, TopJudge, or ELE) generate:
   - Top-k candidate labels (articles, charges, or penalties)
   - Similar precedents based on reorganized facts

3. **LLM Inference**: The LLM makes final predictions using:
   - Reorganized fact of the current case
   - Top-k candidate labels from domain models
   - Retrieved precedents for in-context learning
   - Few-shot examples with labels

### Workflow Example

```
Input: Fact description of a criminal case
  ↓
[Step 1] LLM reorganizes fact into three components
  ↓
[Step 2] Domain models generate:
  - Candidate labels: "Article 263, Article 264, ..."
  - Precedents: Similar cases from database
  ↓
[Step 3] LLM predicts final judgment:
  - Article: "Article 263"
  - Charge: "Robbery"
  - Penalty: "24 to 36 months"
```

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- OpenAI API key (for LLM inference)

### Setup

1. Clone this repository:
```bash
git clone https://github.com/yourusername/PLJP.git
cd PLJP
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure API key:
```bash
export OPENAI_API_KEY="your-api-key-here"
```

Or set it in `config.py`:
  ```python
Config.OPENAI_API_KEY = "your-api-key-here"
```

## 📊 Dataset

### Datasets

We support two datasets:

1. **CAIL2018**: Large-scale Chinese legal judgment dataset
   - Download: [CAIL2018 GitHub](https://github.com/china-ai-law-challenge/CAIL2018)
   
2. **CJO22**: Judicial opinions dataset with 1,698 samples

### Data Format

Each case in the dataset follows this structure:

```json
{
  "caseID": "unique_case_id",
  "fact": "Raw fact description of the case",
  "fact_split": {
    "zhuguan": "Subjective motivation",
    "keguan": "Objective behavior",
    "shiwai": "Post-facto circumstances"
  },
  "meta": {
    "relevant_articles": [263, 264],
    "accusation": ["Robbery"],
    "term_of_imprisonment": {
      "imprisonment": 36,
      "life_imprisonment": false,
      "death_penalty": false
    },
    "punish_of_money": 5000,
    "criminals": ["Defendant A"]
  }
}
```

### Data Download

Preprocessed data can be downloaded from:
- [Baidu Pan](https://pan.baidu.com/s/1MrJdxvwTOfwhOwANJpTLtQ) (Extraction code: vu76)

Extract the data to the `data/` directory.

## 💻 Usage

### Basic Usage

Run inference for a specific task:

```bash
python llm_api.py \
    --model "gpt-3.5-turbo" \
    --dataset "cail18" \
    --small_model "CNN" \
    --task "charge" \
    --shot 3 \
    --use_split_fact
```

### Parameters

#### Model Selection
- `--model`: LLM model to use
  - Options: `text-davinci-002`, `text-davinci-003`, `gpt-3.5-turbo`
  - Default: `gpt-3.5-turbo`

#### Dataset Configuration
- `--dataset`: Dataset name
  - Options: `cail18`, `cjo22`
  - Default: `cail18`

- `--small_model`: Domain model to use
  - Options: `CNN`, `TopJudge`, `ELE`
  - Default: `CNN`

#### Task Configuration
- `--task`: Prediction task
  - Options: `article`, `charge`, `penalty`
  - Default: `charge`

- `--shot`: Number of few-shot examples
  - Default: `3`

- `--use_split_fact`: Use reorganized fact (recommended)
  - Flag: Include this flag to use fact split

#### Advanced Options
- `--batch`: Batch size for API calls (default: 1)
- `--retriever`: Precedent retrieval method
  - Options: `bm25`, `dense_retrieval`
  - Default: `dense_retrieval`

#### Custom Paths
- `--input_path`: Custom test set path
- `--output_path`: Custom output path
- `--precedent_pool_path`: Custom precedent database path
- `--precedent_idx_path`: Custom precedent index path
- `--topk_label_option_path`: Custom top-k labels path

### Example Commands

**Article Prediction:**
```bash
python llm_api.py \
    --model "gpt-3.5-turbo" \
    --dataset "cail18" \
    --small_model "CNN" \
    --task "article" \
    --shot 3 \
    --use_split_fact
```

**Charge Classification:**
```bash
python llm_api.py \
    --model "gpt-3.5-turbo" \
    --dataset "cail18" \
    --small_model "CNN" \
    --task "charge" \
    --shot 3 \
    --use_split_fact
```

**Penalty Estimation:**
```bash
python llm_api.py \
    --model "gpt-3.5-turbo" \
    --dataset "cail18" \
    --small_model "CNN" \
    --task "penalty" \
    --shot 3 \
    --use_split_fact
```

**Note**: For `charge` and `penalty` tasks, ensure that `article` predictions exist at the expected path, as they are used as input features.

## 📈 Evaluation

Evaluate predictions using the provided evaluation scripts:

### Article Evaluation

```bash
python -m src.metrics.evaluation_article \
    --testset_path "data/testset/cail18/testset.json" \
    --result_path "data/output/llm_out/cail18/CNN/article/3shot/gpt-3.5-turbo.json"
```

### Charge Evaluation

```bash
python -m src.metrics.evaluation_charge \
    --testset_path "data/testset/cail18/testset.json" \
    --result_path "data/output/llm_out/cail18/CNN/charge/3shot/gpt-3.5-turbo.json"
```

### Penalty Evaluation

```bash
python -m src.metrics.evaluation_penalty \
    --testset_path "data/testset/cail18/testset.json" \
    --result_path "data/output/llm_out/cail18/CNN/penalty/3shot/gpt-3.5-turbo.json"
```

### Metrics

The evaluation scripts output:
- **Accuracy**: Micro-averaged accuracy
- **Macro Precision**: Macro-averaged precision
- **Macro Recall**: Macro-averaged recall
- **Macro F1**: Macro-averaged F1 score

## 📁 Project Structure

```
PLJP/
├── config.py                 # Configuration settings
├── llm_api.py                # Main inference script
├── requirements.txt         # Python dependencies
├── README.md                # This file
├── data/                    # Data directory
│   ├── testset/            # Test sets
│   ├── precedent_database/ # Precedent cases
│   └── output/             # Output files
│       ├── domain_model_out/ # Domain model predictions
│       └── llm_out/        # LLM predictions
└── mycode/                  # Source code
    ├── utils/              # Utility functions
    │   ├── loader.py       # Data loading utilities
    │   └── prompt_gen.py   # Prompt generation
    └── metrics/            # Evaluation metrics
        ├── evaluator.py    # Base evaluator
        ├── evaluation_article.py
        ├── evaluation_charge.py
        └── evaluation_penalty.py
```

## 🔧 Configuration

Modify `config.py` to customize:
- API keys and endpoints
- Token limits
- Text processing parameters
- Default model settings

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 Citation

If you use this code in your research, please cite our paper:

```bibtex
@inproceedings{pljp2023,
  title={Precedent-Enhanced Legal Judgment Prediction with LLM and Domain-Model Collaboration},
  author={Your Name and Co-authors},
  booktitle={Proceedings of EMNLP},
  year={2023}
}
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- CAIL2018 dataset providers
- OpenAI for API access
- All contributors and reviewers

## 📧 Contact

For questions or issues, please open an issue on GitHub or contact [your-email@example.com].

---

**Note**: This implementation requires OpenAI API access. Ensure you have sufficient API credits before running large-scale experiments.
