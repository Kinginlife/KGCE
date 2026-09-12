# KGCE: Knowledge-Augmented Dual-Graph Evaluator for Cross-Platform Educational Agent Benchmarking with Multimodal Language Models
## Ours work has been accepted by SMC (CCF-C)!

## Overview
KGCE is a Python - centric framework designed to construct benchmark environments for Large Language Model (LLM) agents, with a specific focus on cross - platform educational agent benchmarking, integrated with multimodal language models.

#### Key Features

🌐 Cross-Platform Educational Task Support
* Facilitates cross-platform execution (Windows, Android, educational tools) for synchronized educational workflow tasks.

⚙ ️Knowledge - Augmented Architecture
* Integrates domain-specific knowledge via a structured JSON base, handling closed-domain educational software (e.g., XiaoYa Intelligent Assistant) for accurate interactions.

📐 Dual-Graph Evaluation
* Employs a Dual-Graph Evaluator to capture educational criteria with fine-grained metrics for precise task assessment.

## Installation

#### Prerequisites

- Python 3.10 or newer

```bash
pip install requirements.txt
```

## Experiment on KGCE

All datasets and experiment code are in [kgce-benchmark](./kgce-benchmark/) directory. 
run
```bash
poetry run python -m kgce-benchmark.main --model qwenvl --model-base-url https://dashscope.aliyuncs.com/compatible-mode/v1 --env android --model-api-key XXX --task-id 069x
```

## Demo Video

https://private-user-images.githubusercontent.com/134140488/434705359-7f5739e4-ac47-4a79-b097-fc7883bb88d5


