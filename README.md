# KGCE: Knowledge-Augmented Dual-Graph Evaluator for Cross-Platform Educational Agent Benchmarking with Multimodal Language Models


## Overview
KGCE is a Python - centric framework designed to construct benchmark environments for Large Language Model (LLM) agents, with a specific focus on cross - platform educational agent benchmarking, integrated with multimodal language models.

#### Key Features

🌐 Cross - Platform Educational Task Support
* Facilitates cross - platform execution (Windows, Android, educational tools) for synchronized educational workflow tasks.

⚙ ️Knowledge - Augmented Architecture
* Integrates domain-specific knowledge via a structured JSON base, handling closed-domain educational software (e.g., XiaoYa Intelligent Assistant) for accurate interactions.

📐 Dual - Graph Evaluation
* Employs a Dual-Graph Evaluator to capture educational criteria with fine-grained metrics for precise task assessment.

## Installation

#### Prerequisites

- Python 3.10 or newer

```bash
pip install requirements.txt
```

## Experiment on CRAB-Benchmark-v0

All datasets and experiment code are in [kgce-benchmark](./kgce-benchmark/) directory. 

## Examples

#### Run template environment with openai agent

```bash
export OPENAI_API_KEY=<your api key>
python examples/single_env.py
python examples/multi_env.py
```

## Demo Video

[![demo_video]([KGCE1.mp4](https://github.com/Kinginlife/KGCE/raw/refs/heads/main/KGCE1.mp4))]

https://private-user-images.githubusercontent.com/134140488/434705359-7f5739e4-ac47-4a79-b097-fc7883bb88d5


