# KGCE: Knowledge-Augmented Dual-Graph Evaluator for Cross-Platform Educational Agent Benchmarking with Multimodal Language Models

<p align="center">
  <img src='https://raw.githubusercontent.com/camel-ai/crab/main/assets/CRAB_logo1.png' width=800>
</p>

## Overview
KGCE is a framework for building LLM agent benchmark environments in a Python-centric way.

#### Key Features

🌐 Cross-platform and Multi-environment
* Create build agent environments that support various deployment options including in-memory, Docker-hosted, virtual machines, or distributed physical machines, provided they are accessible via Python functions.
* Let the agent access all the environments in the same time through a unified interface.

⚙ ️Easy-to-use Configuration
* Add a new action by simply adding a `@action` decorator on a Python function.
* Define the environment by integrating several actions together.

📐 Novel Benchmarking Suite
* Define tasks and the corresponding evaluators in an intuitive Python-native way.
* Introduce a novel graph evaluator method providing fine-grained metrics.

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

[![demo_video](https://i.ytimg.com/vi_webp/PNqrHNQlU6I/maxresdefault.webp)](https://www.youtube.com/watch?v=PNqrHNQlU6I&ab_channel=CamelAI)
<<<<<<< HEAD
=======


>>>>>>> dcaba711a4a42d1e36e25db72c3db3900892ce52
