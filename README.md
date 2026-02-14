# Turkish AI Agent with LangGraph

A multi-agent conversational AI system for Turkish language, powered by LangGraph and Turkish-Gemma-9b model.

## Features

- 🤖 **Multi-Agent Architecture**: Uses LangGraph for orchestrating different specialized agents
- 📚 **Wikipedia Integration**: Searches and retrieves information from Turkish Wikipedia
- 🧮 **Mathematical Calculations**: Handles basic to advanced math expressions
- 💬 **Natural Conversation**: Friendly chat interface for general interactions
- ⚡ **4-bit Quantization**: Efficient memory usage with optional quantization

## Architecture

The system uses a router-based architecture:
```
User Query → Router → [Wiki Agent | Math Agent | Chat Agent] → Response
```

## Installation

### Requirements

- Python 3.8+
- CUDA-capable GPU (recommended)
- 8GB+ VRAM for 4-bit quantized model

### Setup
```bash
# Clone repository
git clone https://github.com/yourusername/LangGraph-Agent.git
cd LangGraph-Agent

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Basic Usage
```bash
python agent.py
```

### Example Interactions
```
🧑 Question: Atatürk kimdir?
🤖 ANSWER: Mustafa Kemal Atatürk, Türkiye Cumhuriyeti'nin kurucusudur...

🧑 Question: 15*7 kaç eder?
🤖 ANSWER: 15 çarpı 7 işleminin sonucu 105'tir.

🧑 Question: Merhaba
🤖 ANSWER: Merhaba! Size nasıl yardımcı olabilirim?
```

## Technical Details

### Model

- **Base Model**: ytu-ce-cosmos/Turkish-Gemma-9b-v0.1
- **Quantization**: 4-bit (BitsAndBytes)
- **Framework**: PyTorch + Transformers

### Agent System

- **Orchestration**: LangGraph StateGraph
- **Pattern**: Router-based multi-agent
- **Components**:
  - Router Agent: Query classification
  - Wiki Agent: Information retrieval
  - Math Agent: Calculations
  - Chat Agent: General conversation

## Project Structure
```
turkish-ai-agent/
├── agent.py              # Main agent implementation
├── requirements.txt      # Python dependencies
├── README.md            # This file
└── examples/            # Usage examples
```

İşte güncellenmiş performans bölümü, gerçek test sonuçlarına göre:
📄 README.md - Performance Section
markdown## Performance

Benchmark results on RTX 4070 Ti SUPER with 4-bit quantization:

### Response Times
- **Simple Chat**: ~7-8 seconds
- **Math Calculation**: ~8-11 seconds  
- **Wikipedia Search**: ~30-35 seconds (includes API latency)

### Resource Usage
- **Memory Usage**: +8GB VRAM (4-bit quantization)

### Breakdown by Agent Type
| Agent Type | Average Time | Notes |
|------------|-------------|-------|
| Chat Agent | 7.8s | Direct LLM generation |
| Math Agent | 9.7s | LLM-based calculation |
| Wiki Agent | 32.5s | Includes Wikipedia API call (~15-20s) |


- Turkish-Gemma model by YTU CE Cosmos Lab
- LangGraph by LangChain
- Wikipedia API

## Citation
https://cosmos.yildiz.edu.tr/
https://huggingface.co/ytu-ce-cosmos/Turkish-Gemma-9b-v0.1
