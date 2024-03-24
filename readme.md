# AI 学习笔记

## Transformers 框架

Transformers 用来下载和训练预训练模型，支持PyTorch、TensorFlow 和 JAX 框架互相操作，支持不同模式的常见任务

- NLP(Natural Language Processing): 文本分类(text classification)、命名实体分类(named entity recognition), 问答(question answering),语言建模(
  language modeling), 摘要（summarization）, 翻译(translation)、 多项选择(multiple choice)和文本生成（text generation）
- CV(Computer Vision): 图像分类(image classification), 目标检测(object detection), 图像分割(image segmentation)
- Audio: 自动语音识别(automatic speech recognition)和音频分类(audio classification)
- Multimodal:
  表格问答([table question answering](https://levelup.gitconnected.com/exploring-hugging-face-table-question-answering-86064a3d62c6),
  在NLP中，指的是一项特定的任务，目的是利用表格数据中提供的信息回答问题,表是结构化数据集，通常采用行和列的形式)、光学字符识别(table question
  answering)、扫描文档信息提取( information extraction from
  scanned documents)、视频分类(information extraction from scanned documents)和视觉问答(visual question answering)

### Main Classes

#### [Pipelines](https://huggingface.co/docs/transformers/main_classes/pipelines)

Pipelines抽象了模型的复杂度，提供专注于任务简单的 API，便于使用模型进行推断, 支持多种任务，包括(
可参考[任务简述](https://huggingface.co/docs/transformers/task_summary))：

- Named Entity Recognition
- Masked Language Modeling
- Sentiment Analysis
- Feature Extraction
- Question Answering.

## Datasets

`HF_DATASETS_OFFLINE = 1` 为离线模式

## 环境配置

- [nvidia-smi](https://developer.nvidia.com/cuda-downloads)

## 参考

- [Full Training](https://huggingface.co/learn/nlp-course/chapter0/1?fw=pt)
- [HF Learn](https://huggingface.co/learn)
- [Longchain Custom LLM](https://python.langchain.com/docs/modules/model_io/llms/custom_llm)

### `nvidia-smi`

1. 开启 persistence mode: `nvidia-smi -pm 1`, (`-i` 指定GPU编号)  
