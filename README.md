# InsightFlow

**LLM-Driven Synthesis of Patient Narratives for Mental Health into Causal Models**

InsightFlow is a research-oriented pipeline that uses Large Language Models (LLMs) for clinical case formulation through causal graphs. They are generated from mental health conversations. The system is designed to help researchers, clinicians, and data scientists convert unstructured dialogue into structured, interpretable representations of symptom dynamics and underlying causal relationships.

---

## 🚀 Project Overview

Extracting meaningful structure from mental health conversations is often time-consuming and subjective. InsightFlow automates this process by:

- Processing raw conversational transcripts
- Leveraging LLMs to interpret narrative context and semantics
- Constructing **causal graphs** that capture relationships between symptoms, behaviors, stressors, and outcomes
- Validating generated graphs against human-annotated ground truth

The project supports research in computational mental health, causal inference, explainable AI, and narrative understanding.

---

## 📦 Repository Structure
InsightFlow/
├── AnnotatorGroundTruth/ # Human-annotated causal graphs and mappings
├── AutomaticGraphAnalysis/ # Evaluation and benchmarking scripts
├── CausalGraphGenerationCode/ # Core pipeline for LLM-based graph generation
├── Conversations/ # Raw mental health conversation transcripts
├── LLMGeneratedGraphs/ # Automatically generated causal graphs
├── README.md # Project documentation

Each directory is modular and designed to support independent experimentation and reproducibility.

---

## 🔍 Key Features

### 🧠 Narrative Understanding with LLMs
- Interprets complex patient narratives
- Extracts symptoms, factors, and implied relationships
- Handles nuanced conversational context

### 📉 Causal Graph Generation
- Converts narrative insights into structured causal graphs
- Supports multiple graph formats for analysis and visualization

### 📊 Ground Truth Evaluation
- Includes expert-annotated reference graphs
- Computes similarity and quality metrics between human and LLM-generated graphs

---
