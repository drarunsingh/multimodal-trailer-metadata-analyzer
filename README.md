#  Multimodal Trailer Metadata Analyzer

**Netflix-style multimodal ML system** that ingests movie trailers (YouTube or local),
extracts video and subtitle signals, and predicts **genre, mood, and themes**
to support **content discovery, metadata enrichment, and cold-start ranking**.

## Problem
Streaming platforms face weak or incomplete metadata for new content, which negatively impacts search relevance, discovery, and cold-start ranking.

## Solution
Built a Netflix-style multimodal machine learning system that ingests movie trailers (local files or YouTube URLs) and automatically generates genre, mood, and theme metadata using video and subtitle signals.

## System Overview
![System Architecture](architecture diagram.png)

- Subtitle extraction using Whisper  
- Video embeddings using VideoMAE  
- Text embeddings using Sentence-BERT  
- Multimodal fusion via concatenation  
- Multi-label classifier with sigmoid outputs  
- Deployed via Streamlit demo

## Evaluation
Multi-label classification evaluated using Micro and Macro F1 scores (~0.95 on a curated dataset).  
Emphasis was placed on system robustness rather than raw accuracy due to limited dataset size.

## Core Skills Demonstrated
- Multimodal ML (video + text)
- Metadata enrichment & weak labels
- PyTorch training pipelines
- Multi-label evaluation metrics
- Robust inference and deployment

## Engineering Strengths
- Production-minded ingestion (YouTube, ffmpeg)
- Modular codebase and clean structure
- Model trade-off reasoning (compute vs accuracy)
- Failure analysis and scalability planning
## Where This Would Be Used in Production
- Content discovery & search relevance
- Cold-start ranking for new titles
- Merchandising rows (mood/theme-based collections)
- Editorial decision support

## Interview Positioning Statement
“I build production-oriented ML systems that transform raw media into structured, trustworthy signals for discovery, ranking, and personalization.”
