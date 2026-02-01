# 🎬 Multimodal Trailer Metadata Analyzer

**Netflix-style multimodal ML system** that ingests movie trailers (YouTube or local),
extracts video and subtitle signals, and predicts **genre, mood, and themes**
to support **content discovery, metadata enrichment, and cold-start ranking**.

## Problem
Streaming platforms face weak or incomplete metadata for new content, which negatively impacts search relevance, discovery, and cold-start ranking.

## Solution
Built a Netflix-style multimodal machine learning system that ingests movie trailers (local files or YouTube URLs) and automatically generates genre, mood, and theme metadata using video and subtitle signals.

## System Overview
- Subtitle extraction using Whisper  
- Video embeddings using VideoMAE  
- Text embeddings using Sentence-BERT  
- Multimodal fusion via concatenation  
- Multi-label classifier with sigmoid outputs  
- Deployed via Streamlit demo
