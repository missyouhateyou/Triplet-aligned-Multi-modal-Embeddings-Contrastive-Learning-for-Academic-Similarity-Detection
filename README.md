# TIF-Contrast: Similarity Detection for Academic Documents via Triplet-aligned Text-Image-Formula Embeddings
This is the official implementation of the paper: "Similarity Detection for Academic Documents via Triplet-aligned Text-Image-Formula Embeddings and Contrastive Learning", currently submitted to Information Sciences.

## Abstract
Academic similarity detection is fundamental to research integrity. However, effective detection in multimodal documents is hindered by the semantic heterogeneity across text, figures, and mathematical expressions. We propose the Triplet-aligned Text-Image-Formula Contrastive Similarity Framework (TIF-Contrast).

Central to our approach is a triplet-aligned embedding mechanism that projects diverse modalities into a unified semantic space. By constructing contrastive "anchor-positive-negative" modality triplets, the framework forces semantically consistent cross-modal content to cluster closely in the shared space, enabling accurate, end-to-end similarity retrieval for papers, theses, and technical documents.

## Key Contributions
- Triplet-Aligned Embeddings: A novel strategy to constrain feature distribution across text, image, and formula modalities, bridging the gap of semantic heterogeneity.
- Unified Semantic Space: Facilitates direct cross-modal retrieval without the need for complex, manual late-fusion heuristics.
- Adaptive Multimodal Fusion: Specifically designed to handle variable modal distributions (e.g., documents with varying densities of formulas or figures).
- End-to-End Framework: Optimized for large-scale academic document databases to identify sophisticated plagiarism.

## Framework Architecture
TIF-Contrast utilizes three specialized encoders feeding into a shared contrastive learning objective:
- Text Encoder: Captures the semantic nuances of academic language.
- Image Encoder: Extracts visual features from figures, charts, and tables.
- Formula Encoder: A structural encoder designed for the symbolic semantics of mathematical expressions (LaTeX/MathML).

The alignment is governed by a Triplet Margin Loss:
$$L(a, p, n) = \max(0, d(f(a), f(p)) - d(f(a), f(n)) + \text{margin})$$
where $a, p, n$ represent anchor, positive, and negative modality samples respectively.

## Getting Started
### Prerequisites
- Python 3.9+
- PyTorch 2.0+
- CUDA 11.8+ (Recommended)

### Installation
```bash
git clone https://github.com/YourUsername/TIF-Contrast.git
cd TIF-Contrast
pip install -r requirements.txt
