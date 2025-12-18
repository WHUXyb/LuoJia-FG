# LuoJia-FG: An Attribute-Driven Vision-Language Dataset for Hierarchical Fine-Grained Land Cover Classification

This repository contains the official dataset link, code, and benchmarks for the paper: **"LuoJia-FG: An Attribute-Driven Vision-Language Dataset for Hierarchical Fine-Grained Land Cover Classification"**.

## 📖 Introduction

**LuoJia-FG** is a large-scale, attribute-driven, and hierarchical fine-grained land-cover dataset designed to bridge the "semantic gap" in high-resolution Earth observation. Unlike existing datasets with flat taxonomies and coarse categories, LuoJia-FG introduces a deep semantic structure tailored for next-generation **Vision-Language Models (VLMs)**.

The dataset comprises **119,619 multimodal triplets** (Image, Mask, Text) covering 1.59 million km^2 across diverse biogeographical zones in China, with spatial resolutions ranging from **0.5m to 2.0m**.

### 🔥 Core Innovations

1. 
**Deep Hierarchical Taxonomy:** We establish a three-level hierarchy expanding from **8 primary** classes to **52 secondary** and **95 tertiary** fine-grained classes, offering unprecedented semantic granularity.


2. **Attribute-Driven Text Generation:** We introduce a novel **Attribute-Coding System** based on physical properties (e.g., phenology, canopy density, leaf type). This system programmatically generates rich, attribute-driven text descriptions for every image, bridging the pixel-knowledge gap.


3. 
**CLIP-HCNet Benchmark:** We propose the **CLIP-guided Hierarchical Classification Network**, which leverages attribute-driven text descriptions as semantic priors to resolve visual ambiguity among spectrally similar fine-grained categories.



## 📂 Dataset Structure

The dataset is organized hierarchically by split (`train`, `val`, `test`) and geographic origin (e.g., `Anhui`, `Gansu`). Text descriptions are centralized in JSON format.

```
LuoJia-FG/
├── train/
│   ├── text_descriptions.json      # Attribute-driven text for all training samples
│   ├── Anhui/
│   │   ├── images/                 # Source RGB images (512x512 PNG)
│   │   ├── L1_labels/              # Level-1 Masks (8 classes)
│   │   ├── L2_labels/              # Level-2 Masks (52 classes)
│   │   └── L3_labels/              # Level-3 Masks (95 classes)
│   ├── ... (other provinces)
├── val/
│   ├── text_descriptions.json
│   └── ...
└── test/
    ├── text_descriptions.json
    └── ...

```

### Taxonomy Visualization

*(Place your `sunburst chart` image here, e.g., `assets/sunburst.png`)*

> The three-level hierarchical taxonomy (L1 \rightarrow L2 \rightarrow L3). 
> 
> 

## 📊 Dataset Statistics

* **Total Samples:** 119,619 triplets
* 
**Spatial Resolution:** 0.5m - 2.0m (sourced from Gaofen-2, Ziyuan-3, Aerial Photography) 


* 
**Coverage:** 19,290 km^2 across 6 provinces in China 


* **Annotation:** Pixel-level masks for 3 hierarchy levels + Attribute-rich text descriptions.

## 🚀 Getting Started

### 1. Requirements

* Linux or macOS
* Python 3.8+
* PyTorch 1.10+
* CUDA 11.0+ (recommended)

Install dependencies:

```bash
pip install -r requirements.txt

```

### 2. Data Preparation

Please download the LuoJia-FG dataset from [Link to Dataset Download] and extract it to the `data/` folder.

### 3. Training CLIP-HCNet

To train the proposed CLIP-HCNet (or other baselines like SegNeXt, LSKNet), run:

```bash
# Example training command
python train.py --config configs/clip_hcnet_mit_b2.yaml --data_path ./data/LuoJia-FG

```

### 4. Evaluation

To evaluate the model on the test set for hierarchical metrics (HIoU, HC):

```bash
python test.py --checkpoint checkpoints/best_model.pth --eval_hierarchy

```

## 🏆 Benchmark Results

Performance comparison on the LuoJia-FG test set. **CLIP-HCNet** achieves state-of-the-art results, particularly in fine-grained (L3) metrics.

| Method | Backbone | L1 mIoU | L2 mIoU | L3 mIoU | HIoU |
| --- | --- | --- | --- | --- | --- |
| SegNeXt | MiT-B2 | 87.63 | 74.47 | 74.92 | 74.55 |
| RSSFormer | MiT-B2 | 82.96 | 78.02 | 75.61 | 75.20 |
| LSKNet | MiT-B2 | 86.67 | 78.84 | 76.83 | 76.91 |
| SemHi | MiT-B2 | **89.40** | 81.27 | 79.78 | 80.02 |
| **CLIP-HCNet (Ours)** | **MiT-B2** | 87.95 | **83.38** | **81.90** | **81.57** |

## 🖼️ Visualization

*(Place qualitative result images here, e.g., `assets/vis_comparison.png`)*

> Comparison of segmentation results. CLIP-HCNet (j) shows superior boundary details and fewer misclassifications compared to baselines.
> 
> 

## 🖊️ Citation

If you find this dataset or code useful for your research, please consider citing our paper:

```bibtex
@article{LuoJiaFG2025,
  title={LuoJia-FG: An Attribute-Driven Vision-Language Dataset for Hierarchical Fine-Grained Land Cover Classification},
  author={Xiong, Yibing and Wang, Yibo and Hu, Xiangyun and Deng, Kai and Liang, Aokun and Wang, Jiongwei and Gao, Baoguang},
  journal={Big Earth Data},
  year={2025}
}

```

## 📧 Contact

For questions regarding the dataset or code, please contact:

* **Yibo Wang**: wangyb@gzhu.edu.cn
* **Xiangyun Hu**: huxy@whu.edu.cn

## 🤝 Acknowledgements

This work was supported by the Fundamental Research Funds for the Central Universities, China (Grant No. 2042022dx0001) and other funding agencies.
