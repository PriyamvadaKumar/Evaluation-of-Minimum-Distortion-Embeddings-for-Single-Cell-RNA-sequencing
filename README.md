
# Computational Genomics Project(02-710)-- Final Project 
---

## Evaluation of  Minimum Distortion Embeddings for Single-Cell RNA sequencing
---
## Introduction 

Organs are comprised of cells, each originating from different lineages and facilitating unique functions[1]. Even within distinct lineages, accumulated evidence shows that cells of the same type can exhibit a striking degree of heterogeneity, such as in the case of  macrophages[2]. Similarly, cancer cells are known to exhibit cellular plasticity, meaning that cancer cells can demonstrate a variety of cell behavior such as stemness, epithelial and mesenchymal transition. Therefore, this creates difficulties in trying to understand biological mechanisms and strategies for therapeutics action. 
Because of the advancement of flow cytometry and fluorescent activated cell sorting (FACS), we are able to distinguish several subtypes in a cell population based on cell-specific markers and fluorescent-labeled using antibodies[3]. This allows us to isolate the cells which express several markers. We can also perform multi-omics analysis to have deeper information of targeted cells after FACS[4]. However, assessing the full spectrum of the transcriptional profile of cells is not possible with these technologies.

The advent of single-cell transcriptomics allows for the molecular profiling at single-cell resolution thereby making it possible to investigate biological mechanisms.  Single-cell RNA sequencing is commonly used  to evaluate mRNA expression profiles of thousands of genes from thousands of individual cells in a high-throughput fashion. In order to analyze such high-dimensional data, dimensionality reduction is performed to enable assessment of different cell clusters present in the data. The latent space induced from such methods can then be used to inform model on downstream tasks, such cell type classification. However, there are many dimensionality reduction methods to choose from, and identifying the one most appropriate for your data can be challenging.

One such method, Minimum Distortion Embeddings (MDE)[5],has recently been developed that claims to be the generalization of traditional dimensionality reduction techniques such as PCA, tSNE[6,7], and UMAP[8]. MDE's unique formulation permits its users to specify inductive biases over the distance between samples in the induced latent space. Here, we aim to determine if MDE's flexibility does in fact provide advantages over canonical methods in the downstream task of cell-type classification. We benchmark MDE against PCA[9], tSNE, and UMAP on a single-cell RNA-seq dataset with 8 known cell types. The embeddings from each dimensionality reduction technique are used as input for a KNN and GMM classifier[10], where they are evaluated on a held out test set.



## Requirements: 
* Python 3.7+
	- numpy
	- pandas
	- scanpy
	- matplotlib
	- scikit-learn


## collaborators 
Daniel Penaherrera

Keng-Jung Lee
