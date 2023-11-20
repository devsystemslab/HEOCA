import os
import sys
import torch
import numpy as np
import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import anndata

from sklearn.metrics import classification_report
from sklearn.metrics.pairwise import cosine_similarity

from scarches.dataset.trvae.data_handling import remove_sparsity
from scarches.models.scpoli import scPoli
import scarches as sca
import scvi
from scib_metrics.benchmark import Benchmarker

in_dir = "/home/xuq44/projects/hgioa/data/results/organoid_samples_summary/results/benchmarker"
tissue = "gut"

adata_harmony = sc.read_h5ad(f"{in_dir}/{tissue}_harmonypy_integration.h5ad")
adata_bbknn = sc.read_h5ad(f"{in_dir}/{tissue}_bbknn_integration.h5ad")
adata_combat = sc.read_h5ad(f"{in_dir}/{tissue}_combat_integration.h5ad")
adata_scvi = sc.read_h5ad(f"{in_dir}/{tissue}_scvi_integration.h5ad")
adata_scanvi = sc.read_h5ad(f"{in_dir}/{tissue}_scanvi_integration.h5ad")
adata_scpoli = sc.read_h5ad(f"{in_dir}/{tissue}_scpoli_integration.h5ad")
adata_rsspearson = sc.read_h5ad(f"{in_dir}/{tissue}_csspearson_integration.h5ad")
adata_rssspearman= sc.read_h5ad(f"{in_dir}/{tissue}_cssspearman_integration.h5ad")

adata_scanvi.obsm['X_scPoli'] = adata_scpoli.obsm['X_scPoli']
adata_scanvi.obsm['X_harmony'] = adata_harmony.obsm['X_pca_harmony']
adata_scanvi.obsm['X_bbknn'] = adata_bbknn.obsm['X_pca']
adata_scanvi.obsm['X_combat'] = adata_combat.obsm['X_pca']
adata_scanvi.obsm['X_csspearson'] = adata_rsspearson.obsm['X_corr']
adata_scanvi.obsm['X_cssspearman'] = adata_rssspearman.obsm['X_corr']

bm = Benchmarker(
    adata_scanvi,
    batch_key="sample_id",
    label_key="level_2",
    embedding_obsm_keys=["X_pca", "X_scVI", "X_scANVI", "X_scPoli", 
                         "X_harmony", "X_bbknn", "X_combat", 
                         "X_csspearson", "X_cssspearman"],
    n_jobs=-1,
)

bm.benchmark()

bench_results1 = bm.plot_results_table(min_max_scale=False, save_dir=f"{in_dir}/figures")
os.system(f"mv {in_dir}/figures/scib_results1.svg {in_dir}/figures/{tissue}_scib_results.svg")

bench_results2 = bm.plot_results_table(min_max_scale=True, save_dir=f"{in_dir}/figures")
os.system(f"mv {in_dir}/figures/scib_results2.svg {in_dir}/figures/{tissue}_scib_results_scale.svg")

df1 = bm.get_results(min_max_scale=False)
df1.to_csv(f"{in_dir}/{tissue}_scib_results.txt", sep="\t")

df2 = bm.get_results(min_max_scale=True)
df2.to_csv(f"{in_dir}/{tissue}_scib_results_scale.txt", sep="\t")

