import os
import sys
import argparse

import numpy as np
import pandas as pd
import torch
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import umap

import scanpy as sc
import scvi
import anndata

import scanpy.external as sce
from scvi.model.utils import mde

from sklearn.metrics import classification_report
from sklearn.metrics.pairwise import cosine_similarity

from scarches.dataset.trvae.data_handling import remove_sparsity
from scarches.models.scpoli import scPoli


project_dir = "/home/xuq44/projects/hgioa/data/results/organoid_samples_summary"


def pre_inti(adata):
    adata.layers['counts']=adata.X
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata.raw = adata
    sc.pp.highly_variable_genes(adata, n_top_genes=3000, batch_key='Batch')
    adata = adata[:,adata.var.highly_variable]
    return adata

def umap_quan(adata, use_rep=None):
    if use_rep==None:
        use_rep = 'X'
        model = umap.UMAP(n_neighbors=5, random_state=1, min_dist=0.5).fit(adata.X)
    
    adata.obsm['X_umap'] = model.embedding_

    return adata, model

adata = sc.read_h5ad(f"{project_dir}/db/fetal_atlas/local_1M.h5ad")
adata = adata[(adata.obs.tissue.isin(["pancreas","lung","liver","intestine",
                                        "spleen","thymus","stomach"])) &
                (adata.obs.disease=='normal')]

adata.var.index = adata.var.feature_name

clear_genes = pd.read_csv("/home/xuq44/refgenomes/hg38/hg_genes_clear_new.txt",sep='\t', header=None)[0].tolist()
adata = adata[:,adata.var.index.isin(clear_genes)]
adata.var.index = adata.var.feature_name

sample='Yu_Cell_2021_H9_tHIO_WK4'

sample_file=f"{project_dir}/h5ad_pyraw/{sample}_pyraw.h5ad"
adata00 = sc.read_h5ad(sample_file)
adata = adata[:,adata.var.index.isin(adata00.var.index)]

adata.var.index = adata.var.feature_name

# mitochondrial genes
adata.var['mt'] = adata.var_names.str.startswith('MT-') 
# ribosomal genes
adata.var['ribo'] = adata.var_names.str.startswith(("RPS","RPL"))

adata.var['mribo'] = adata.var_names.str.startswith(("MRPS","MRPL"))

# hemoglobin genes.
adata.var['hb'] = adata.var_names.str.contains(("^HB[^(P)]"))

sc.pp.calculate_qc_metrics(adata, qc_vars=['mt','ribo','mribo','hb'], 
                           percent_top=None, log1p=False, inplace=True)
sc.pp.filter_cells(adata, min_genes=200)
malat1 = adata.var_names.str.startswith('MALAT1')
mito_genes = adata.var_names.str.startswith('MT-')
rb_genes = adata.var_names.str.startswith(("RPS","RPL"))

remove = np.add(mito_genes, malat1)
remove = np.add(remove, rb_genes)
keep = np.invert(remove)

adata = adata[:,keep]
adata = pre_inti(adata)

early_stopping_kwargs = {
    "early_stopping_metric": "val_prototype_loss",
    "mode": "min",
    "threshold": 0,
    "patience": 20,
    "reduce_lr": True,
    "lr_patience": 13,
    "lr_factor": 0.1,
}

scpoli_model = scPoli(
    adata=adata,
    unknown_ct_names=['na'],
    condition_key='Batch',
    cell_type_keys=['Main_cluster_name'],
    embedding_dim=3,
    hidden_layer_sizes=[int(np.sqrt(adata.shape[0]))]
)

scpoli_model.train(
    n_epochs=5,
    pretraining_epochs=4,
    early_stopping_kwargs=early_stopping_kwargs,
    eta=10,
    alpha_epoch_anneal=100
)

adata.obsm["X_scPoli"] = scpoli_model.get_latent(
    adata.X, 
    adata.obs["Batch"].values,
    mean=True
)

ref_path = f'{project_dir}/db/fetal_atlas/scpoli_ref/scpoli_model/'
scpoli_model.save(ref_path, overwrite=True)

sc.pp.neighbors(adata, use_rep="X_scPoli")
sc.tl.umap(adata)

empty_adata=sc.pp.filter_cells(adata, max_counts=-1, copy=True)
empty_adata.obs=pd.DataFrame()
empty_adata.var=pd.DataFrame(empty_adata.var.index).set_index('feature_name')
del(empty_adata.uns)
del(empty_adata.obsm)
del(empty_adata.obsp)
del(empty_adata.layers)

empty_adata.write_h5ad(f"{project_dir}/db/fetal_atlas/scpoli_ref/empty.h5ad", 
                               compression="gzip")

data_latent_source = adata.obsm["X_scPoli"]

adata_latent_source = sc.AnnData(data_latent_source)
adata_latent_source.obs = adata.obs.copy()

adata_latent_source, umap_model = umap_quan(adata_latent_source)

adata_latent_source.write_h5ad(f"{project_dir}/db/fetal_atlas/scpoli_ref/adata_latent_source.h5ad", 
                               compression="gzip")

pickle.dump(umap_model, open(f"{project_dir}/db/fetal_atlas/scpoli_ref/umap_model.sav", 'wb'))

