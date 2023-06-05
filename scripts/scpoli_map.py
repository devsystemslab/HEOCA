import os

import numpy as np
import pandas as pd
import torch
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

import scanpy as sc
import scvi
import anndata
from anndata import AnnData

import scanpy.external as sce
from scvi.model.utils import mde

from scarches.dataset.trvae.data_handling import remove_sparsity
from scarches.models.scpoli import scPoli

import umap

# sys.path.append('/home/xuq44/git/')
# import util_single_cell.scripts.py_util as qsc
# WD = "/mnt/atlas_building"
WD = "/dss/dssfs02/lwp-dss-0001/pn36po/pn36po-dss-0001/di93vel/atlas_building"

def read_samples(project_dir, sample_sheet, annot_dir):
    adata_list=[]
    for sample in sample_sheet.sample_id:
        sample_file=f"{project_dir}/h5ad_pyraw/cp_h5ad/{sample}_pyraw.h5ad"
        adata = sc.read_h5ad(sample_file)
        adata.obs['sample_id'] = sample
        adata.var = adata.var.drop(adata.var.columns, axis=1)
        # adata = cc_reg(adata, ccgenes_file)
        meta_file = f"{annot_dir}/{sample}_annotation.txt"
        if os.path.exists(meta_file) and os.path.getsize(meta_file)>1:
            meta_data = pd.read_csv(meta_file, sep="\t", index_col=0)
            adata.obs = pd.merge(adata.obs, meta_data, left_index=True, right_index=True)
        else:
            adata.obs['level_1'] = 'epithelial'
            adata.obs['level_2'] = 'na'

        adata_list.append(adata)
    adata_concat = anndata.concat(adata_list, join='outer', fill_value=0)
    adata_concat.obs['publication'] = ['_'.join(i.split('_')[:3]) for i in adata_concat.obs.sample_id.tolist()]

    adata_concat.obs.index.name = "cells"

    return adata_concat
    
def clear_genes(project_dir, adata):
    # clear_genes = pd.read_csv("/home/xuq44/refgenomes/hg38/hg_genes_clear_nocc.txt", header=None)[0].tolist()
    clear_genes = pd.read_csv(os.path.join(project_dir, "hg_genes_clear.txt"), header=None)[0].tolist()
    sub_clear_genes = [i for i in clear_genes if i in adata.var.index.tolist()]
    adata = adata[:, sub_clear_genes]
    
    return adata

def cc_reg(adata, ccgenes_file):
    cell_cycle_genes = [x.strip() for x in open(ccgenes_file)]
    s_genes = cell_cycle_genes[:43]
    g2m_genes = cell_cycle_genes[43:]
    cell_cycle_genes = [x for x in cell_cycle_genes if x in adata.var_names]
    sc.tl.score_genes_cell_cycle(adata, s_genes=s_genes, g2m_genes=g2m_genes)
    sc.pp.regress_out(adata, ['S_score', 'G2M_score'])
    return adata

def pre_inti(adata):
    adata.layers['counts']=adata.X
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata.raw = adata
    sc.pp.highly_variable_genes(adata, n_top_genes=3000, batch_key='sample_id')
    adata = adata[:,adata.var.highly_variable]
    return adata

def pre_inti0(adata):
    adata.layers['counts']=adata.X
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata.raw = adata

    return adata

def run_merge(adata):
    sc.pp.neighbors(adata, n_pcs=30)
    sc.tl.umap(adata)
    return adata

def run_scvi(adata, method='scANVI'):
    adata = adata.copy()
    scvi.model.SCVI.setup_anndata(adata, layer='counts', batch_key="sample_id")
    vae = scvi.model.SCVI(adata, n_layers=2, n_latent=30, gene_likelihood="nb")
    vae.train()
    adata.obsm["X_scVI"] = vae.get_latent_representation()

    if method == 'scVI':
        sc.pp.neighbors(adata, use_rep="X_scVI")
        sc.tl.leiden(adata)
        sc.tl.umap(adata)

        return adata

    else:
        lvae = scvi.model.SCANVI.from_scvi_model(
            vae,
            adata=adata,
            labels_key="level_2",
            unlabeled_category="na",
        )
        lvae.train(max_epochs=20, n_samples_per_label=100)
        
        adata.obsm["X_scANVI"] = lvae.get_latent_representation(adata)

        sc.pp.neighbors(adata, use_rep="X_scANVI")
        sc.tl.leiden(adata)
        sc.tl.umap(adata)

        return adata

def get_kwargs():
    early_stopping_kwargs = {
        "early_stopping_metric": "val_prototype_loss",
        "mode": "min",
        "threshold": 0,
        "patience": 20,
        "reduce_lr": True,
        "lr_patience": 13,
        "lr_factor": 0.1,
    }
    return early_stopping_kwargs

def train_scpoli(adata):
    early_stopping_kwargs = get_kwargs()

    scpoli_model = scPoli(
        adata=adata,
        unknown_ct_names=['na'],
        condition_keys='sample_id',
        cell_type_keys=['level_1', 'level_2'],
        embedding_dims=3,
        hidden_layer_sizes=[int(np.sqrt(adata.shape[0]))]
    )

    scpoli_model.train(
        n_epochs=5,
        pretraining_epochs=4,
        early_stopping_kwargs=early_stopping_kwargs,
        eta=10,
        alpha_epoch_anneal=100
    )

    return scpoli_model

def map_query(adata, model):
    early_stopping_kwargs = get_kwargs()

    scpoli_query = scPoli.load_query_data(
        adata=adata,
        reference_model=model,
        labeled_indices=[],
    )

    scpoli_query.train(
        n_epochs=5,
        pretraining_epochs=4,
        early_stopping_kwargs=early_stopping_kwargs,
        eta=10,
        alpha_epoch_anneal=100
    )

    return scpoli_query

def umap_quan(adata, use_rep=None):
    if use_rep==None:
        use_rep = 'X'
        model = umap.UMAP(n_neighbors=5, random_state=42, min_dist=0.5).fit(adata.X)
    
    adata.obsm['X_umap'] = model.embedding_
    adata.uns['umap_model'] = model

    return adata

def umap_transform_quan(adata_ref, adata_que):
    model = adata_ref.uns['umap_model']
    que_embedding = model.transform(adata_que.X)
    adata_que.obsm['X_umap'] = que_embedding
    adata_all = adata_ref.concatenate(adata_que, batch_key='query')
    
    return adata_all

if __name__ == "__main__":
    ## load data
    project_dir = os.path.join(WD, "data")
    annot_dir = os.path.join(project_dir, "sample_annot")
    sample_sheet = pd.read_csv(os.path.join(project_dir, "all_samples_sheets_Merel.txt"), sep="\t")
    sample_sheet = sample_sheet[sample_sheet.tissue=='lung']

    reference = sc.read(os.path.join(project_dir, "Assembled10DomainsEpithelial.h5ad"))
    reference = clear_genes(project_dir, reference)
    reference.obs['level_1'] = 'epithelial'
    # TODO
    reference.obs['level_2'] = 'na'
    reference.obs['sample_id'] = reference.obs['donor'].copy()
    sc.pp.highly_variable_genes(reference, n_top_genes=3000, batch_key='sample_id')
    reference = reference[:,reference.var.highly_variable]

    organoid = read_samples(project_dir, sample_sheet, annot_dir)
    organoid = organoid[organoid.obs.level_1=='epithelial']
    organoid = pre_inti0(organoid)
    organoid.X = organoid.X.astype(np.float32)
    
    try:
        organoid = organoid[:,reference.var.index]
    except:
        a = set(reference.var.index)
        b = set(organoid.var.index)
        overlap = a.intersection(b)
        reference = reference[:,pd.Index(overlap)]
        organoid = organoid[:,pd.Index(overlap)]

    ## establish reference model
    scpoli_model = train_scpoli(reference)

    reference.obsm["X_scPoli"] = scpoli_model.get_latent(
        reference,
        mean=True,
    )

    sc.pp.neighbors(reference, use_rep="X_scPoli")
    sc.tl.umap(reference)

    ## map organoid data to reference
    scpoli_query = map_query(organoid, scpoli_model)

    # get latent representation of reference data
    scpoli_query.model.eval()
    data_latent_source = scpoli_query.get_latent(
        reference,
        mean=True,
    )

    adata_latent_source = AnnData(data_latent_source)
    adata_latent_source.obs = reference.obs.copy()

    # get latent representation of query data
    data_latent_target = scpoli_query.get_latent(
        organoid,
        mean=True,
    )

    adata_latent_target = AnnData(data_latent_target)
    adata_latent_target.obs = organoid.obs.copy()

    breakpoint()
    # get label annotations
    results_dict = scpoli_query.classify(organoid)
    adata_latent_target.obs['cell_type_pred'] = results_dict['level_2']['preds'].tolist()
    adata_latent_target.obs['cell_type_uncert'] = results_dict['level_2']['uncert'].tolist()
    adata_latent_target.obs['classifier_outcome'] = (
        adata_latent_target.obs['cell_type_pred'] == adata_latent_target.obs['level_2']
    )

    breakpoint()
    # get prototypes
    labeled_prototypes = scpoli_query.get_prototypes_info()
    labeled_prototypes.obs['study'] = 'labeled prototype'
    unlabeled_prototypes = scpoli_query.get_prototypes_info(prototype_set='unlabeled')
    unlabeled_prototypes.obs['study'] = 'unlabeled prototype'

    adata_latent_source = umap_quan(adata_latent_source)
    adata_latent_target = umap_transform_quan(adata_latent_source, adata_latent_target)


