import os
import sys
import argparse

import umap
import pickle

import numpy as np
import pandas as pd
import torch
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

import scanpy as sc
import scvi
import anndata

import scanpy.external as sce
from scvi.model.utils import mde

from sklearn.metrics import classification_report
from sklearn.metrics.pairwise import cosine_similarity

from scarches.dataset.trvae.data_handling import remove_sparsity
from scarches.models.scpoli import scPoli

def read_samples(project_dir, sample_sheet, annot_dir):
    adata_list=[]
    for sample in sample_sheet.sample_id:
        sample_file=f"{project_dir}/h5ad_pyraw/{sample}_pyraw.h5ad"
        adata0 = sc.read_h5ad(sample_file)
        adata0.obs['sample_id'] = sample
        adata0.var = adata0.var.drop(adata0.var.columns, axis=1)
        # adata0 = cc_reg(adata0, ccgenes_file)
        meta_file = f"{annot_dir}/{sample}_annotation.txt"
        if os.path.exists(meta_file) and os.path.getsize(meta_file)>1:
            meta_data = pd.read_csv(meta_file, sep="\t", index_col=0)
            adata0.obs = pd.merge(adata0.obs, meta_data, left_index=True, right_index=True)
        else:
            adata0.obs['level_1'] = 'epithelial'
            adata0.obs['level_2'] = 'na'

        adata_list.append(adata0)
    adata = anndata.AnnData.concatenate(*adata_list, join='outer', fill_value=0)
    del adata_list
    adata.obs['publication'] = ['_'.join(i.split('_')[:3]) for i in adata.obs.sample_id.tolist()]

    adata.obs.index.name = "cells"

    return adata
    
def clear_genes(adata):
    # clear_genes = pd.read_csv("/home/xuq44/refgenomes/hg38/hg_genes_clear_nocc.txt", header=None)[0].tolist()
    clear_genes = pd.read_csv("/home/xuq44/refgenomes/hg38/hg_genes_clear.txt", header=None)[0].tolist()
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

def pre_inti(adata, project_name):
    adata.layers['counts']=adata.X
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata.raw = adata
    if project_name == 'intestine':
        variable_genes = pd.read_csv("/home/xuq44/projects/hgioa/data/results/tissue_samples_db/CuratedAtlasQueryR/results_old/intestine_exp_avg.csv", 
                sep="\t", index_col=0).index.tolist()
        adata = adata[:, [i for i in variable_genes if i in adata.var.index]]
    else:
        sc.pp.highly_variable_genes(adata, n_top_genes=2000, batch_key='sample_id')
        adata = adata[:,adata.var.highly_variable]

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
            labels_key='level_2',
            unlabeled_category="na",
        )
        lvae.train(max_epochs=20, n_samples_per_label=100)
        
        adata.obsm["X_scANVI"] = lvae.get_latent_representation(adata)

        sc.pp.neighbors(adata, use_rep="X_scANVI")
        sc.tl.leiden(adata)
        sc.tl.umap(adata)

        return adata

def run_scpoli(adata):
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
        condition_key='sample_id',
        cell_type_keys=['level_1', 'level_2'],
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
        adata.obs["sample_id"].values,
        mean=True
    )
    sc.pp.neighbors(adata, use_rep="X_scPoli")
    sc.tl.leiden(adata)
    sc.tl.umap(adata, min_dist=0.1)
    adata.obsm['X_umap_min'] = adata.obsm['X_umap'].copy()
    sc.tl.umap(adata)
    adata.obsm['X_umap_sc'] = adata.obsm['X_umap'].copy()

    return scpoli_model, adata

def plot_results(adata, out_folder, project_name, method='scvi', mdist=False):
    if mdist:
        if "level_3" in adata.obs:
            with plt.rc_context():  # Use this to set figure params like size and dpi
                sc.pl.umap(adata, color=["publication", "level_1", "level_2", "level_3"], 
                            ncols=1, frameon=False, show=False)
                plt.savefig(f"{out_folder}/figures/{project_name}_{method}_integration2.png", dpi=300, bbox_inches='tight')
        else:
            with plt.rc_context():  # Use this to set figure params like size and dpi
                sc.pl.umap(adata, color=["publication", "level_1", "level_2"], 
                            ncols=1, frameon=False, show=False)
                plt.savefig(f"{out_folder}/figures/{project_name}_{method}_integration2.png", dpi=300, bbox_inches='tight')
    else:
        if "level_3" in adata.obs:
            with plt.rc_context():  # Use this to set figure params like size and dpi
                sc.pl.umap(adata, color=["publication", "level_1", "level_2", "level_3"], 
                            ncols=1, frameon=False, show=False)
                plt.savefig(f"{out_folder}/figures/{project_name}_{method}_integration.png", dpi=300, bbox_inches='tight')
        else:
            with plt.rc_context():  # Use this to set figure params like size and dpi
                sc.pl.umap(adata, color=["publication", "level_1", "level_2"], 
                            ncols=1, frameon=False, show=False)
                plt.savefig(f"{out_folder}/figures/{project_name}_{method}_integration.png", dpi=300, bbox_inches='tight')

def mk_new_annot(adata, group_name, level):
    aa = adata.obs.groupby([group_name, level]).size().reset_index()
    new_clust_map={}

    for i in aa[group_name].unique():
        bb = aa[aa[group_name]==i]
        new_clust_map[i]=bb.loc[bb[0].idxmax()][level]

    adata.obs[f'{level}_late'] = adata.obs[group_name].map(new_clust_map)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--inti_method', required=True, help='intigration method')
    parser.add_argument('-n', '--project_name', required=True, help='samples to intigration')
    parser.add_argument('-a', '--annotation', required=True, help='annotation folder')
    parser.add_argument('-o', '--out_dir', required=True, help='out dir')
    parser.add_argument('-p', '--project_dir', default="/home/xuq44/projects/hgioa/data/results/organoid_samples_summary")
    parser.add_argument('-b', '--benchmarker', default=False, action=argparse.BooleanOptionalAction)

    o = parser.parse_args()

    out_folder = f"{o.project_dir}/results/{o.out_dir}"
    os.system(f"mkdir -pv {out_folder}/figures")
    os.system(f"mkdir -pv {out_folder}/{o.project_name}_scpoli_model")
    ccgenes_file = f'{o.project_dir}/regev_lab_cell_cycle_genes.txt'

    sample_sheet = pd.read_csv(f"{o.project_dir}/all_samples_sheets.txt", sep="\t")
    
    # Random select samples for benchmarker
    if o.benchmarker:
        sample_sheet = sample_sheet.sample(10, random_state=0)

    if o.project_name=='gut':
        sample_sheet = sample_sheet
    if o.project_name=='intestine':
        sample_sheet = sample_sheet[sample_sheet.tissue=='intestine']
    if o.project_name=='lung':
        sample_sheet = sample_sheet[sample_sheet.tissue=='lung']
    if o.project_name=='liver_bileduct':
        sample_sheet = sample_sheet[sample_sheet.tissue.isin(['liver', 'bileduct'])]
    if o.project_name=='pancreas':
        sample_sheet = sample_sheet[sample_sheet.tissue.isin(['pancreas'])]
      
    annot_dir = f"{o.project_dir}/{o.annotation}/"
    adata = read_samples(o.project_dir, sample_sheet, annot_dir)
    adata = clear_genes(adata)

    if o.inti_method == 'scvi':
        adata = pre_inti(adata, o.project_name)
        adata = run_scvi(adata, method='scVI')
        plot_results(adata, out_folder, o.project_name, method='scvi', mdist=False)
        adata.write_h5ad(f"{out_folder}/{o.project_name}_scvi_integration.h5ad", compression="gzip")

        sc.tl.umap(adata, min_dist=0.1)
        plot_results(adata, out_folder, o.project_name, method='scvi', mdist=True)
        adata.write_h5ad(f"{out_folder}/{o.project_name}_scvi_integration2.h5ad", compression="gzip")

    if o.inti_method == 'scanvi':
        adata = pre_inti(adata, o.project_name)
        adata = run_scvi(adata, method='scANVI')
        plot_results(adata, out_folder, o.project_name, method='scanvi', mdist=False)
        adata.write_h5ad(f"{out_folder}/{o.project_name}_scanvi_integration.h5ad", compression="gzip")

        sc.tl.umap(adata, min_dist=0.1)
        plot_results(adata, out_folder, o.project_name, method='scanvi', mdist=True)
        adata.write_h5ad(f"{out_folder}/{o.project_name}_scanvi_integration2.h5ad", compression="gzip")

    if o.inti_method == 'merge':
        os.system(f"mkdir -pv {out_folder}/{o.project_name}_merge")
        adata.write_h5ad(f"{out_folder}/{o.project_name}_merge/{o.project_name}_merge.h5ad", 
                            compression="gzip")

    if o.inti_method == 'scpoli':
        adata = pre_inti(adata, o.project_name)

        scpoli_model, adata = run_scpoli(adata)

        ref_path = f'{out_folder}/{o.project_name}_scpoli_model/scpoli_model/'
        scpoli_model.save(ref_path, overwrite=True)

        ## make empty adata for query to have the same genes
        empty_adata=sc.pp.filter_cells(adata, max_counts=-1, copy=True)
        empty_adata.obs=pd.DataFrame()
        empty_adata.var=pd.DataFrame(empty_adata.var.index).rename(columns={0:'cells'}).set_index('cells')
        del(empty_adata.uns)
        del(empty_adata.obsm)
        del(empty_adata.obsp)
        del(empty_adata.layers)
        empty_adata.write_h5ad(f"{out_folder}/{o.project_name}_scpoli_model/empty.h5ad", 
                                    compression="gzip")

        ## save scpoli results
        data_latent_source = adata.obsm["X_scPoli"]
        adata_latent_source = sc.AnnData(data_latent_source)
        adata_latent_source.obs = adata.obs.copy()

        model = umap.UMAP(n_neighbors=15, random_state=0, min_dist=0.5).fit(adata_latent_source.X)
        adata_latent_source.obsm['X_umap'] = model.embedding_
        adata.obsm['X_umap'] = model.embedding_

        adata_latent_source.write_h5ad(f"{out_folder}/{o.project_name}_scpoli_model/adata_latent_source.h5ad", 
                                    compression="gzip")

        pickle.dump(model, open(f"{out_folder}/{o.project_name}_scpoli_model/umap_model.sav", 'wb'))

        sc.tl.leiden(adata, resolution=10, key_added='leiden_10.0')

        mk_new_annot(adata, 'leiden_10.0', 'level_1')
        mk_new_annot(adata, 'leiden_10.0', 'level_2')
        mk_new_annot(adata, 'leiden_10.0', 'level_3')

        adata.obs[['level_1_late', 'level_2_late', 'level_3_late']].to_csv(f"{out_folder}/{o.project_name}_scpoli_model/{o.project_name}_scpoli_late_annot.txt",
                        sep='\t')

        adata.write_h5ad(f"{out_folder}/{o.project_name}_scpoli_model/{o.project_name}_scpoli_integration.h5ad", 
                            compression="gzip")
