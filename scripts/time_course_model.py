import os
import sys
import argparse

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

import scanpy as sc, anndata as ad

from adpbulk import ADPBulk
import statsmodels.api as sm
import qnorm

import warnings
warnings.filterwarnings('ignore')



def read_samples(project_dir, sample_sheet):
    adata_list=[]
    for sample in sample_sheet.sample_id:
        sample_file=f"{project_dir}/h5ad_pyraw/{sample}_pyraw.h5ad"
        adata0 = sc.read_h5ad(sample_file)
        adata0.obs['sample_id'] = sample
        adata0.var = adata0.var.drop(adata0.var.columns, axis=1)
        # adata0 = cc_reg(adata0, ccgenes_file)
        meta_file = f"{project_dir}/sample_annot/{sample}_annotation.txt"
        if os.path.exists(meta_file) and os.path.getsize(meta_file)>1:
            meta_data = pd.read_csv(meta_file, sep="\t", index_col=0)
            adata0.obs = pd.merge(adata0.obs, meta_data, left_index=True, right_index=True)
        else:
            adata0.obs['level_1'] = 'epithelial'
            adata0.obs['level_2'] = 'na'

        adata_list.append(adata0)
    adata = anndata.AnnData.concatenate(*adata_list, join='outer', fill_value=0)
    adata.obs['publication'] = ['_'.join(i.split('_')[:3]) for i in adata.obs.sample_id.tolist()]

    adata.obs.index.name = "cells"
    
    return adata

def clear_genes(adata):
    # clear_genes = pd.read_csv("/home/xuq44/refgenomes/hg38/hg_genes_clear_nocc.txt", header=None)[0].tolist()
    clear_genes = pd.read_csv("/home/xuq44/refgenomes/hg38/hg_genes_clear.txt", header=None)[0].tolist()
    sub_clear_genes = [i for i in clear_genes if i in adata.var.index.tolist()]
    adata = adata[:, sub_clear_genes]
    
    return adata

def pre_inti(adata):
    adata.layers['counts']=adata.X
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=3000, batch_key='sample_id')
    adata.raw = adata
    adata = adata[:,adata.var.highly_variable]

    sc.pp.scale(adata)
    sc.tl.pca(adata, svd_solver='arpack')
    
    sc.pp.neighbors(adata, n_pcs=30)
    sc.tl.umap(adata)
    
    return adata


def add_sample_info(adata):
    sample_folder = "/home/xuq44/projects/hgioa/data/results/organoid_samples_summary"
    publications=[]
    subsamples=[]
    for i in adata.obs.sample_id.tolist():
        a = i.split('_')
        publications.append("_".join(a[:3]))
        subsamples.append("_".join(a[3:]))

    adata.obs['publication'] = publications
    adata.obs['sample_name'] = subsamples

    sample_df = pd.read_csv(f"{sample_folder}/duo_timecourse_samples.txt", sep="\t")
    # allsamples = sample_df.publication.unique()

    sample_df2 = sample_df.copy()
    sample_df2 = sample_df2.drop(columns=['sample','sample_id'])

    sample_df = sample_df.drop(columns=['sample','publication'])

    adata.obs = adata.obs.reset_index().merge(sample_df, on='sample_id', how='left').set_index('cells')
        
    return adata


def deseq2_norm(_data : pd.DataFrame):

    # step 1: log normalize
    log_data = np.log(_data)

    # step 2: average rows
    row_avg = np.mean(log_data, axis=1)
    
    # step 3: filter rows with zeros
    rows_no_zeros = row_avg[row_avg != -np.inf].index
    
    # step 4: subtract avg log counts from log counts
    ratios = log_data.loc[rows_no_zeros].subtract(row_avg.loc[rows_no_zeros], axis=0)
    
    # step 5: calculate median of ratios
    medians = ratios.median(axis=0)
    
    # step 6: median -> base number
    scaling_factors = np.e ** medians
#     print(medians)
    
    # step 7: normalize!
    normalized_data = _data / scaling_factors
    return normalized_data

def get_pseudo_bulk(adata, meta):
    adpb = ADPBulk(adata, "time")
    pseudobulk_matrix = adpb.fit_transform().T
    sample_meta = adpb.get_meta()
    pseudobulk_normlized = deseq2_norm(pseudobulk_matrix)
    return pseudobulk_normlized

def find_coef_genes(pseudo_bulk):
    x=np.array(pseudo_bulk.T)
    y=np.array([int(i[5:]) for i in pseudo_bulk.columns])
    model=LinearRegression().fit(x, y)
    r_sq = model.score(x, y)

    aq_results = pd.DataFrame(pseudo_bulk.index)
    aq_results =aq_results.rename(columns={0:'genes'})
    aq_results['coef'] = model.coef_
    aq_results = aq_results.sort_values('coef', ascending=False).reset_index()

    return aq_results

def plot_gene(gene):
    forplot = pd.DataFrame(pseudo_bulk.T[gene])
    forplot['time'] = [0, 47,59, 72,80, 
                     85, 101, 122, 127, 132]
    sns.scatterplot(forplot, x='time', y=gene)


project_dir = "/home/xuq44/projects/hgioa/data/results/organoid_samples_summary"
sample_sheet = pd.read_csv(f"{project_dir}/duo_timecourse_samples.txt", sep="\t")


adata = read_samples(project_dir, sample_sheet)

adata = clear_genes(adata)
# adata = pre_inti(adata)
adata = add_sample_info(adata)

celltype_genes=pd.DataFrame()

for celltype in adata.obs.level_2.unique():
        
        sub_adata = adata[(adata.obs.level_2==celltype)|(adata.obs.time=='d0')]

        if len(sub_adata.obs.time.unique())>2:
            pseudo_bulk = get_pseudo_bulk(sub_adata, "time")

            x=np.array(pseudo_bulk.T)
            y=np.array([int(i[5:-3]) for i in pseudo_bulk.columns])

            X = sm.add_constant(x)
            est = sm.OLS(y, X).fit()

            smresults = pd.DataFrame(pseudo_bulk.index).rename(columns={0:'genes'}) 
            smresults['coefect'] = est.params[1:]
            smresults = smresults.sort_values('coefect', ascending=False).reset_index()

            celltype_genes[celltype] = pd.concat([smresults.head(200), smresults.tail(200)]).genes
    

celltype_genes.to_csv(f"{project_dir}/db/fetal_celltype_genes.txt", 
                      index=None, sep="\t")


