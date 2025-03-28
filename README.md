# HEOCA: Human Endoderm-derived Organoids Cell Atlas

## Integrated human cell atlas of endoderm-derived organoids 
Organoids of the endoderm can recapitulate aspects of developing and adult human physiology. Organoids derived from embryonic or induced pluripotent stem cells model development and are guided to specific tissue types via morphogens, whereas organoids derived from tissue-resident fetal or adult stem cells are organ-identity-determined and may model ontogenetic features. However, it has remained difficult to assess the similarity and differences between organoid protocols, and to understand the precision and accuracy of organoid cell states through comparison with primary counterparts. Advances in computational single-cell biology allow the comprehensive integration of datasets with high technical variability. Here, we integrate published single-cell transcriptome datasets from organoids of diverse endoderm-derived tissues including lung, pancreas, intestine, salivary glands, liver, biliary system, stomach, and prostate to establish an initial version of a human endoderm organoid cell atlas (HEOCA). The integration includes nearly one million cells across diverse conditions and data sources. We align and compare cell types and states between organoid models, and harmonize cell type annotations by mapping the atlas to primary tissue counterparts. We focus on intestine and lung, and clarify developmental and adult physiology that can be modeled in vitro. We provide examples of data incorporation from new organoid protocols to expand the atlas, and showcase how comparison to the atlas can illuminate interesting biological features of new datasets. We also show that mapping disease organoid single-cell samples to HEOCA identifies shifts in cell proportion and gene expressions between normal and diseased cells. Taken together, the atlas makes diverse datasets centrally available (https://cellxgene.cziscience.com/), and it will be useful to assess organoid fidelity, characterize perturbed and diseased states, streamline protocol development, and will continuously grow in the future. 

![](figures/fig1.png)

## Cell browser


<table border="0">
 <tr>
    <td align="center"><a href="https://cellxgene.cziscience.com/e/6725ee8e-ef5b-4e68-8901-61bd14a1fe73.cxg/" </a> <b style="font-size:20px">HEOCA(Endoderm)</b></td>
    <td align="center"><a href="https://cellxgene.cziscience.com/e/776a1e4a-f141-49bb-9978-d0588a4cee9f.cxg/" </a> <b style="font-size:20px">HEOCA(Intestine)</b></td>
    <td align="center"><a href="https://cellxgene.cziscience.com/e/569bce19-14c3-436f-bebf-543e5ea025dc.cxg/" </a> <b style="font-size:20px">HEOCA(Lung)</b></td>
 </tr>
</table>

 
## Key methods
* [snapspeed (hierarchy cell type annotation)](https://github.com/devsystemslab/snapseed)
* [sc2heoca (query new organoid scRNA-seq data to HEOCA)](https://github.com/devsystemslab/sc2heoca)

## Analytic reproducibility
* [Code for integration](https://github.com/devsystemslab/HEOCA/tree/main/scripts)
* [Code for plot](https://github.com/devsystemslab/HEOCA/tree/main/notebooks)

## Citation
* Quan Xu, ..., Gray Camp. Integrated transcriptomic cell atlas of human endoderm-derived organoids. 2023.

## Help and support
* The preferred way to get support is through the [Github issues page](https://github.com/devsystemslab/HEOCA/issues).

## License
- **[MIT license](http://opensource.org/licenses/mit-license.php)** 
- Copyright 2023 © <a href="https://github.com/devsystemslab" target="_blank">devsystemslab</a>.

