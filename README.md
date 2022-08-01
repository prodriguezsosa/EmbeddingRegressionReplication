# Embedding Regression
This repository contains the replication materials for the article ["Embedding Regression: Models for Context-Specific Description and Inference"](https://github.com/prodriguezsosa/EmbeddingRegression), to be published in _The American Political Science Review_, by Pedro L. Rodriguez, Arthur Spirling and Brandon M. Stewart.

## Data

All necessary data and estimated models are available in [this Dropbox folder](https://www.dropbox.com/sh/7al371qtr9102qq/AADKhjhYgnFCxOOQaugQloTBa?dl=0). Keep in mind the folder is quite large at 12.59 GB.

## Required Software and Packages

```
R version 4.2.0 (2022-04-22)
Platform: x86_64-apple-darwin17.0 (64-bit)
Running under: macOS Monterey 12.4

Matrix products: default
LAPACK: /Library/Frameworks/R.framework/Versions/4.2/Resources/lib/libRlapack.dylib

locale:
[1] en_US.UTF-8/en_US.UTF-8/en_US.UTF-8/C/en_US.UTF-8/en_US.UTF-8

attached base packages:
[1] stats     graphics  grDevices utils     datasets  methods   base     

other attached packages:
 [1] SnowballC_0.7.0           hunspell_3.0.1            zoo_1.8-10                reticulate_1.25           quanteda.textmodels_0.9.4
 [6] cluster_2.1.3             text2vec_0.6.1            conText_1.3.1             tidyr_1.2.0               stargazer_5.2.3          
[11] ggplot2_3.3.6             progress_1.2.2            readtext_0.81             pbapply_1.5-0             quanteda_3.2.1           
[16] stringr_1.4.0             dplyr_1.0.9              

loaded via a namespace (and not attached):
 [1] httr_1.4.3             jsonlite_1.8.0         splines_4.2.0          foreach_1.5.2          RhpcBLASctl_0.21-247.1
 [6] RcppParallel_5.1.5     LiblineaR_2.10-12      assertthat_0.2.1       lgr_0.4.3              yaml_2.3.5            
[11] pillar_1.7.0           lattice_0.20-45        glue_1.6.2             digest_0.6.29          colorspace_2.0-3      
[16] htmltools_0.5.2        Matrix_1.4-1           plyr_1.8.7             rsparse_0.5.0          pkgconfig_2.0.3       
[21] SparseM_1.81           purrr_0.3.4            scales_1.2.0           tibble_3.1.7           generics_0.1.2        
[26] ellipsis_0.3.2         withr_2.5.0            cli_3.3.0              survival_3.3-1         magrittr_2.0.3        
[31] crayon_1.5.1           evaluate_0.15          float_0.3-0            stopwords_2.3          fansi_1.0.3           
[36] tools_4.2.0            data.table_1.14.2      prettyunits_1.1.1      hms_1.1.1              lifecycle_1.0.1       
[41] munsell_0.5.0          glmnet_4.1-4           compiler_4.2.0         mlapi_0.1.1            rlang_1.0.2           
[46] grid_4.2.0             iterators_1.0.14       rstudioapi_0.13        rmarkdown_2.14         gtable_0.3.0          
[51] codetools_0.2-18       DBI_1.1.2              reshape2_1.4.4         R6_2.5.1               knitr_1.39            
[56] fastmap_1.1.0          utf8_1.2.2             fastmatch_1.1-3        shape_1.4.6            stringi_1.7.6         
[61] parallel_4.2.0         Rcpp_1.0.8.3           vctrs_0.4.1            png_0.1-7              tidyselect_1.1.2      
[66] xfun_0.30  

```

`Python (3.7)`:
- gensim
- sentence-transformers

* [`conText`](https://github.com/prodriguezsosa/conText) is the R software associated with this paper. It is available on [CRAN](https://cran.r-project.org/web/packages/conText/index.html). For the latest development version use:

```r
devtools::install_github("prodriguezsosa/conText")
```

## Code

In the folder `/code` you will find an `.R` script for every figure and table both in the paper and the appendix. The script names match those of the corresponding figure/table.
