# --------------------------------
# setup
# --------------------------------

# libraries
library(dplyr)
library(progress)
library(quanteda.textmodels)
library(quanteda)
library(conText)
library(reticulate)
library(cluster)
library(ggplot2)

# set working directory to the location of the master "EmbeddingRegressionReplication" folder
setwd("")

# --------------------------------
# functions
# --------------------------------
# formulas: https://scikit-learn.org/stable/modules/clustering.html#clustering-performance-evaluation

# conditional entropy
conditional_entropy <- function(x, row = TRUE){
  n <- sum(x)
  if(!row) x <- t(x)
  out <- c()
  for(i in 1:nrow(x)){
    for(j in 1:ncol(x)){
      n_rc <- x[i,j]
      n_c <- sum(x[,j])
      out <- append(out, (n_rc/n) * log(n_rc/n_c))
    }
  }
  
  # replace NaNs with 0
  out[is.nan(out)] <- 0
  return(-1*sum(out))
}

#  entropy
entropy <- function(x, row = TRUE){
  n <- sum(x)
  if(!row) x <- t(x)
  out <- c()
  for(i in 1:nrow(x)){
    n_r <- sum(x[i,])
    out <- append(out, (n_r/n) * log((n_r/n)))
  }
  return(-1*sum(out))
}

# v-score, homogeneity and completeness
clustering_score <- function(x){
  
  # compute elements
  class_ce <- conditional_entropy(x, row = TRUE)
  cluster_ce <- conditional_entropy(x, row = FALSE)
  class_e <- entropy(x, row = TRUE)
  cluster_e <- entropy(x, row = FALSE)
  
  # homogeneity
  homogeneity <- 1 - class_ce/class_e

  # completeness
  completeness <- 1 - cluster_ce/cluster_e

  # v-score
  vscore = 2 * ((homogeneity * completeness)/(homogeneity + completeness))
  if(is.nan(vscore)) vscore <- 0 # if divide by 0, useless classifier
  
  return(list("v-measure" = vscore, "homogeneity" = homogeneity, "completeness" = completeness))
}

# other performance metrics
performance_metrics <- function(confusion_matrix){
  
  # assign cluster name of majority class
  colnames(confusion_matrix)[1] <- names(which.max(confusion_matrix[,1]))
  colnames(confusion_matrix)[2] <- setdiff(rownames(confusion_matrix), colnames(confusion_matrix)[1])
  confusion_matrix <- confusion_matrix[colnames(confusion_matrix), colnames(confusion_matrix)] # make sure order is correct

  # accuracy: TP/(TP + FP + TN + FN)
  accuracy <- sum(diag(confusion_matrix))/sum(confusion_matrix)
  
  # class weight
  weight1 <- sum(confusion_matrix[1,])/sum(confusion_matrix)
  weight2 <- sum(confusion_matrix[2,])/sum(confusion_matrix)
  
  # precision: TP/(TP + FP)
  precision1 <- confusion_matrix[1,1]/(confusion_matrix[1,1] + confusion_matrix[2,1])
  precision2 <- confusion_matrix[2,2]/(confusion_matrix[2,2] + confusion_matrix[1,2])
  precision <- weight1*precision1 + weight2*precision2
    
  # recall: TP/(TP + FN)
  recall1 <- confusion_matrix[1,1]/(confusion_matrix[1,1] + confusion_matrix[1,2])
  recall2 <- confusion_matrix[2,2]/(confusion_matrix[2,2] + confusion_matrix[2,1])
  recall <- weight1*recall1 + weight2*recall2

  # F1
  f1 <- weight1*2*(precision1*recall1/(precision1 + recall1)) + weight2*2*(precision2*recall2/(precision2 + recall2))
  if(is.nan(f1)) f1 <- 0
  
  # output
  return(list(accuracy = accuracy, precision = precision, recall = recall, f1 = f1))
}

# --------------------------------
# load data
# --------------------------------

# nyt corpus
corpus_nyt <- readRDS("data/fg03_corpus.rds")

# pre-trained embeddings & transformation matrix
pre_trained <- readRDS("data/stanford-glove/glove.rds")  # pre-trained
transform_matrix <- readRDS("data/stanford-glove/khodakA.rds")  # pre-trained

#-----------------------
# load python modules
#-----------------------
use_virtualenv("./python/alacarte")
sentence_transformers <- import("sentence_transformers")

#-----------------------
# load pre-trained model
# see: https://github.com/UKPLab/sentence-transformers for other model options
#-----------------------
# larger model: 'roberta-large-nli-stsb-mean-tokens'
embedder = sentence_transformers$SentenceTransformer('all-distilroberta-v1')

# --------------------------------
# tokenize
# --------------------------------

# tokenize corpus
toks <- tokens(corpus_nyt, remove_punct = T, remove_symbols = T, remove_numbers = T, remove_separators = T)

# build a tokenized corpus of contexts sorrounding the target
toks_target <- tokens_context(x = toks, pattern = c('trump', 'Trump'), valuetype = "fixed", window = 6L, case_insensitive = FALSE, hard_cut = TRUE, verbose = FALSE)
toks_target <- tokens_subset(toks_target, target == pattern)

#---------------------------------
# bootstrap regressions
#---------------------------------
bs_samples <- 100
out_list <- list()
bs <- 1
pb <- progress_bar$new(total = bs_samples)
set.seed(1984L)
while(bs <= bs_samples){
  
  # sample contexts
  toks_target_ss <- tokens_sample(toks_target, size = min(table(docvars(toks_target)$target)), replace = TRUE, by = target)

  # untokenize for bert
  raw_contexts <- tibble(docid = docid(toks_target_ss), "pre" = tolower(unname(sapply(toks_target_ss, function(x) paste(paste(x[1:6], sep = " ", collapse = " "))))), pattern = tolower(as.character(docvars(toks_target_ss)$pattern)), "post" = tolower(unname(sapply(toks_target_ss, function(x) paste(x[7:12], sep = " ", collapse = " ")))))
  raw_contexts <- raw_contexts %>% tidyr::unite("context", pre:post, sep = ' ', remove = TRUE)                
  
  # build a document-feature-matrix
  toks_dfm <- dfm(toks_target_ss, tolower = TRUE)
  
  # embed each instance using lsa, embeddings, ALC & Roberta
  wv_lsa <- textmodel_lsa(dfm_tfidf(toks_dfm), nd = 2)
  wv_embeds <- dem(x = toks_dfm, pre_trained = pre_trained, transform = FALSE, transform_matrix = NULL, verbose = FALSE)
  wv_alc <- dem(x = toks_dfm, pre_trained = pre_trained, transform = TRUE, transform_matrix = transform_matrix, verbose = FALSE)
  wv_bert <- embedder$encode(raw_contexts$context, show_progress_bar = FALSE)
  
  # force individual data points into two clusters
  clusters_list <- vector('list', 4) %>% setNames(c('lsa', 'embeddings', 'bert', 'alc'))
  clusters_list[['lsa']] <- kmeans(wv_lsa$docs, 2, nstart = 100)
  clusters_list[['embeddings']]  <- kmeans(wv_embeds, 2, nstart = 100)
  clusters_list[['bert']]  <- kmeans(wv_bert, 2, nstart = 100)
  clusters_list[['alc']]  <- kmeans(wv_alc, 2, nstart = 100)
  
  # confusion matrix
  true_labels <- docvars(toks_target_ss, 'pattern')
  clusters_out <- lapply(c('lsa', 'embeddings', 'bert', 'alc'), function(algo){
    confusion_matrix <- table(true_labels, clusters_list[[algo]]$cluster, dnn = c('true_labels', 'predicted_labels'))
    clustering_scores <- clustering_score(x = confusion_matrix) %>% unlist()
    performance_scores <- performance_metrics(confusion_matrix) %>% unlist()
    return(tibble(algorithm = algo, metric = c(names(clustering_scores), names(performance_scores)), score = c(unname(clustering_scores),unname(performance_scores))))
  })
  
  # output
  out_list <- append(out_list, list(clusters_out))
  
  # update count
  pb$tick()
  bs <- bs + 1
}

# save results
#saveRDS(out_list, 'data/fg03_output.rds')

#---------------------------------
# visualize
#---------------------------------

# read results
out_list <- readRDS('data/fg03_output.rds')

cluster_tibble <-  out_list %>% bind_rows() %>%
  mutate(algorithm = factor(algorithm, levels = c("lsa", "embeddings", "bert", "alc")),
         metric = factor(metric, levels = c("v-measure", "homogeneity", "completeness", "accuracy", "precision", "recall", "f1" )))

# figure 3
fg3 <- ggplot(subset(cluster_tibble, metric == 'homogeneity'), aes(x = algorithm, y = score)) + 
  geom_point(alpha = 1/8, size = 2) +
  stat_summary(geom = "point", fun = "mean", col = "black", size = 3, shape = 17) +
  ylab('Cluster Homogeneity') +
  theme(axis.text.x = element_text(size=18, hjust = 1, vjust = 0.5, angle = 90),
        axis.text.y = element_text(size=18),
        axis.title.y = element_text(size=20, vjust = 2),
        axis.title.x = element_blank(),
        plot.margin = unit(c(1,1,1,1), "cm"))


ggsave(filename = "fg03.pdf", plot = fg3, height = 10, width = 12, path = './figures/', dpi = 1000)

# other metrics
ggplot(cluster_tibble, aes(x = algorithm, y = score)) + 
  geom_point(alpha = 1/4, size = 2) +
  stat_summary(geom = "point", fun = "mean", col = "black", size = 3, shape = 17) +
  ylim(0,1) +
  facet_wrap(~ metric, ncol = 3) +
  theme(axis.text.x = element_text(size=24, hjust = 1, vjust = 0.5, angle = 90),
        axis.text.y = element_text(size=24),
        axis.title.y = element_blank(),
        axis.title.x = element_blank(),
        strip.text = element_text(size = 20),
        plot.margin = unit(c(1,1,1,1), "cm"))






