# --------------------------------
# setup
# --------------------------------

# libraries
library(dplyr)
library(quanteda)
library(conText)
library(ggplot2)
library(cluster)
library(progress)
library(tidyr)

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

# --------------------------------
# load data
# --------------------------------

# nyt corpus
corpus_nyt <- readRDS("data/fg03_corpus.rds")

# pre-trained embeddings & transformation matrix
pre_trained <- readRDS("data/stanford-glove/glove.rds")  # pre-trained
transform_matrix <- readRDS("data/stanford-glove/khodakA.rds")  # pre-trained

# --------------------------------
# tokenize
# --------------------------------

# tokenize corpus
toks_raw <- tokens(corpus_nyt, remove_punct = T, remove_symbols = T, remove_numbers = T, remove_separators = T)

# --------------------------------
# run experiments
# --------------------------------
out <- list()
set.seed(2022L)
for(w1 in c(2L, 6L, 12L)){
  for(rm_stopwords in c(FALSE, TRUE)){
    
    # start with raw tokens
    toks <- toks_raw
    
    # clean out stopwords
    if(rm_stopwords) toks <- tokens_select(toks, pattern = stopwords("en"), selection = "remove", case_insensitive = TRUE)
    
    # build a tokenized corpus of contexts sorrounding the target
    toks <- tokens_context(x = toks, pattern = c('trump', 'Trump'), valuetype = "fixed", window = w1, case_insensitive = FALSE, hard_cut = FALSE, verbose = FALSE)
    
    # --------------------------------
    # bootstrapping
    # --------------------------------
    num_bootstraps <- 100
    num_contexts <- min(table(docvars(toks)$target))
    out_bs <- vector("list", num_bootstraps)
    pb <- progress_bar$new(total = num_bootstraps)
    for(i in 1:num_bootstraps){
      
      # sample contexts
      trump_toks <- tokens_sample(tokens_subset(toks, target == "trump"), size = num_contexts, replace = TRUE)
      Trump_toks <- tokens_sample(tokens_subset(toks, target == "Trump"), size = num_contexts, replace = TRUE)
      
      # build a document-feature-matrix
      trump_dfm <- dfm(trump_toks, tolower = TRUE)
      Trump_dfm <- dfm(Trump_toks, tolower = TRUE)
      
      # build a document-embedding-matrix
      trump_dem <- dem(x = trump_dfm, pre_trained = pre_trained, transform = TRUE, transform_matrix = transform_matrix, verbose = FALSE)
      Trump_dem <- dem(x = Trump_dfm, pre_trained = pre_trained, transform = TRUE, transform_matrix = transform_matrix, verbose = FALSE)
      
      # prepare for clustering
      alc_embeddings <- rbind(trump_dem, Trump_dem)
      true_labels <- c(rep("trump", nrow(trump_dem)), rep("Trump", nrow(Trump_dem)))
      
      # force instances into 2 clusters
      alc_clusters <- kmeans(alc_embeddings, 2, nstart = 50)
      
      # confusion matrix
      confusion_matrix <- table(true_labels, unname(alc_clusters$cluster))
      out_bs[[i]] <- clustering_score(x = confusion_matrix)
      
      # clean up
      rm(trump_dem, trump_dfm, Trump_dem, Trump_dfm, trump_toks, Trump_toks)
      
      # progress bar
      pb$tick()
    }
    
    # results data.frame
    results_df <- bind_rows(out_bs) %>% mutate(rm_stopwords = rm_stopwords, window_size = w1)
    out[[length(out) + 1]] <- results_df
    cat("\n done with window size", w1, "and rm_stopwords", rm_stopwords, "\n")
  }
}

# save results
#saveRDS(out, 'data/fgF6_output.rds')

# --------------------------------
# plot
# --------------------------------

# read results
out <- readRDS('data/fgF6_output.rds')

# join results into one plot data.frame
out_df <- bind_rows(out) %>% 
  pivot_longer(c("v-measure", "homogeneity", "completeness"), names_to = "metric") %>% # pivot to long form
  mutate(metric = factor(metric, levels = c("v-measure", "homogeneity", "completeness")),
         rm_stopwords = if_else(rm_stopwords, "rm_stopwords = TRUE", "rm_stopwords = FALSE")) # order factor for nice plotting

out_upper_ci <- out_df %>% group_by(metric, rm_stopwords, window_size) %>% arrange(value) %>% slice(ceiling(0.95*num_bootstraps)) %>% ungroup() %>% rename(ci_upper = value)
out_lower_ci <- out_df %>% group_by(metric, rm_stopwords, window_size) %>% arrange(value) %>% slice(ceiling(0.05*num_bootstraps)) %>% ungroup() %>% rename(ci_lower = value)
out_mean <- out_df %>% group_by(metric, rm_stopwords, window_size) %>% summarize(value = mean(value), .groups = "drop_last") %>% ungroup()
plot_df <- left_join(out_mean, out_upper_ci, by = c("metric", "rm_stopwords", "window_size"))
plot_df <- left_join(plot_df, out_lower_ci, by = c("metric", "rm_stopwords", "window_size"))

# plot
fgF6 <- ggplot(plot_df, aes(x = as.factor(window_size), y = value, group = metric, shape = metric, color = metric)) + 
  geom_point(size = 2, position = position_dodge(width = 1/2)) +
  geom_pointrange(aes(ymin = ci_lower, ymax = ci_upper), size = 1, position = position_dodge(width = 1/2)) +
  scale_colour_manual(labels = c('V-measure', 'Homogeneity', 'Completeness'), values = c("gray60", "grey30", "black")) +   
  scale_shape_manual(labels = c('V-measure', 'Homogeneity', 'Completeness'), values = c(19, 4, 17)) +
  facet_wrap(~rm_stopwords) +
  xlab("Window size") + 
  ylim(0,0.5) +
  theme(axis.text.x = element_text(size=18, vjust = 0.5),
        axis.text.y = element_text(size=18),
        axis.title.y = element_blank(),
        axis.title.x = element_text(size=20, vjust = -2),
        legend.text=element_text(size=18),
        legend.title=element_blank(),
        legend.key=element_blank(),
        legend.position = "top",
        legend.spacing.x = unit(0.5, 'cm'),
        strip.text = element_text(size = 24),
        plot.margin = unit(c(1,1,1,1), "cm"))

ggsave(filename = "fgF6.pdf", plot = fgF6, height = 10, width = 12, path = './figures/', dpi = 1000)

