# --------------------------------
# setup
# --------------------------------

# libraries
library(dplyr)
library(ggplot2)
library(quanteda)
library(conText)
library(cluster)

# set working directory to the location of the master "EmbeddingRegressionReplication" folder
setwd("")

# --------------------------------
# load data
# --------------------------------

# nyt corpus
corpus_nyt <- readRDS("data/fg03_corpus.rds")

# pre-trained embeddings & transformation matrix
pre_trained <- readRDS("data/stanford-glove/glove.rds")  # pre-trained
transform_matrix <- readRDS("data/stanford-glove/khodakA.rds")  # pre-trained

# --------------------------------
# ALC embed
# --------------------------------

# tokenize corpus
toks <- tokens(corpus_nyt, remove_punct = T, remove_symbols = T, remove_numbers = T, remove_separators = T)

# build a tokenized corpus of contexts sorrounding the target
trump_toks <- tokens_context(x = toks, pattern = c('trump', 'Trump'), valuetype = "fixed", window = 6L, case_insensitive = FALSE, hard_cut = TRUE, verbose = FALSE)
trump_toks <- tokens_subset(trump_toks, target == pattern) # there is one instance where both Trump and trump appear, hence target!=pattern

# sample 400 instances of each sense (there are 403 instances of trump)
set.seed(2022L)
trump_toks_sample <- tokens_sample(x = trump_toks, size = 400, replace = FALSE, by = docvars(trump_toks,'target'))

# build a document-feature-matrix
trump_dfm <- dfm(trump_toks_sample, tolower = TRUE)

# embed each instance using ALC
trump_dem <- dem(x = trump_dfm, pre_trained = pre_trained, transform = TRUE, transform_matrix = transform_matrix, verbose = FALSE)

# force individual data points into two clusters
trump_clusters <- kmeans(trump_dem, 2, nstart = 100)

# find principal components using pca
trump_pca <- prcomp(trump_dem, scale = TRUE)

# first two pcs
ind_coord <- as_tibble(trump_pca$x[,1:2])

# tibble for plotting
plot_tibble <- tibble(doc_id = 1:length(trump_toks_sample), ind_coord, text = unname(sapply(trump_toks_sample, function(i) paste0(i, collapse = " "))), target = docvars(trump_toks_sample, 'target'), cluster = unname(trump_clusters$cluster))

# identify majority label in each cluster and misclassification
trump_cluster <- which(as.vector(table(plot_tibble$target, plot_tibble$cluster)['trump',]) == max(table(plot_tibble$target, plot_tibble$cluster)['trump',]))
Trump_cluster <- which(as.vector(table(plot_tibble$target, plot_tibble$cluster)['Trump',]) == max(table(plot_tibble$target, plot_tibble$cluster)['Trump',]))
plot_tibble <- plot_tibble %>% mutate(cluster = if_else((target == 'trump' & cluster!=trump_cluster) | (target == 'Trump' & cluster!=Trump_cluster), 3L, cluster))
plot_tibble <- plot_tibble %>% mutate(target = if_else(cluster == 3L, 'misclassified', target))
plot_tibble <- plot_tibble %>% mutate(target = factor(target, levels = c('trump', 'misclassified', 'Trump')))

# --------------------------------
# plot
# --------------------------------
fg2 <- ggplot(plot_tibble, aes(x = PC1, y = PC2, color = target, shape = target)) +  
  geom_point(size = 4) +
  geom_hline(yintercept = 0, linetype="dashed", color = "black", size = 0.5) + 
  geom_vline(xintercept = 0, linetype="dashed", color = "black", size = 0.5) +
  scale_colour_manual(labels = c('Trump', 'misclassified', 'trump'),
                      values = c("red", "grey20", "blue")) +   
  scale_shape_manual(labels = c('Trump', 'misclassified', 'trump'),
                     values = c(19, 4, 17)) +
  xlab('PC1') + ylab('PC2') +
  #xlim(-12, 12) + ylim(-10,10) +
  theme(panel.background = element_blank(),
        axis.text.x = element_text(size=18),
        axis.text.y = element_text(size=18),
        axis.title.y = element_text(size=20, margin = margin(t = 0, r = 15, b = 0, l = 15)),
        axis.title.x = element_text(size=20, margin = margin(t = 15, r = 0, b = 15, l = 0)),
        legend.text=element_text(size=18),
        legend.title=element_blank(),
        legend.key=element_blank(),
        legend.position = "top",
        legend.spacing.x = unit(0.25, 'cm'))

ggsave(filename = "fg02.pdf", plot = fg2, height = 12, width = 12, path = './figures/', dpi = 1000)
