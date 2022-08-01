# --------------------------------
# setup
# --------------------------------

# libraries
library(dplyr)
library(stringr)
library(ggplot2)
library(conText)
library(quanteda)

# set working directory to the location of the master "EmbeddingRegressionReplication" folder
setwd("")

# --------------------------------
# load data
# --------------------------------

# new york times corpus
corpus_nyt <- readRDS("data/nyt_data.rds")
corpus_nyt$lead_paragraph[is.na(corpus_nyt$lead_paragraph)] <- corpus_nyt$snippet[is.na(corpus_nyt$lead_paragraph)] # in 2018 lead_paragraph is missing

# pre-trained embeddings & transformation matrix
pre_trained <- readRDS("data/stanford-glove/glove.rds")  # pre-trained
transform_matrix <- readRDS("data/stanford-glove/khodakA.rds")  # pre-trained

#---------------------------------
# pre-processing 
#---------------------------------

# keep only documents where the target words appear
trump_corpus <- corpus_nyt[grep('Trump', corpus_nyt$lead_paragraph, fixed = TRUE, ignore.case = FALSE), c('lead_paragraph', 'year')] %>% 
  distinct(lead_paragraph, .keep_all = TRUE) %>% 
  filter(year %in% c(2001:2014,2017:2020)) %>% 
  rename(text = lead_paragraph) %>% 
  mutate(year = as.integer(year),
         target = case_when(year>=2001 & year<=2014 ~ "celebrity",
                          year>=2017 & year<=2020 ~ "president"))

# basic preprocessing of text
trump_corpus$text <- trump_corpus$text %>%
  gsub('LEAD:', '', .) %>% # remove header banner
  gsub("[^[:alpha:]]", " ", .) %>% # remove all non-alpha characters
  str_replace_all("\\b\\w{1,2}\\b", "") %>% # remove 1-2 letter words
  str_replace_all("^ +| +$|( ) +", "\\1") %>% # remove excess white space
  tolower() # lowercase

# quanteda corpus
trump_corpus <- corpus(trump_corpus$text, docvars = trump_corpus[,c("year", "target")])

#---------------------------------
# alc embeddings
#---------------------------------

# tokenize corpus
toks <- tokens(trump_corpus, remove_punct = T, remove_symbols = T, remove_numbers = T, remove_separators = T)

# build a tokenized corpus of contexts sorrounding the target
trump_toks <- tokens_context(x = toks, pattern = 'trump', valuetype = "fixed", window = 6L, case_insensitive = TRUE, hard_cut = FALSE, verbose = FALSE)

# sample 500 instances of each sense
set.seed(2022L)
trump_toks_sample <- tokens_sample(x = trump_toks, size = 500, replace = FALSE, by = docvars(trump_toks,'target'))

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
celebrity_cluster <- which(as.vector(table(plot_tibble$target, plot_tibble$cluster)['celebrity',]) == max(table(plot_tibble$target, plot_tibble$cluster)['celebrity',]))
president_cluster <- which(as.vector(table(plot_tibble$target, plot_tibble$cluster)['president',]) == max(table(plot_tibble$target, plot_tibble$cluster)['president',]))
plot_tibble <- plot_tibble %>% mutate(cluster = if_else((target == 'celebrity' & cluster!=celebrity_cluster) | (target == 'president' & cluster!=president_cluster), 3L, cluster))
plot_tibble <- plot_tibble %>% mutate(target = if_else(cluster == 3L, 'misclassified', target))
plot_tibble <- plot_tibble %>% mutate(target = factor(target, levels = c('president', 'misclassified', 'celebrity')))

# --------------------------------
# plot
# --------------------------------
fgB1 <- ggplot(plot_tibble, aes(x = PC1, y = PC2, color = target, shape = target)) +  
  geom_point(size = 4) +
  geom_hline(yintercept = 0, linetype="dashed", color = "black", size = 0.5) + 
  geom_vline(xintercept = 0, linetype="dashed", color = "black", size = 0.5) +
  scale_colour_manual(labels = c('President \n Trump','misclassified', 'celebrity \n Trump'),
                      values = c("red","grey20", "blue")) +   
  scale_shape_manual(labels = c('President \n Trump','misclassified', 'celebrity \n Trump'),
                     values = c(19,4, 17)) +
  xlab('PC1') + ylab('PC2') +
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

ggsave(filename = "fgB1.pdf", plot = fgB1, height = 12, width = 12, path = './figures/', dpi = 1000)

# --------------------------------
# nearest neighbors
# --------------------------------

# build a document-feature-matrix with all instances
trump_dfm_all <- dfm(trump_toks, tolower = TRUE)

# embed each instance using ALC
trump_dem_all <- dem(x = trump_dfm_all, pre_trained = pre_trained, transform = TRUE, transform_matrix = transform_matrix, verbose = FALSE)

# aggregate by sense
trump_dem_all_agg <- dem_group(x = trump_dem_all, groups = trump_dem_all@docvars$target)

# nns
nns(trump_dem_all_agg, N = 10, candidates = trump_dem_all_agg@features, pre_trained = pre_trained, stem = FALSE, as_list = TRUE)

