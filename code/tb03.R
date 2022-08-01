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

# build a document-feature-matrix
trump_dfm <- dfm(trump_toks, tolower = TRUE)

# embed each instance using ALC
trump_alc <- dem(x = trump_dfm, pre_trained = pre_trained, transform = TRUE, transform_matrix = transform_matrix, verbose = FALSE)

# embed each instance averaging embeddings (w/o applying transformation)
trump_embeds <- dem(x = trump_dfm, pre_trained = pre_trained, transform = FALSE, transform_matrix = NULL, verbose = FALSE)

# average single-instace alc embeddings by sense
trump_alc_agg <- dem_group(x = trump_alc, groups = trump_alc@docvars$target)

# average single-instace average embeddings by sense
trump_embeds_agg <- dem_group(x = trump_embeds, groups = trump_alc@docvars$target)

# --------------------------------
# nearest neighbors
# --------------------------------

# alc
nns(x = trump_alc_agg, N = 10, candidates = trump_alc_agg@features, pre_trained = pre_trained, stem = FALSE, as_list = TRUE)

# Embeddings
nns(x = trump_embeds_agg, N = 10, candidates = trump_embeds_agg@features, pre_trained = pre_trained, stem = FALSE, as_list = TRUE)



