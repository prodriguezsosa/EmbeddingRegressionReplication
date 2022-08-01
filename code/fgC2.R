# --------------------------------
# setup
# --------------------------------

# libraries
library(dplyr)
library(ggplot2)
library(utf8)
library(stringr)
library(quanteda)
library(conText)
library(text2vec)

# set working directory to the location of the master "EmbeddingRegressionReplication" folder
setwd("")

# --------------------------------
# load data
# --------------------------------
# load corpus
corpus <- readRDS('data/corpus_full_cr.rds')
text <- corpus %>% filter(period_start >= 2004) %>% .$speech
text <- utf8_encode(text) # fix encoding
rm(corpus)

# pre-trained embeddings & transformation matrix
pre_trained <- readRDS('data/word_vectors_cr.rds')
transform_matrix <- readRDS('data/local_transform_cr.rds')

#---------------------------------
# basic pre-processing
#---------------------------------
text <- text %>% 
  gsub("[^[:alpha:]]", " ", .) %>% # remove all non-alpha characters
  str_replace_all("\\b\\w{1,2}\\b", "") %>% # remove 1-2 letter words
  str_replace_all("^ +| +$|( ) +", "\\1") %>% # remove excess white space
  tolower() # lowercase

#---------------------------------
# define targets
#--------------------------------

# random set
tokens <- space_tokenizer(text)
it <- itoken(tokens, progressbar = TRUE)
vocab <- create_vocabulary(it)
vocab_pruned <- prune_vocabulary(vocab, term_count_min = 3000)
set.seed(2022L)
targets <- sample(vocab_pruned$term, 10, replace = FALSE)

# politics set
politics <- c("democracy", "freedom", "equality", "justice", "immigration", "abortion", "welfare", "taxes", "republican", "democrat")

# combine
targets <- c(targets, politics)

#---------------------------------
# ALC (samples of instances)
#--------------------------------

# tokenize corpus
toks <- tokens(text, remove_punct = T, remove_symbols = T, remove_numbers = T, remove_separators = T)

# build a tokenized corpus of contexts sorrounding the target
targets_toks <- tokens_context(x = toks, pattern = targets, valuetype = "fixed", window = 6L, case_insensitive = TRUE, hard_cut = FALSE, verbose = FALSE)

# build a document-feature-matrix
targets_dfm <- dfm(targets_toks, tolower = TRUE)

# embed each instance using ALC
targets_dem <- dem(x = targets_dfm, pre_trained = pre_trained, transform = TRUE, transform_matrix = transform_matrix, verbose = TRUE)

# function to bootstrap
# computes ALC embedding for each feature
# then the cosine similarity with the respective pretrained embedding
compute_alc <- function(x, ss){
  
  # sample dem 
  targets_sample <- dem_sample(x, size = ss, replace = TRUE, weight = NULL, by = x@docvars$pattern)
  
  # aggregate by target term (creating each target's ALC embedding)
  targets_sample_alc <- dem_group(x = targets_sample, groups = targets_sample@docvars$pattern)
  
  # compute cosine similartiy
  out <- feature_sim(x = targets_sample_alc, y = pre_trained, features = rownames(targets_sample_alc))
  
  # return output
  return(out)
}

# run bootstrapping
num_bootstraps <- 100
sample_size <- c(5, 25, 50, 100, 150, 200, 250, 500, 1000, 3000)
out_list <- vector("list", length(sample_size)) %>% setNames(as.character(sample_size))
set.seed(2022L)
for(ss in sample_size){
  bs_out <- replicate(num_bootstraps, compute_alc(targets_dem, ss), simplify = FALSE)
  out <- bind_rows(bs_out) %>% cbind(., "sample_size" = ss)
  out_list[[ss]] <- out
  cat("done with sample size", ss, "\n")
}

# row bind results
sample_tibble <- out_list %>% bind_rows()

#---------------------------------
# ALC (full set of instances)
#--------------------------------

compute_alc_full <- function(x, target){
  
  # sample dem 
  targets_sample <- dem_sample(x, size = nrow(x), replace = TRUE, weight = NULL, by = x@docvars$pattern)
  
  # aggregate by target term (creating each target's ALC embedding)
  targets_sample_alc <- dem_group(x = targets_sample, groups = targets_sample@docvars$pattern) 
  
  # compute cosine similartiy
  out <- unname(sim2(x = as.matrix(targets_sample_alc), y = matrix(pre_trained[target,], nrow = 1), norm = 'l2')[1,])
  
  # return output
  return(out)
}

full_out_list <- vector("list", length(targets)) %>% setNames(targets)
for(target in targets){
  targets_dem_sub <- dem(x = dfm_subset(targets_dfm, pattern == target), pre_trained = pre_trained, transform = TRUE, transform_matrix = transform_matrix, verbose = TRUE)
  bs_out <- replicate(100, compute_alc_full(x = targets_dem_sub, target = target))
  out <- tibble(feature = target, value = bs_out, sample_size = "max sample")
  full_out_list[[target]] <- out
  cat("done with sample target", target, "\n")
}

# row bind results
full_sample_tibble <- full_out_list %>% bind_rows()

#---------------------------------
# plot
#--------------------------------
plot_tibble <- rbind(sample_tibble, full_sample_tibble) %>% 
  mutate(sample_size = factor(sample_size, levels = c(5, 25, 50, 100, 150, 200, 250, 500, 1000, 3000, 'max sample')),
         type = ifelse(feature %in% politics, "politics", "random"))

# politics
fgC2a <- ggplot(subset(plot_tibble, type == "politics"), aes(x = sample_size, y = value)) + 
  geom_point(alpha = 1/100) +
  stat_summary(fun = 'mean', geom = "point", size = 4, shape = 17) +
  xlab('Sample size') + ylab('Cosine similarity') +
  scale_y_continuous(breaks = seq(0,1,0.1), limits = c(0,1)) +
  theme(axis.text.x = element_text(size=18),
        axis.text.y = element_text(size=18),
        axis.title.y = element_text(size=20, margin = margin(t = 0, r = 15, b = 0, l = 15)),
        axis.title.x = element_text(size=20, margin = margin(t = 15, r = 0, b = 15, l = 0)),
        legend.text=element_text(size=18),
        legend.title=element_blank(),
        legend.key=element_blank(),
        legend.position = "top",
        legend.spacing.x = unit(0.25, 'cm'))

ggsave(filename = "fgC2a.pdf", plot = fgC2a, height = 10, width = 12, path = './figures/', dpi = 1000)

# random
fgC2b <- ggplot(subset(plot_tibble, type == "random"), aes(x = sample_size, y = value)) + 
  geom_point(alpha = 1/100) +
  stat_summary(fun = 'mean', geom = "point", size = 4, shape = 17) +
  xlab('Sample size') + ylab('Cosine similarity') +
  scale_y_continuous(breaks = seq(0,1,0.1), limits = c(0,1)) +
  theme(axis.text.x = element_text(size=18),
        axis.text.y = element_text(size=18),
        axis.title.y = element_text(size=20, margin = margin(t = 0, r = 15, b = 0, l = 15)),
        axis.title.x = element_text(size=20, margin = margin(t = 15, r = 0, b = 15, l = 0)),
        legend.text=element_text(size=18),
        legend.title=element_blank(),
        legend.key=element_blank(),
        legend.position = "top",
        legend.spacing.x = unit(0.25, 'cm'))
  
ggsave(filename = "fgC2b.pdf", plot = fgC2b, height = 10, width = 12, path = './figures/', dpi = 1000)
