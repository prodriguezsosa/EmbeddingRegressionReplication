#-----------------------
#
# SETUP
#
#-----------------------

# libraries
library(dplyr)
library(ggplot2)
library(stringr)
library(quanteda)
library(pbapply)
library(conText)
library(text2vec)

# set working directory to the location of the master "EmbeddingRegressionReplication" folder
setwd("")

# pre-trained embeddings & transformation matrix
pre_trained <- readRDS("data/stanford-glove/glove.rds")  # pre-trained
transform_matrix <- readRDS("data/stanford-glove/khodakA.rds")  # pre-trained

# load full embedding models
sample_sizes <- c(5, 10, 15, 25, 50, 100, 200, 'full')
wvs_list <- lapply(sample_sizes, function(i) readRDS(paste0('data/benchmarks/word_vectors_', i, '.rds'))) %>% setNames(sample_sizes)

# load corpora
corpus_list <- readRDS('data/benchmarks/corpus_list.rds')

# targets
targets <- c('immigration', 'economy', 'climatechange', 'healthcare', 'middleeast', 'floor')

#-----------------------
# GLOVE
#-----------------------
process_glove <- function(wvs, target){

  # targets
  target1 <- paste0(target, 'r')
  target2 <- paste0(target, 'd')
  
  # target embeddings
  target_embedding1 <- matrix(wvs[target1,], nrow = 1)
  target_embedding2 <- matrix(wvs[target2,], nrow = 1)
  
  # remove targets from wvs
  wvs <- wvs[-which(rownames(wvs) == target1),]
  wvs <- wvs[-which(rownames(wvs) == target2),]
  
  # cosine similarity
  cos_sim <- sim2(target_embedding1, target_embedding2, method = 'cosine', norm = 'l2')[1,]
  cos_sim <- tibble(target = target, cos_sim = cos_sim)
  
  # output
  return(cos_sim)
}

#-----------------------
# ALC
#-----------------------
process_alc <- function(toks, target, pre_trained, transform_matrix){
  
  # run conText regression 
  model1 <- conText(formula = as.formula(paste0(target, ' ~ party')), data = toks, pre_trained = pre_trained, transform = TRUE, transform_matrix = transform_matrix, bootstrap = TRUE, num_bootstraps = 100, confidence_level = 0.95, stratify = TRUE, permute = TRUE, num_permutations = 100, window = 6, valuetype = "fixed", case_insensitive = TRUE, hard_cut = FALSE, verbose = TRUE)
  
  # output
  return(model1@normed_coefficients)
}

#-----------------------
#
# FULL CORPUS ANALYSIS
#
#-----------------------

# GLOVE
glove_full <- pblapply(targets, function(target) process_glove(wvs = wvs_list[['full']], target))

# extract elements
cos_sim_glove <- bind_rows(glove_full)
cos_sim_glove <- cos_sim_glove %>% 
  mutate(target = recode(target, "middleeast" = "Middle East", "healthcare" = "Health Care", "climatechange" = "Climate Change")) %>%
  arrange(cos_sim) %>% mutate(target = factor(target, levels = target))

# ALC
set.seed(2022L)
corpus_df <- corpus_list[['full']]
corpus <- corpus(corpus_df$text, docvars = data.frame(party = corpus_df$party))
toks <- tokens(corpus, remove_punct = T, remove_symbols = T, remove_numbers = T, remove_separators = T)
alc_full <- lapply(targets, function(target) process_alc(toks = toks, target, pre_trained, transform_matrix))

# extract elements
normed_betas_alc <- alc_full %>% bind_rows() %>% cbind(., target = targets) %>% 
  mutate(target = recode(target, "middleeast" = "Middle East", "healthcare" = "Health Care", "climatechange" = "Climate Change")) %>%
  arrange(-normed.estimate) %>% mutate(target = factor(target, levels = target))

#-----------------------
# VISUALIZE
#-----------------------

# PARTISAN DIFFERENCE

# glove
fgD3a <- ggplot(cos_sim_glove, aes(x = target, y = 1 - cos_sim)) + 
  geom_bar(position = position_dodge(), stat="identity", alpha = 0.6) +
  ylab('Cosine distance') +
  ylim(0,1) +
  theme(panel.background = element_blank(),
        axis.ticks.x = element_blank(),
        axis.text.x = element_text(size=18, vjust = 0.5, hjust = 1, margin = margin(t = 0, r = 0, b = 15, l = 0), angle = 90),
        axis.text.y = element_text(size=18),
        axis.title.y = element_text(size=20, margin = margin(t = 0, r = 15, b = 0, l = 15)),
        axis.title.x = element_blank(),
        plot.margin=unit(c(1,1,0,0),"cm"))

ggsave(filename = "fgD3a.pdf", plot = fgD3a, height = 10, width = 12, path = './figures/', dpi = 1000)

# alc
fgD3b <- ggplot(normed_betas_alc, aes(x = target, y = normed.estimate)) + 
  geom_bar(position=position_dodge(), stat="identity", alpha = 0.6) +
  geom_errorbar(aes(x = target, y = normed.estimate, 
                      ymin = lower.ci,
                      ymax = upper.ci), width=.2, lwd = 0.75, position=position_dodge(.9)) + 
  ylab('Norm of party coefficient') +
  #ylim(0,0.04) +
  geom_text(aes(label=c('***', '***', '***', '***', '***', '***')), position=position_dodge(width=0.9), hjust=0.5, vjust = normed_betas_alc$upper.ci-2, size = 8) + # based on the empirical p-values
  theme(panel.background = element_blank(),
        axis.ticks.x = element_blank(),
        axis.text.x = element_text(size=18, vjust = 0.5, hjust = 1, margin = margin(t = 0, r = 0, b = 15, l = 0), angle = 90),
        axis.text.y = element_text(size=18),
        axis.title.y = element_text(size=20, margin = margin(t = 0, r = 15, b = 0, l = 15)),
        axis.title.x = element_blank(),
        plot.margin=unit(c(1,1,0,0),"cm"))

ggsave(filename = "fgD3b.pdf", plot = fgD3b, height = 10, width = 12, path = './figures/', dpi = 1000)

#-----------------------
#
# 5 DOC CORPUS ANALYSIS
#
#-----------------------

# GLOVE
glove_5 <- pblapply(targets, function(target) process_glove(wvs = wvs_list[['5']], target))

# extract elements
cos_sim_glove_5 <- bind_rows(glove_5)
cos_sim_glove_5 <- cos_sim_glove_5 %>% 
  mutate(target = recode(target, "middleeast" = "Middle East", "healthcare" = "Health Care", "climatechange" = "Climate Change")) %>%
  arrange(cos_sim) %>% mutate(target = factor(target, levels = target))

# ALC
set.seed(2022L)
corpus_5_df <- corpus_list[['5']]
corpus_5 <- corpus(corpus_5_df$text, docvars = data.frame(party = corpus_5_df$party))
toks_5 <- tokens(corpus_5, remove_punct = T, remove_symbols = T, remove_numbers = T, remove_separators = T)
alc_5 <- lapply(targets, function(target) process_alc(toks = toks_5, target, pre_trained, transform_matrix))

# extract elements
normed_betas_alc_5 <- alc_5 %>% bind_rows() %>% cbind(., target = targets) %>% 
  mutate(target = recode(target, "middleeast" = "Middle East", "healthcare" = "Health Care", "climatechange" = "Climate Change")) %>%
  arrange(-normed.estimate) %>% mutate(target = factor(target, levels = target))

#-----------------------
# VISUALIZE
#-----------------------

# PARTISAN DIFFERENCE

# glove
fgD4a <- ggplot(cos_sim_glove_5, aes(x = target, y = 1 - cos_sim)) + 
  geom_bar(position = position_dodge(), stat="identity", alpha = 0.6) +
  ylab('Cosine distance') +
  ylim(0,1) +
  theme(panel.background = element_blank(),
        axis.ticks.x = element_blank(),
        axis.text.x = element_text(size=18, vjust = 0.5, hjust = 1, margin = margin(t = 0, r = 0, b = 15, l = 0), angle = 90),
        axis.text.y = element_text(size=18),
        axis.title.y = element_text(size=20, margin = margin(t = 0, r = 15, b = 0, l = 15)),
        axis.title.x = element_blank(),
        plot.margin=unit(c(1,1,0,0),"cm"))

ggsave(filename = "fgD4a.pdf", plot = fgD4a, height = 10, width = 12, path = './figures/', dpi = 1000)

# alc
fgD4b <- ggplot(normed_betas_alc_5, aes(x = target, y = normed.estimate)) + 
  geom_bar(position=position_dodge(), stat="identity", alpha = 0.6) +
  geom_errorbar(aes(x = target, y = normed.estimate, 
                    ymin = lower.ci,
                    ymax = upper.ci), width=.2, lwd = 0.75, position=position_dodge(.9)) + 
  ylab('Norm of party coefficient') +
  geom_text(aes(label=c('**', '', '', '', '**', '**')), position=position_dodge(width=0.9), hjust=0.5, vjust = normed_betas_alc_5$upper.ci-7, size = 8) + # based on the empirical p-values
  theme(panel.background = element_blank(),
        axis.ticks.x = element_blank(),
        axis.text.x = element_text(size=18, vjust = 0.5, hjust = 1, margin = margin(t = 0, r = 0, b = 15, l = 0), angle = 90),
        axis.text.y = element_text(size=18),
        axis.title.y = element_text(size=20, margin = margin(t = 0, r = 15, b = 0, l = 15)),
        axis.title.x = element_blank(),
        plot.margin=unit(c(1,1,0,0),"cm"))

ggsave(filename = "fgD4b.pdf", plot = fgD4b, height = 10, width = 12, path = './figures/', dpi = 1000)

