# --------------------------------
# setup
# --------------------------------
library(dplyr)
library(quanteda)
library(conText)
library(text2vec)
library(ggplot2)
library(SnowballC)

# set working directory to the location of the master "EmbeddingRegressionReplication" folder
setwd("")

# --------------------------------
# load corpus
# --------------------------------
cr <- readRDS("data/corpus_daily.rds") %>% 
  select(speech, party, session_id) %>% 
  filter(session_id %in% 107:114 & party %in% c('D', 'R')) %>%
  mutate(speech = tolower(iconv(speech, from = "latin1", to = "UTF-8"))) %>%
  distinct(speech, .keep_all = TRUE) %>% tidyr::drop_na() %>% 
  select(speech, party, session_id) %>% tidyr::unite(c(party, session_id), col = "group", sep = '-', remove = FALSE)

# quanteda corpus
cr_corpus <- corpus(cr$speech, docvars = data.frame(party = cr$party, group = cr$group))

# define target terms
targets <- c("democracy", "freedom", "equality", "justice", "immigration", "abortion", "welfare", "taxes", "republican", "democrat")

# --------------------------------
# pre-processing
# --------------------------------
cr_toks_base <- tokens(cr_corpus, remove_punct = T, remove_symbols = T, remove_numbers = T, remove_url = T, remove_separators = T)
feats <- dfm(cr_toks_base, verbose = TRUE) %>% dfm_trim(min_termfreq = 10) %>% featnames() # min_termfreq set to same value as used for training

# subset tokens object to selected features
cr_toks_base <- tokens_select(cr_toks_base, pattern = feats, selection = "keep", padding = TRUE, min_nchar = 3)

# feature counts
feat_count <- featfreq(dfm(cr_toks_base, verbose = TRUE))
feat_count <- tibble(feature = names(feat_count), n = unname(feat_count)) %>% arrange(-n)
feat_count[feat_count$feature %in% targets,]

# overlap in features
glove_feats <- rownames(readRDS("data/stanford-glove/glove.rds"))
length(intersect(feat_count$feature, glove_feats))/length(feats)

# --------------------------------
#
#
# run experiments
#
#
# --------------------------------

# preliminaries
out1 <- list()
toks_fixed <- FALSE
sample_contexts <- FALSE
set.seed(1984L)
# --------------------------------
# define pre-trained embeddings
# --------------------------------
for(target in targets){
  for(w1 in c(2L,6L,12L, "glove")){
    if(w1 == "glove"){
      pre_trained <- readRDS("data/stanford-glove/glove.rds")
      transform_matrix <- readRDS("data/cr_experiments/cr.am.glove.300d.rds")
    }else{
      pre_trained <- readRDS("data/cr_experiments/cr.wvs.", w1, "w.300d.rds")
      transform_matrix <- readRDS("data/cr_experiments/cr.am.", w1, "w.300d.rds")
    }
    # --------------------------------
    # build contexts
    # --------------------------------
    for(w2 in c(2L,6L,12L)){
      for(rstw in c(FALSE, TRUE)){
        
        # reset toks
        cr_toks <- cr_toks_base
        
        # clean out stopwords
        if(rstw) cr_toks <- tokens_select(cr_toks, pattern = stopwords("en"), selection = "remove", case_insensitive = TRUE, padding = FALSE)
        
        # select contexts (tokens) around target
        cr_toks <- tokens_context(x = cr_toks, pattern = target, window = w2, valuetype = "fixed", case_insensitive = TRUE, hard_cut = FALSE, verbose = TRUE)
        
        # sample same number of contexts
        if(sample_contexts) cr_toks <- tokens_sample(cr_toks, size = 500, replace = TRUE)
        
        # toks_fixed (i.e. hold number of tokens fixed), sample min number of tokens (2*2)
        if(toks_fixed){
          docvars <- docvars(cr_toks)
          cr_toks <- lapply(cr_toks, function(i) sample(i, 4, replace = TRUE)) %>% as.tokens()
          docvars(cr_toks) <- docvars
        }
        
        # --------------------------------
        # compute ALC embeddings
        # --------------------------------
        
        # build a document-feature-matrix
        cr_dfm <- dfm(cr_toks)
        
        # build a document-embedding-matrix
        cr_dem <- dem(x = cr_dfm, pre_trained = pre_trained, transform = TRUE, transform_matrix = transform_matrix, verbose = FALSE)
        
        # average dem by group
        cr_dem_grouped <- dem_group(x = cr_dem, groups = cr_dem@docvars$group)
        
        # average dem
        cr_dem_avg <- matrix(colMeans(cr_dem), nrow = 1)
        
        # --------------------------------
        # add stemming
        # --------------------------------
        for(stm in c(FALSE, TRUE)){
          # --------------------------------
          # performance metrics
          # --------------------------------
          
          for(nrm in c("none", "l2")){
            
            # nearest neighbors
            if(stm){
              spellcheck <- hunspell_check(toupper(cr_dem@features), dict = dictionary("en_US"))
              #nns_out <- lapply(rownames(cr_dem_grouped), function(group) find_nns(cr_dem_grouped[group,], pre_trained = pre_trained, N = 10, candidates = cr_dem_grouped@features[spellcheck], norm = nrm, stem = TRUE)) %>% setNames(rownames(cr_dem_grouped))
              nns_out <- find_nns(cr_dem_avg, pre_trained = pre_trained, N = 10, candidates = cr_dem@features[spellcheck], norm = nrm, stem = TRUE)
              
            } else {
              #nns_out <- lapply(rownames(cr_dem_grouped), function(group) find_nns(cr_dem_grouped[group,], pre_trained = pre_trained, N = 10, candidates = cr_dem_grouped@features, norm = nrm)) %>% setNames(rownames(cr_dem_grouped))
              nns_out <- find_nns(cr_dem_avg, pre_trained = pre_trained, N = 10, candidates = cr_dem@features, norm = nrm)
            }
            
            # reconstruction
            rec_out <- sim2(cr_dem_avg, matrix(pre_trained[target,], nrow = 1), norm = nrm)[,1]
            
            # substantive
            pol_out <- sapply(107:114, function(group) sim2(matrix(cr_dem_grouped[paste('D',group, sep = '-'),], nrow = 1), matrix(cr_dem_grouped[paste('R',group, sep = '-'),], nrow = 1), norm = nrm)[,1])
            
            # save
            out1[[length(out1) + 1]] <- list(nns = nns_out, rec = data.frame(target = target, w1 = w1, w2 = w2, rm_stopwords = rstw, stem = stm, norm = nrm, rec = rec_out), subs = data.frame(target = target, session = 107:114, w1 = w1, w2 = w2, rm_stopwords = rstw, stem = stm, norm = nrm, num_instances = num_instances, pol = pol_out))
            rm(nns_out, pol_out, rec_out)
          }
        }
        rm(cr_toks, cr_dfm, cr_dem, cr_dem_grouped, cr_dem_avg)
      }
    }
  }
  cat("done with", target, "\n")
}

# save
#saveRDS(out1, "data/cr_experiments/cr_experiments_out1.rds")

# --------------------------------
#
#
# evaluate results
#
#
# --------------------------------

# --------------------------------
# reconstruction metric
# -------------------------------

# load results
out1 <- readRDS("data/cr_experiments/cr_experiments_out1.rds")

# rec dataframe
rec_df <- lapply(out1, "[[", "rec") %>% 
  bind_rows() %>% 
  mutate(model = 1L:nrow(.), 
         w2 = if_else(w1 == "glove" & w2 == 6, "glove", as.character(w2)),
         w2 = factor(w2, levels = c('2','6','12','glove')),
         w1 = factor(w1, levels = c('2','6','12','glove')),
         rm_stopwords = if_else(rm_stopwords, "rm_stopwords = TRUE", "rm_stopwords = FALSE"),
         norm = if_else(norm == 'none', "inner \n product", "cosine \n similarity"))

# NOTE: we only evaluate models with same window size
# i.e. w1 == w2

# cosine similarity
fgF7a <- ggplot(subset(rec_df, w1 == w2 & !stem & norm == "cosine \n similarity"), aes(x = target, y = rec, fill = w2)) +
  geom_point(size = 4, alpha = 0.70, color = 'black') +
  scale_shape_manual(values = c(21,22,23,24)) +
  scale_fill_grey(start = 0, end = 0.6) +
  stat_summary(aes(x = 0.1, y = rec, yintercept = stat(y)), fun = median, geom = "hline", linetype = "dotted") +
  ylab('Cosine similarity between \n ALC embedding and \n full GloVe model embedding') +
  facet_grid(w2~rm_stopwords) +
  theme(axis.text.x = element_text(size=18, angle = 90, hjust = 1, vjust = 0.5),
        axis.text.y = element_text(size=18),
        axis.title.y = element_text(size=20, vjust = 5),
        axis.title.x = element_blank(),
        legend.text=element_text(size=18),
        legend.title=element_blank(),
        legend.key=element_blank(),
        legend.position = "none",
        legend.spacing.x = unit(0.25, 'cm'),
        strip.text = element_text(size = 18),
        plot.margin = unit(c(1,1,1,1), "cm"))

ggsave(filename = "fgF7a.pdf", plot = fgF7a, height = 10, width = 12, path = './figures/', dpi = 1000)


# inner product
fgF8a <- ggplot(subset(rec_df, w1 == w2 & !stem & norm == "inner \n product"), aes(x = target, y = rec, fill = w2)) +
  geom_point(size = 4, alpha = 0.70, color = 'black') +
  scale_shape_manual(values = c(21,22,23,24)) +
  scale_fill_grey(start = 0, end = 0.6) +
  stat_summary(aes(x = 0.1, y = rec, yintercept = stat(y)), fun = median, geom = "hline", linetype = "dotted") +
  ylab('Inner product between \n ALC embedding and \n full GloVe model embedding') +
  facet_grid(w2~rm_stopwords, scales = "free_y") +
  theme(axis.text.x = element_text(size=18, angle = 90, hjust = 1, vjust = 0.5),
        axis.text.y = element_text(size=18),
        axis.title.y = element_text(size=20, vjust = 5),
        axis.title.x = element_blank(),
        legend.text=element_text(size=18),
        legend.title=element_blank(),
        legend.key=element_blank(),
        legend.position = "none",
        legend.spacing.x = unit(0.25, 'cm'),
        strip.text = element_text(size = 18),
        plot.margin = unit(c(1,1,1,1), "cm"))

ggsave(filename = "fgF8a.pdf", plot = fgF8a, height = 10, width = 12, path = './figures/', dpi = 1000)

# --------------------------------
# substantive metric
# -------------------------------

# subs dataframe
subs_df <- lapply(out1, "[[", "subs") %>% 
  bind_rows() %>% 
  mutate(w2 = if_else(w1 == "glove" & w2 == 6, "glove", as.character(w2)),
         w2 = factor(w2, levels = c('2','6','12','glove')),
         w1 = factor(w1, levels = c('2','6','12','glove')),
         rm_stopwords = if_else(rm_stopwords, "rm_stopwords = TRUE", "rm_stopwords = FALSE"),
         norm = if_else(norm == 'none', "inner \n product", "cosine \n similarity"))

# party (cosine based) similarity 
fgF7d <- ggplot(subset(subs_df, w1 == w2 & !stem & norm == "cosine \n similarity" & rm_stopwords == "rm_stopwords = FALSE"), aes(x = session, y = pol)) +
  geom_point(size = 2, alpha = 0.7, color = 'black') +
  geom_line(size = 1) +
  ylab('Similarity between ALC party embeddings \n (Republicans and Democrats)') +
  xlab('Session') +
  facet_grid(w1 ~ target, scales = "free") +
  theme(axis.text.x = element_text(size=18, angle = 90, hjust = 1, vjust = 0.5),
        axis.text.y = element_text(size=18),
        axis.title.y = element_text(size=20, vjust = 5),
        axis.title.x = element_text(size=20, vjust = -2),
        legend.text=element_text(size=18),
        legend.title=element_blank(),
        legend.key=element_blank(),
        legend.position = "top",
        legend.spacing.x = unit(0.25, 'cm'),
        strip.text = element_text(size = 16),
        plot.margin = unit(c(1,1,1,1), "cm"))

ggsave(filename = "fgF7d.pdf", plot = fgF7d, height = 10, width = 14, path = './figures/', dpi = 1000)


# party (inner product based) similarity 
fgF8c <- ggplot(subset(subs_df, w1 == w2 & !stem & norm == "inner \n product" & rm_stopwords == "rm_stopwords = FALSE"), aes(x = session, y = pol)) +
  geom_point(size = 2, alpha = 0.7, color = 'black') +
  geom_line(size = 1) +
  ylab('Similarity between ALC party embeddings \n (Republicans and Democrats)') +
  xlab('Session') +
  facet_grid(w1 ~ target, scales = "free") +
  theme(axis.text.x = element_text(size=18, angle = 90, hjust = 1, vjust = 0.5),
        axis.text.y = element_text(size=18),
        axis.title.y = element_text(size=20, vjust = 5),
        axis.title.x = element_text(size=20, vjust = -2),
        legend.text=element_text(size=18),
        legend.title=element_blank(),
        legend.key=element_blank(),
        legend.position = "top",
        legend.spacing.x = unit(0.25, 'cm'),
        strip.text = element_text(size = 16),
        plot.margin = unit(c(1,1,1,1), "cm"))

ggsave(filename = "fgF8c.pdf", plot = fgF8c, height = 10, width = 14, path = './figures/', dpi = 1000)

# correlations
cor_out <- list()
for(i in targets){
  for(rmst in c("rm_stopwords = FALSE", "rm_stopwords = TRUE")){
    for(nrm in c("cosine \n similarity", "inner \n product")){
      cor_i <- subs_df %>% filter(w1 == w2 & !stem & norm == nrm & rm_stopwords == rmst & target == i) %>% 
        select(session, w1, w2, pol) %>% 
        group_by(w1,w2) %>% 
        tidyr::pivot_wider(names_from = c(w1,w2), values_from = pol) %>% 
        select(-session) %>% as.matrix() %>% cor() %>% reshape2::melt() %>%
        rowwise() %>% mutate(group = paste(sort(c(Var1,Var2)), collapse = "", sep = "-")) %>% distinct(group, .keep_all = TRUE) %>% ungroup()
      cor_i <- cor_i %>% mutate(target = i) %>% select(-group)
      cor_i <- cor_i %>% mutate(Var1 = recode_factor(Var1, `2_2` = "2", `6_6` = "6", `12_12` = "12", `glove_glove` = "glove"),
                                Var2 = recode_factor(Var2, `2_2` = "2", `6_6` = "6", `12_12` = "12", `glove_glove` = "glove"),
                                norm = nrm, rm_stopwords = rmst)
      cor_out[[length(cor_out) + 1]] <- cor_i
    }
  }
}

# correlations are almost perfect
cor_df <- bind_rows(cor_out)

# cor plot
fgF7e <- ggplot(subset(cor_df, norm == "cosine \n similarity" & rm_stopwords == "rm_stopwords = FALSE"), aes(x = Var1, y = Var2, fill=value)) + 
  geom_tile() +
  facet_wrap(~target) +
  geom_text(aes(label = round(value, 2)), size = 6, color = "white") +
  theme(panel.background = element_blank(),
        axis.text.x = element_text(size=18, angle = 90, vjust = 0.5, hjust = 1),
        axis.text.y = element_text(size=18),
        axis.title.y = element_blank(),
        axis.title.x = element_blank(),
        legend.position = "none",
        strip.text = element_text(size = 18),
        plot.margin = unit(c(1,1,1,1), "cm"))

ggsave(filename = "fgF7e.pdf", plot = fgF7e, height = 10, width = 12, path = './figures/', dpi = 1000)

# cor plot
fgF8d <- ggplot(subset(cor_df, norm == "inner \n product" & rm_stopwords == "rm_stopwords = FALSE"), aes(x = Var1, y = Var2, fill=value)) + 
  geom_tile() +
  facet_wrap(~target) +
  geom_text(aes(label = round(value, 2)), size = 6, color = "white") +
  theme(panel.background = element_blank(),
        axis.text.x = element_text(size=18, angle = 90, vjust = 0.5, hjust = 1),
        axis.text.y = element_text(size=18),
        axis.title.y = element_blank(),
        axis.title.x = element_blank(),
        legend.position = "none",
        strip.text = element_text(size = 18),
        plot.margin = unit(c(1,1,1,1), "cm"))

ggsave(filename = "fgF8d.pdf", plot = fgF8d, height = 10, width = 12, path = './figures/', dpi = 1000)


# cor plot
fgF7f <- ggplot(subset(cor_df, norm == "cosine \n similarity" & rm_stopwords == "rm_stopwords = TRUE"), aes(x = Var1, y = Var2, fill=value)) + 
  geom_tile() +
  facet_wrap(~target) +
  geom_text(aes(label = round(value, 2)), size = 6, color = "white") +
  theme(panel.background = element_blank(),
        axis.text.x = element_text(size=18, angle = 90, vjust = 0.5, hjust = 1),
        axis.text.y = element_text(size=18),
        axis.title.y = element_blank(),
        axis.title.x = element_blank(),
        legend.position = "none",
        strip.text = element_text(size = 18),
        plot.margin = unit(c(1,1,1,1), "cm"))

ggsave(filename = "fgF7f.pdf", plot = fgF7f, height = 10, width = 12, path = './figures/', dpi = 1000)

# --------------------------------
# jaccard index
# --------------------------------

# jaccard index fcn
jaccard_index <- function(x,y){
  return(length(intersect(x,y))/length(unique(c(x,y))))
}

nns_list <- lapply(out1, "[[", "nns")
jacc_out <- list()
for(i in targets){
  for(rstw in c("rm_stopwords = TRUE", "rm_stopwords = FALSE")){
    for(nrm in c('inner \n product', 'cosine \n similarity')){
      for(stm in c(TRUE, FALSE)){
      rec_df_subset <- rec_df %>% filter(w1 == w2 & target == i & rm_stopwords == rstw & norm == nrm & stem == stm)
      model_grid <- expand.grid(rec_df_subset$model, rec_df_subset$model, KEEP.OUT.ATTRS = FALSE)
      model_grid$value <- sapply(1:nrow(model_grid), function(j) jaccard_index(x = nns_list[[model_grid$Var1[j]]], y = nns_list[[model_grid$Var2[[j]]]]))
      model_grid <- model_grid %>% mutate(target = i, rm_stopwords = rstw, norm = nrm, stem = stm)
      model_grid <- left_join(model_grid, y = rec_df_subset[,c("w1","model")], by = c("Var1" = "model"))
      model_grid <- left_join(model_grid, y = rec_df_subset[,c("w1","model")], by = c("Var2" = "model"))
      model_grid <- model_grid %>% select(x = w1.x, y = w1.y, target, rm_stopwords, norm, stem, value)
      jacc_out[[length(jacc_out) + 1]] <- model_grid
      }
    }
  }
}

jaccard_df <- bind_rows(jacc_out) %>%  mutate(stem = if_else(stem, "stem = TRUE", "stem = FALSE"),
                                    x = factor(x, levels = c('2','6','12','glove')),
                                    y = factor(y, levels = c('2','6','12','glove')))

# nns
fgF7b <- ggplot(subset(jaccard_df, stem == "stem = FALSE" & norm == "cosine \n similarity" & rm_stopwords == "rm_stopwords = FALSE" & as.integer(x) >= as.integer(y)), aes(x = x, y = y, fill=value)) + 
  geom_tile() +
  facet_wrap(~target) +
  geom_text(aes(label = round(value, 2)), size = 6, color = "white") +
  theme(panel.background = element_blank(),
        axis.text.x = element_text(size=18, angle = 90, vjust = 0.5, hjust = 1),
        axis.text.y = element_text(size=18),
        axis.title.y = element_blank(),
        axis.title.x = element_blank(),
        legend.position = "none",
        strip.text = element_text(size = 18),
        plot.margin = unit(c(1,1,1,1), "cm"))

ggsave(filename = "fgF7b.pdf", plot = fgF7b, height = 10, width = 12, path = './figures/', dpi = 1000)

fgF7c <- ggplot(subset(jaccard_df, stem == "stem = FALSE" & norm == "cosine \n similarity" & rm_stopwords == "rm_stopwords = TRUE" & as.integer(x) >= as.integer(y)), aes(x = x, y = y, fill=value)) + 
  geom_tile() +
  facet_wrap(~target) +
  geom_text(aes(label = round(value, 2)), size = 6, color = "white") +
  theme(panel.background = element_blank(),
        axis.text.x = element_text(size=18, angle = 90, vjust = 0.5, hjust = 1),
        axis.text.y = element_text(size=18),
        axis.title.y = element_blank(),
        axis.title.x = element_blank(),
        legend.position = "none",
        strip.text = element_text(size = 18),
        plot.margin = unit(c(1,1,1,1), "cm"))

ggsave(filename = "fgF7c.pdf", plot = fgF7c, height = 10, width = 12, path = './figures/', dpi = 1000)

# nns inner
fgF8b <- ggplot(subset(jaccard_df, stem == "stem = FALSE" & norm == "inner \n product" & rm_stopwords == "rm_stopwords = FALSE" & as.integer(x) >= as.integer(y)), aes(x = x, y = y, fill=value)) + 
  geom_tile() +
  facet_wrap(~target) +
  geom_text(aes(label = round(value, 2)), size = 6, color = "white") +
  theme(panel.background = element_blank(),
        axis.text.x = element_text(size=18, angle = 90, vjust = 0.5, hjust = 1),
        axis.text.y = element_text(size=18),
        axis.title.y = element_blank(),
        axis.title.x = element_blank(),
        legend.position = "none",
        strip.text = element_text(size = 18),
        plot.margin = unit(c(1,1,1,1), "cm"))

ggsave(filename = "fgF8b.pdf", plot = fgF8b, height = 10, width = 12, path = './figures/', dpi = 1000)

# nns + stem
fgF9 <- ggplot(subset(jaccard_df, stem == "stem = TRUE" & norm == "cosine \n similarity" & rm_stopwords == "rm_stopwords = FALSE" & as.integer(x) >= as.integer(y)), aes(x = x, y = y, fill=value)) + 
  geom_tile() +
  facet_wrap(~target) +
  geom_text(aes(label = round(value, 2)), size = 6, color = "white") +
  theme(panel.background = element_blank(),
        axis.text.x = element_text(size=18, angle = 90, vjust = 0.5, hjust = 1),
        axis.text.y = element_text(size=18),
        axis.title.y = element_blank(),
        axis.title.x = element_blank(),
        legend.position = "none",
        strip.text = element_text(size = 18),
        plot.margin = unit(c(1,1,1,1), "cm"))

ggsave(filename = "fgF9.pdf", plot = fgF9, height = 10, width = 12, path = './figures/', dpi = 1000)

# --------------------------------
# nearest neighbors
# --------------------------------
nns_list <- lapply(out1, "[[", "nns")
args_list <- lapply(out1, "[[", "rec")
nns_df <- cbind(do.call(rbind, nns_list), do.call(rbind, args_list)) %>% rename(embeddings_window = w1, contexts_window = w2) %>% mutate(norm = if_else(norm == 'none', 'inner', 'cosine'))
write.csv(nns_df, file = "data/cr_experiments/nns.csv", row.names = FALSE)

# --------------------------------
#
# uncertainty around A matrix
#
# --------------------------------

# load model and pre-trained embeddings
transform_matrix_list <- readRDS("data/cr_experiments/cr.am.bs.6w.300d.rds")
pre_trained <- readRDS("data/cr_experiments/cr.wvs.6w.300d.rds")
out2 <- list()
for(target in targets){
  
  # select contexts (tokens) around target
  cr_toks <- tokens_context(x = cr_toks_base, pattern = target, window = 6, valuetype = "fixed", case_insensitive = TRUE, hard_cut = FALSE, verbose = TRUE)
  
  for(i in 1:length(transform_matrix_list)){
    
    # select transform matrix
    transform_matrix <- transform_matrix_list[[i]]
    
    # --------------------------------
    # compute ALC embeddings
    # --------------------------------
    
    # build a document-feature-matrix
    cr_dfm <- dfm(cr_toks)
    
    # build a document-embedding-matrix
    cr_dem <- dem(x = cr_dfm, pre_trained = pre_trained, transform = TRUE, transform_matrix = transform_matrix, verbose = FALSE)
    
    # average dem by group
    cr_dem_grouped <- dem_group(x = cr_dem, groups = cr_dem@docvars$group)
    
    # average dem
    cr_dem_avg <- matrix(colMeans(cr_dem), nrow = 1)
    
    # --------------------------------
    # performance metrics
    # --------------------------------
    
    # nns
    nns_out <- find_nns(cr_dem_avg, pre_trained = pre_trained, N = 10, candidates = cr_dem@features, norm = "l2")
    
    # reconstruction
    rec_out <- sim2(cr_dem_avg, matrix(pre_trained[target,], nrow = 1), norm = 'l2')[,1]
    
    # substantive
    pol_out <- sapply(107:114, function(group) sim2(matrix(cr_dem_grouped[paste('D',group, sep = '-'),], nrow = 1), matrix(cr_dem_grouped[paste('R',group, sep = '-'),], nrow = 1), norm = 'l2')[,1])
    
    # save
    out2[[length(out2) + 1]] <- list(nns = nns_out, rec = data.frame(target = target, a_matrix = as.character(i), rec = rec_out), subs = data.frame(target = target, a_matrix = as.character(i), session = 107:114, pol = pol_out))
  }
  cat("done with", target, "\n")
}

# save
#saveRDS(out2, "data/cr_experiments/cr_experiments_out2.rds")

# --------------------------------
# reconstruction metric
# --------------------------------

# read results
out2 <- readRDS("data/cr_experiments/cr_experiments_out2.rds")

# rec dataframe
rec_df <- lapply(out2, "[[", "rec") %>% 
  bind_rows()

# true embedding reconstruction
fgG10a <- ggplot(rec_df, aes(x = target, y = rec)) +
  geom_point(size = 4, alpha = 1/5, color = 'black') +
  ylab('Cosine similarity between \n ALC embedding and \n full (local) GloVe model embedding') +
  theme(axis.text.x = element_text(size=18, angle = 90, hjust = 1, vjust = 0.5),
        axis.text.y = element_text(size=18),
        axis.title.y = element_text(size=20, vjust = 5),
        axis.title.x = element_blank(),
        plot.margin = unit(c(1,1,1,1), "cm"))

ggsave(filename = "fgG10a.pdf", plot = fgG10a, height = 10, width = 12, path = './figures/', dpi = 1000)

# --------------------------------
# substantive metric
# --------------------------------

# subs dataframe
subs_df <- lapply(out2, "[[", "subs") %>% 
  bind_rows()

# party (cosine based) similarity 
fgG10b <- ggplot(subs_df, aes(x = session, y = pol, group = a_matrix)) +
  geom_point(size = 1, alpha = 1/10, color = 'black') +
  geom_line(size = 0.25, alpha = 1/10) +
  ylab('Similarity between ALC party embeddings \n (Republicans and Democrats)') +
  xlab("Session") +
  facet_wrap(~ target, scales = 'free') +
  theme(axis.text.x = element_text(size=18, angle = 90, hjust = 1, vjust = 0.5),
        axis.text.y = element_text(size=18),
        axis.title.y = element_text(size=20, vjust = 5),
        axis.title.x = element_text(size=20, vjust = -2),
        legend.position = "none",
        strip.text = element_text(size = 22),
        plot.margin = unit(c(1,1,1,1), "cm"))

ggsave(filename = "fgG10b.pdf", plot = fgG10b, height = 10, width = 12, path = './figures/', dpi = 1000)

# correlations
cor_out <- vector('list', length(targets)) %>% setNames(targets)
for(i in targets){
  cor_i <- subs_df %>% filter(target == i) %>% select(-target) %>% tidyr::pivot_wider(names_from = a_matrix, values_from = pol) %>% select(-session) %>% as.matrix() %>% cor() %>% .[upper.tri(.)]
  cor_out[[i]] <- tibble(target = i, value = cor_i)
}

# correlations are alomost perfect
cor_df <- bind_rows(cor_out)

# cor plot
fgG10c <- ggplot(cor_df, aes(x = target, y = value)) + 
  stat_summary(geom = "pointrange", fun.data =  'mean_se', size = 2) +
  ylab("Mean pairwise pearson correlation across models") +
  coord_flip() +
  theme(axis.text.x = element_text(size=18),
        axis.text.y = element_text(size=18),
        axis.title.x = element_text(size=20, vjust = -5),
        axis.title.y = element_blank(),
        plot.margin = unit(c(1,1,1,1), "cm"))

ggsave(filename = "fgG10c.pdf", plot = fgG10c, height = 10, width = 12, path = './figures/', dpi = 1000)

# --------------------------------
# jaccard index
# --------------------------------

nns_list <- lapply(out2, "[[", "nns")
jacc_out <- list()
for(i in targets){
  nns_list_subset <- nns_list[which(rec_df$target == i)]
  model_grid <- expand.grid(1:length(nns_list_subset), 1:length(nns_list_subset), KEEP.OUT.ATTRS = FALSE)
  model_grid$value <- sapply(1:nrow(model_grid), function(j) jaccard_index(x = nns_list_subset[[model_grid$Var1[j]]], y = nns_list_subset[[model_grid$Var2[[j]]]]))
  model_grid <- model_grid %>% mutate(target = i) %>% select(target, value)
  jacc_out[[length(jacc_out) + 1]] <- model_grid
}

jaccard_df <- bind_rows(jacc_out)

# nns
fgG10d <- ggplot(jaccard_df, aes(x = target, y = value)) + 
  stat_summary(geom = "pointrange", fun.data =  'mean_se', size = 2) +
  ylab("Mean pairwise jaccard index across the 10 bootstraps of A") +
  coord_flip() +
  theme(axis.text.x = element_text(size=18),
        axis.text.y = element_text(size=18),
        axis.title.x = element_text(size=20, vjust = -5),
        axis.title.y = element_blank(),
        plot.margin = unit(c(1,1,1,1), "cm"))

ggsave(filename = "fgG10d.pdf", plot = fgG10d, height = 10, width = 12, path = './figures/', dpi = 1000)

