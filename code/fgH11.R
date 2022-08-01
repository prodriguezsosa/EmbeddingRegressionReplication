# --------------------------------
# setup
# --------------------------------
library(dplyr)
library(quanteda)
library(conText)
library(text2vec)
library(ggplot2)

# set working directory to the location of the master "EmbeddingRegressionReplication" folder
setwd("")

# --------------------------------
# load corpus
# --------------------------------

# session 43 - 50
cr043 <- readRDS('data/corpus_full_cr.rds') %>% 
  select(speech, party, session_id) %>% 
  filter(session_id %in% 43:50 & party %in% c('D', 'R')) %>%
  mutate(speech = tolower(iconv(speech, from = "latin1", to = "UTF-8"))) %>%
  distinct(speech, .keep_all = TRUE) %>% tidyr::drop_na() %>% 
  select(speech, party, session_id) %>% tidyr::unite(c(party, session_id), col = "group", sep = '-', remove = FALSE)

# session 107 - 114
cr107 <- readRDS('data/corpus_daily.rds') %>% 
  select(speech, party, session_id) %>% 
  filter(session_id %in% 107:114 & party %in% c('D', 'R')) %>%
  mutate(speech = tolower(iconv(speech, from = "latin1", to = "UTF-8"))) %>%
  distinct(speech, .keep_all = TRUE) %>% tidyr::drop_na() %>% 
  select(speech, party, session_id) %>% tidyr::unite(c(party, session_id), col = "group", sep = '-', remove = FALSE)

# build corpora
cr043_corpus <- corpus(cr043$speech, docvars = data.frame(party = cr043$party, group = cr043$group))
cr107_corpus <- corpus(cr107$speech, docvars = data.frame(party = cr107$party, group = cr107$group))
cr_corpus <- list('cr43' = cr043_corpus, 'cr' = cr107_corpus)

# --------------------------------
# pre-processing
# --------------------------------
cr_toks_base <- lapply(cr_corpus, function(i) {
  cr_toks <- tokens(i, remove_punct = T, remove_symbols = T, remove_numbers = T, remove_url = T, remove_separators = T)
  feats <- dfm(cr_toks, verbose = TRUE) %>% dfm_trim(min_termfreq = 10) %>% featnames() # min_termfreq set to same value as used for training
  cr_toks <- tokens_select(cr_toks, pattern = feats, selection = "keep", padding = TRUE, min_nchar = 3) # subset tokens object to selected features
  return(cr_toks)
}) %>% setNames(names(cr_corpus))

# --------------------------------
#
#
# run experiments
#
#
# --------------------------------

# preliminaries
path_to_inputs <- "~/Dropbox/GitHub/large_data/ALaCarteR/alacarteR/cr/"
out1 <- list()
set.seed(1984L)
# --------------------------------
# define pre-trained embeddings
# --------------------------------
targets <- c("democracy", "freedom", "equality", "justice", "immigration", "abortion", "welfare", "taxes", "republican", "democrat")
for(cr in c('cr', 'cr43')){
  for(model in c('glove-khodak', 'glove-local', 'local-local')){
    if(model == 'glove-khodak'){
      pre_trained <- readRDS("data/stanford-glove/glove.rds")
      transform_matrix <- readRDS("data/stanford-glove/khodakA.rds")
    } else if (model == 'glove-local') {
      pre_trained <- readRDS("data/stanford-glove/glove.rds")
      transform_matrix <- readRDS('data/cr_experiments/', cr, ".am.glove.300d.rds")
    }else{
      pre_trained <- readRDS('data/cr_experiments/', cr, ".wvs.6w.300d.rds")
      transform_matrix <- readRDS('data/cr_experiments/', cr, ".am.6w.300d.rds")
    }
    for(target in targets){
      # --------------------------------
      # tokenize contexts
      # --------------------------------
      
      # reset toks
      cr_toks <- cr_toks_base[[cr]]
      
      # select contexts (tokens) around target
      cr_toks <- tokens_context(x = cr_toks, pattern = target, window = 6L, valuetype = "fixed", case_insensitive = TRUE, hard_cut = FALSE, verbose = TRUE)
      
      # --------------------------------
      # compute ALC embeddings
      # --------------------------------
      
      # build a document-feature-matrix
      cr_dfm <- dfm(cr_toks)
      num_instances <- nrow(cr_dfm)
      
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
      nns_out <- find_nns(cr_dem_avg, pre_trained = pre_trained, N = 10, candidates = cr_dem@features, norm = 'l2')
      
      # reconstruction
      rec_out <- sim2(cr_dem_avg, matrix(pre_trained[target,], nrow = 1), norm = 'l2')[,1]
      
      # substantive
      #if(cr == 'cr43'){sessions <- 43:50}else{sessions <- 107:114} 
      # this annoying setup is due to some session in the early years having no instances of "abortion" for Republicans
      if(cr == 'cr43'){sessions <- names(which(table(as.integer(substr(rownames(cr_dem_grouped), 3,4))) == 2))}else{sessions <- names(which(table(as.integer(substr(rownames(cr_dem_grouped), 3,5))) == 2))}
      pol_out <- sapply(sessions, function(session) sim2(matrix(cr_dem_grouped[paste('D',session, sep = '-'),], nrow = 1), matrix(cr_dem_grouped[paste('R',session, sep = '-'),], nrow = 1), norm = 'l2')[,1])
      
      # save
      out1[[length(out1) + 1]] <- list(nns = nns_out, rec = data.frame(target = target, cr = cr, model = model, num_instances = num_instances, rec = rec_out), subs = data.frame(target = target, session = sessions, cr = cr, model = model, num_instances = num_instances, pol = pol_out))
      rm(nns_out, pol_out, rec_out, cr_toks, cr_dfm, cr_dem, cr_dem_grouped, cr_dem_avg)
      # progress
    }
    cat("done with", model, "\n")
  }
  cat("done with corpus", cr, "\n")
}

# save
saveRDS(out1, "data/cr_experiments/cr_time_experiments_out1.rds")

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

out1 <- readRDS("data/cr_experiments/cr_time_experiments_out1.rds")

# rec dataframe
rec_df <- lapply(out1, "[[", "rec") %>% 
  bind_rows() %>% 
  mutate(id = 1L:nrow(.)) %>%
  filter(target!='abortion') %>%
  mutate(cr = if_else(cr == 'cr', "Sessions 107 - 114", "Sessions 43 - 50"),
         num_instances = scale(num_instances)[,1])

# NOTE: we only evaluate models with same window size
# i.e. w1 == w2

# add frequency of instances to shape size

# cosine similarity
fgH11a <- ggplot(rec_df, aes(x = target, y = rec)) +
  geom_point(aes(size = num_instances), alpha = 0.70, color = 'black') +
  stat_summary(aes(x = 0.1, y = rec, yintercept = stat(y)), fun = median, geom = "hline", linetype = "dotted") +
  ylab('Cosine similarity between \n ALC embedding and \n full (local) GloVe model embedding') +
  facet_grid(cr~model) +
  theme(axis.text.x = element_text(size=18, angle = 90, hjust = 1, vjust = 0.5),
        axis.text.y = element_text(size=18),
        axis.title.y = element_text(size=20, vjust = 5),
        axis.title.x = element_blank(),
        legend.position = "none",
        legend.spacing.x = unit(0.25, 'cm'),
        strip.text = element_text(size = 16),
        plot.margin = unit(c(1,1,1,1), "cm"))

ggsave(filename = "fgH11a.pdf", plot = fgH11a, height = 10, width = 12, path = './figures/', dpi = 1000)

# --------------------------------
# substantive metric
# -------------------------------

# subs dataframe
subs_df <- lapply(out1, "[[", "subs") %>% 
  bind_rows() %>%
  filter(target!='abortion') %>%
  mutate(cr = if_else(cr == 'cr', "Sessions 107 - 114", "Sessions 43 - 50"))

# party (cosine based) similarity 
fgH11b <- ggplot(subset(subs_df, cr == "Sessions 43 - 50"), aes(x = as.integer(session), y = pol)) +
  geom_point(size = 2, alpha = 0.7, color = 'black') +
  geom_line(size = 1) +
  scale_x_continuous(breaks = 43:50) +
  ylab('Similarity between ALC party embeddings \n (Republicans and Democrats)') +
  xlab('Session') +
  facet_grid(model ~ target, scales = "free_y") +
  theme(axis.text.x = element_text(size=18, angle = 90, hjust = 1, vjust = 0.5),
        axis.text.y = element_text(size=18),
        axis.title.y = element_text(size=20, vjust = 5),
        axis.title.x = element_text(size=20, vjust = -2),
        strip.text = element_text(size = 16),
        plot.margin = unit(c(1,1,1,1), "cm"))

ggsave(filename = "fgH11b.pdf", plot = fgH11b, height = 10, width = 14, path = './figures/', dpi = 1000)


fgH11c <- ggplot(subset(subs_df, cr == "Sessions 107 - 114"), aes(x = as.integer(session), y = pol)) +
  geom_point(size = 2, alpha = 0.7, color = 'black') +
  geom_line(size = 1) +
  scale_x_continuous(breaks = 43:50) +
  ylab('Similarity between ALC party embeddings \n (Republicans and Democrats)') +
  xlab('Session') +
  facet_grid(model ~ target, scales = "free_y") +
  theme(axis.text.x = element_text(size=18, angle = 90, hjust = 1, vjust = 0.5),
        axis.text.y = element_text(size=18),
        axis.title.y = element_text(size=20, vjust = 5),
        axis.title.x = element_text(size=20, vjust = -2),
        strip.text = element_text(size = 16),
        plot.margin = unit(c(1,1,1,1), "cm"))

ggsave(filename = "fgH11c.pdf", plot = fgH11c, height = 10, width = 14, path = './figures/', dpi = 1000)

# correlations
cor_out <- list()
for(i in unique(subs_df$target)){
  for(j in c("Sessions 107 - 114", "Sessions 43 - 50")){
  num_instances <- subs_df %>% filter(target == i & cr == j) %>% pull(num_instances) %>% unique()
  cor_ij <- subs_df %>% filter(target == i & cr == j) %>% 
    select(session, model, pol) %>% 
    tidyr::pivot_wider(names_from = model, values_from = pol) %>% 
    select(-session) %>% as.matrix() %>% cor() %>% reshape2::melt() %>%
  rowwise() %>% mutate(group = paste(sort(c(Var1,Var2)), collapse = "", sep = "-")) %>% distinct(group, .keep_all = TRUE) %>% ungroup()
  cor_ij <- cor_ij %>% mutate(target = i, cr = j, num_instances = num_instances) %>% select(-group)
  cor_out[[length(cor_out) + 1]] <- cor_ij
  }
}

cor_df <- bind_rows(cor_out)

# cor plot
fgH11d <- ggplot(cor_df, aes(x = Var1, y = Var2, fill=value)) + 
  geom_tile() +
  facet_grid(cr ~ target) +
  geom_text(aes(label = round(value, 2)), size = 4, color = "white") +
  theme(panel.background = element_blank(),
        axis.text.x = element_text(size=18, angle = 90, vjust = 0.5, hjust = 1),
        axis.text.y = element_text(size=18),
        axis.title.y = element_blank(),
        axis.title.x = element_blank(),
        legend.position = "none",
        strip.text = element_text(size = 16),
        plot.margin = unit(c(1,1,1,1), "cm"))

ggsave(filename = "fgH11d.pdf", plot = fgH11d, height = 10, width = 16, path = './figures/', dpi = 1000)

# --------------------------------
# jaccard index
# --------------------------------

# jaccard index fcn
jaccard_index <- function(x,y){
  return(length(intersect(x,y))/length(unique(c(x,y))))
}

nns_list <- lapply(out1, "[[", "nns")
jacc_out <- list()
for(i in unique(rec_df$target)){
  for(j in c("Sessions 107 - 114", "Sessions 43 - 50")){
    rec_df_subset <- rec_df %>% filter(rec_df$target == i & rec_df$cr == j)
    model_grid <- expand.grid(rec_df_subset$id, rec_df_subset$id, KEEP.OUT.ATTRS = FALSE)
    model_grid$value <- sapply(1:nrow(model_grid), function(k) jaccard_index(x = nns_list[[model_grid$Var1[k]]], y = nns_list[[model_grid$Var2[[k]]]]))
    model_grid <- model_grid %>% mutate(target = i, cr = j)
    model_grid <- left_join(model_grid, y = rec_df_subset[,c("id","model")], by = c("Var1" = "id"))
    model_grid <- left_join(model_grid, y = rec_df_subset[,c("id","model")], by = c("Var2" = "id"))
    model_grid <- model_grid %>% select(x = model.x, y = model.y, target, cr, value)
    jacc_out[[length(jacc_out) + 1]] <- model_grid
    
  }
}

jaccard_df <- bind_rows(jacc_out) %>% rowwise() %>% mutate(group = paste(sort(c(x,y)), collapse = "", sep = "-")) %>% group_by(target, cr) %>% distinct(group, .keep_all = TRUE) %>% ungroup()
  
# nns
fgH11e <- ggplot(jaccard_df, aes(x = x, y = y, fill=value)) + 
  geom_tile() +
  facet_grid(cr ~ target) +
  geom_text(aes(label = round(value, 2)), size = 4, color = "white") +
  theme(panel.background = element_blank(),
        axis.text.x = element_text(size=18, angle = 90, vjust = 0.5, hjust = 1),
        axis.text.y = element_text(size=18),
        axis.title.y = element_blank(),
        axis.title.x = element_blank(),
        legend.position = "none",
        strip.text = element_text(size = 16),
        plot.margin = unit(c(1,1,1,1), "cm"))

ggsave(filename = "fgH11e.pdf", plot = fgH11e, height = 10, width = 16, path = './figures/', dpi = 1000)


