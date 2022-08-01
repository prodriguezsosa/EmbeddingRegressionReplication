library(text2vec)
library(dplyr)
library(stringr)
library(pbapply)

# set working directory to the location of the master "EmbeddingRegressionReplication" folder
setwd("")

# ================================
# choice parameters
# ================================
WINDOW_SIZE <- 6
DIM <- 300
ITERS <- 100
MIN_COUNT <- 5

# ================================
# load data
# ================================
corpus <- readRDS("data/corpus_daily.rds")

# subset
corpus <- corpus %>% filter(period_start >= 2009) %>% select(speech, party) %>% filter(party %in% c('D', 'R')) %>% rename(text = speech)

# ================================
# basic preprocessing of text
# ================================
corpus$text <- corpus$text %>% 
  gsub("[^[:alpha:]]", " ", .) %>% # remove all non-alpha characters
  str_replace_all("\\b\\w{1,2}\\b", "") %>% # remove 1-2 letter words
  str_replace_all("^ +| +$|( ) +", "\\1") %>% # remove excess white space
  tolower() # lowercase

# ================================
# prepare datasets
# ================================

# define targets
targets <- c('immigration', 'economy', 'climatechange', 'healthcare', 'middleeast', 'floor')
#target_present <- grep(target, corpus$text, fixed = TRUE)
corpus$text <- str_replace_all(corpus$text, 'middle east', 'middleeast')
corpus$text <- str_replace_all(corpus$text, 'health care', 'healthcare')
corpus$text <- str_replace_all(corpus$text, 'immigrants', 'immigration')
corpus$text <- str_replace_all(corpus$text, 'immigrant', 'immigration')
corpus$text <- str_replace_all(corpus$text, 'climate change', 'climatechange')

# find indices for all targets
set.seed(1984L)
target_index_list <- pblapply(targets, function(target){ 
  target_present <- grep(target, corpus$text, fixed = TRUE)
  return(list(D = target_present[corpus$party[target_present] == 'D'], 
              R = target_present[corpus$party[target_present] == 'R']))
}) %>% setNames(targets)

# exclude speeches with multiple target words
overlap_indices <- lapply(target_index_list, function(x) unlist(x) %>% unname() %>% unique()) %>% unlist() %>% unname()
overlap_indices <- overlap_indices[duplicated(overlap_indices)]
target_index_list <- lapply(targets, function(target) lapply(target_index_list[[target]], function(x) setdiff(x, overlap_indices))) %>% setNames(targets)

# sample rows to be excluded
sample_sizes <- c(5, 10, 15, 25, 50, 100, 200)
exclusion_list <- vector('list', length(sample_sizes)) %>% setNames(sample_sizes)
for(sample_size in sample_sizes){
  exclusion_list[[as.character(sample_size)]] <- lapply(targets, function(target) lapply(target_index_list[[target]], function(x) sample(x, (length(x) - sample_size))) %>% unlist() %>% unname()) %>% setNames(targets)
}

# collapse
exclusion_list <- lapply(exclusion_list, function(x) unlist(x) %>% unname() %>% union(., overlap_indices) %>% unique())

# build corpora for each sample size
corpus_list <- vector('list', length(sample_sizes)) %>% setNames(sample_sizes)
for(sample_size in sample_sizes){
  corpus_list[[as.character(sample_size)]] <- corpus %>% slice(-exclusion_list[[as.character(sample_size)]])
}

# add full corpus
sample_sizes <- c(sample_sizes,'full')
corpus_list[['full']] <- corpus[-overlap_indices,]

# save corpora
saveRDS(corpus_list, paste0(out_path, 'corpus_list.rds'))

# tag target term
corpus_list <- pblapply(corpus_list, function(df){
  for(target in targets){
    df$text[df$party == 'R'] <- gsub(target, paste0(target, 'r'), df$text[df$party == 'R'])
    df$text[df$party == 'D'] <- gsub(target, paste0(target, 'd'), df$text[df$party == 'D'])
  }
  return(df)
})

# spring cleaning
rm(corpus, exclusion_list, target_index_list)

set.seed(1984L)
for(i in 1:length(corpus_list)){
  
  text <- corpus_list[[i]]$text
  
  # begin timer
  start_time_full <- Sys.time()
  
  # ================================
  # create vocab
  # ================================
  tokens <- space_tokenizer(text)
  it <- itoken(tokens, progressbar = FALSE)
  vocab <- create_vocabulary(it)
  vocab_pruned <- prune_vocabulary(vocab, term_count_min = MIN_COUNT)  # keep only words that meet count threshold
  
  # ================================
  # create term co-occurrence matrix
  # ================================
  vectorizer <- vocab_vectorizer(vocab_pruned)
  tcm <- create_tcm(it, vectorizer, skip_grams_window = WINDOW_SIZE, skip_grams_window_context = "symmetric", weights = rep(1, WINDOW_SIZE))
  
  # ================================
  # set model parameters
  # ================================
  
  glove <- GlobalVectors$new(rank = DIM,
                             x_max = 100)
  
  # ================================
  # fit model
  # ================================
  
  word_vectors_main <- glove$fit_transform(tcm, 
                                           n_iter = ITERS,
                                           convergence_tol = 1e-3, 
                                           #n_check_convergence = 1L,
                                           n_threads = RcppParallel::defaultNumThreads())
  
  # ================================
  # get output
  # ================================
  word_vectors_context <- glove$components
  word_vectors <- word_vectors_main + t(word_vectors_context) # word vectors
  comp_time <- Sys.time() - start_time_full
  
  # ================================
  # save
  # ================================
  saveRDS(word_vectors, file = paste0("data/benchmarks/word_vectors_", sample_sizes[i], ".rds"))
  #saveRDS(comp_time, file = paste0("data/cr_experiments/comp_time_", sample_sizes[i], ".rds"))
}
