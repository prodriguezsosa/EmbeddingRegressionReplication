library(text2vec)
library(dplyr)
library(stringr)

# ================================
# choice parameters
# ================================
WINDOW_SIZE <- 6
DIM <- 300
ITERS <- 100
MIN_COUNT <- 10

# ================================
# define paths
# ================================
# set working directory to the location of the master "EmbeddingRegressionReplication" folder
setwd("")

# ================================
# load data
# ================================
corpus <- readRDS("data/corpus_full_cr.rds")

# subset
text <- corpus %>% filter(period_start >= 2004) %>% .$speech

# ================================
# basic preprocessing of text
# ================================
text <- text %>% 
  gsub("[^[:alpha:]]", " ", .) %>% # remove all non-alpha characters
  str_replace_all("\\b\\w{1,2}\\b", "") %>% # remove 1-2 letter words
  str_replace_all("^ +| +$|( ) +", "\\1") %>% # remove excess white space
  tolower() # lowercase

# shuffle text
set.seed(42L)
text <- sample(text)

# ================================
# create vocab
# ================================
tokens <- space_tokenizer(text)
it <- itoken(tokens, progressbar = FALSE)
vocab <- create_vocabulary(it)
vocab <- prune_vocabulary(vocab, term_count_min = MIN_COUNT)  # keep only words that meet count threshold

# ================================
# create term co-occurrence matrix
# ================================
vectorizer <- vocab_vectorizer(vocab)
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

# ================================
# save
# ================================
saveRDS(word_vectors, file = "data/word_vectors_cr.rds")
