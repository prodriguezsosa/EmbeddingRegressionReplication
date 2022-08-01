library(text2vec)
library(dplyr)
library(stringr)
library(hunspell)

# ================================
# choice parameters
# ================================
WINDOW_SIZE <- 6
DIM <- 300
ITERS <- 25
TERM_MAX <- 5000

# ================================
# define paths
# ================================
# set working directory to the location of the master "EmbeddingRegressionReplication" folder
setwd("")

# ================================
# load data
# ================================
cr <- readRDS("data/corpus_full_cr.rds") %>% 
  filter(period_start >= 1935 & period_end <= 2009) %>%
  select(speech, session_id) %>% 
  mutate(speech = iconv(speech, from = "latin1", to = "UTF-8")) %>%
  distinct(speech, .keep_all = TRUE) %>% tidyr::drop_na() %>%
  group_by(session_id) %>%
  sample_frac(size = 0.25) %>%
  ungroup() %>% pull(speech)

ps <- readRDS("data/corpus_full_ps.rds") %>%
  filter(year >= 1935 & year <= 2009) %>%
  select(text, year) %>% 
  mutate(text = iconv(text, from = "latin1", to = "UTF-8")) %>%
  distinct(text, .keep_all = TRUE) %>% tidyr::drop_na() %>%
  group_by(year) %>%
  sample_frac(size = 0.25) %>%
  ungroup() %>% pull(text)

# shuffle text
set.seed(42L)
text <- sample(c(cr, ps))
rm(cr,ps)

# pre-porocessing
text <- text %>% 
  gsub("[^[:alpha:]]", " ", .) %>% # remove all non-alpha characters
  str_replace_all("\\b\\w{1,2}\\b", "") %>% # remove 1-2 letter words
  str_replace_all("^ +| +$|( ) +", "\\1") %>% # remove excess white space
  tolower() %>% # lowercase
  distinct() # remove duplicates

# ================================
# create vocab
# ================================
tokens <- space_tokenizer(text)
it <- itoken_parallel(tokens, n_chunks = 6)
vocab <- create_vocabulary(it)
vocab_pruned <- prune_vocabulary(vocab, term_count_min = 10)  # keep only words that meet count threshold

# rm misspelled words
spellcheck_us <-  hunspell_check(toupper(vocab_pruned$term), dict = dictionary("en_US")) # remove incorrectly spelled words
spellcheck_gb <-  hunspell_check(toupper(vocab_pruned$term), dict = dictionary("en_GB")) # remove incorrectly spelled words
spellcheck <- spellcheck_us | spellcheck_gb
vocab_pruned <- vocab_pruned[spellcheck,]

# ================================
# create term co-occurrence matrix
# ================================
vectorizer <- vocab_vectorizer(vocab_pruned)
tcm <- create_tcm(it, vectorizer, skip_grams_window = WINDOW_SIZE, skip_grams_window_context = "symmetric", weights = rep(1, WINDOW_SIZE))

# ================================
# set model parameters
# ================================
glove <- GlobalVectors$new(rank = DIM,
                           x_max = 100,
                           learning_rate = 0.05)

# ================================
# fit model
# ================================
word_vectors_main <- glove$fit_transform(tcm,
                                         n_iter = ITERS,
                                         convergence_tol = 1e-3, 
                                         #n_check_convergence = 1L,
                                         n_threads = 6)

# ================================
# get output
# ================================
word_vectors_context <- glove$components
word_vectors <- word_vectors_main + t(word_vectors_context) # word vectors

# ================================
# save
# ================================
saveRDS(word_vectors, file = "data/word_vectors_6_300_5000.rds")
