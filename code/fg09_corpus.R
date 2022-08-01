# --------------------------------
# setup
# --------------------------------

# libraries
library(dplyr)
library(conText)
library(quanteda)

# set working directory to the location of the master "EmbeddingRegressionReplication" folder
setwd("")

# --------------------------------
# load data
# --------------------------------

# Osnabrügge et al. data
uk_data <- read.csv("data/uk_data.csv", encoding = "UTF-8")

# glove pre-trained embeddings
pre_trained <- readRDS("data/stanford-glove/glove.rds")

# word ratings
warriner_raw <- read.csv("data/warriner.csv")

# --------------------------------
# build dictionary following Osnabrügge et al.
# --------------------------------

# use Osnabrügge's framework to select and classify dictionary words
warriner <- warriner_raw %>% 
  select(word = Word, mean.val = V.Mean.Sum, sd.val = V.SD.Sum) %>% 
  mutate(polarity = case_when(mean.val > 7 & sd.val < 2 ~ "positive",
                              mean.val < 3 & sd.val < 2 ~ "negative",
                              mean.val > 4 & mean.val < 6 & sd.val < 2 ~ "neutral"),
         emotive = if_else(polarity == "neutral", 0L, 1L)) %>% tidyr::drop_na()

# select seed words
pos_valence_words <- warriner %>% filter(polarity == "positive") %>% pull(word)
neg_valence_words <- warriner %>% filter(polarity == "negative") %>% pull(word)
val_dict <- dictionary(list("positive" = pos_valence_words, "negative" = neg_valence_words))
saveRDS(val_dict, "data/fg09_warriner.rds")

# --------------------------------
# estimate local transformation matrix
# --------------------------------
text <- uk_data %>% 
  filter(party %in% c("Conservative", "Labour")) %>%
  mutate(text = iconv(text, from = "latin1", to = "UTF-8")) %>%
  distinct(text, .keep_all = TRUE) %>% tidyr::drop_na() %>% pull(text) %>% tolower()

# pre-processing
uk_toks <- tokens(text, remove_punct = T, remove_symbols = T, remove_numbers = T, remove_url = T, remove_separators = T)
feats <- dfm(uk_toks, verbose = TRUE) %>% dfm_trim(min_termfreq = 10) %>% featnames()

# subset tokens object to selected features
uk_toks <- tokens_select(uk_toks, pattern = feats, selection = "keep", padding = TRUE, min_nchar = 3)

# feature co-occurrence matrix
uk_fcm <- fcm(uk_toks, context = "window", count = "frequency", window = 6, tri = FALSE)

# local transform
transform_matrix <- compute_transform(x = uk_fcm, pre_trained = pre_trained, weighting = 500)
saveRDS(transform_matrix, "data/fg09_Amatrix.rds")

# --------------------------------
# build corpus
# --------------------------------
uk_data <- uk_data %>% mutate(date = as.Date(date, "%Y-%m-%d"), myear = zoo::as.yearmon(date, "%m/%Y"))
leaders <- uk_data %>% filter(leader == 1 & party %in% c("Conservative", "Labour")) %>% select(id_mp, text, myear, party) %>% unite(c(party, myear), col = "group", sep = '-', remove = FALSE)
cabinet <- uk_data %>% filter((shadow == 1 | cabinet == 1) & party %in% c("Conservative", "Labour")) %>% select(id_mp, text, myear, party) %>% unite(c(party, myear), col = "group", sep = '-', remove = FALSE)
backbenchers <- uk_data %>% filter(leader == 0 & cabinet == 0 & shadow == 0 & chair == 0 & party %in% c("Conservative", "Labour")) %>% select(id_mp, text, myear, party) %>% unite(c(party, myear), col = "group", sep = '-', remove = FALSE)

# corpora
leaders <- corpus(leaders$text, docvars = leaders[, c("id_mp", "myear", "party", "group")])
cabinet <- corpus(cabinet$text, docvars = cabinet[, c("id_mp", "myear", "party", "group")])
backbenchers <- corpus(backbenchers$text, docvars = backbenchers[, c("id_mp", "myear", "party", "group")])
corpora <- list("leaders" = leaders, "cabinet" = cabinet, "backbenchers" = backbenchers)
rm(leaders, cabinet, backbenchers)
saveRDS(corpora, "data/fg09_corpus.rds")

# -------------------------------------
# coverage of count-based vs. embeddings-based approach
# -------------------------------------

# subset corpus
uk_data_subset <- uk_data %>% filter(party %in% c("Conservative", "Labour")) %>% select(text, party)
corpus_all <- corpus(uk_data_subset$text, docvars = data.frame(party = uk_data_subset$party))

# tokenize
toks <- tokens(corpus_all, remove_punct = T, remove_symbols = T, remove_numbers = T, remove_separators = T)

# get tokenized target contexts
targets <- c("education", "nhs", "eu")
target_toks <- tokens_context(x = toks, pattern = targets, valuetype = "fixed", window = 6L, hard_cut = FALSE, case_insensitive = TRUE)

# document-feature matrix
target_dfm <- dfm(target_toks, tolower = TRUE)

# subset dfm to valence dictionary
val_dfm <- dfm_select(target_dfm, pattern = c(pos_valence_words, neg_valence_words), selection = "keep")

# proportion of dictionary with some overlap
length(colSums(val_dfm))/length(c(pos_valence_words, neg_valence_words))

# proportion of texts with 0 score
unscored_texts <- rownames(val_dfm)[apply(val_dfm, 1, function(x) all(x == 0))]
length(unscored_texts)/nrow(val_dfm)

# proportion of valence dictionary with words in pre-trained GloVe
prop.table(table(c(pos_valence_words, neg_valence_words) %in% rownames(pre_trained)))