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
# basic pre-processing of corpus
#---------------------------------

# targets
targets <- c('Trump', 'Clinton')

# keep only documents where the target words appear
trump_corpus <- corpus_nyt[grep('Trump', corpus_nyt$lead_paragraph, fixed = TRUE, ignore.case = FALSE), c('lead_paragraph', 'year')] %>% distinct(lead_paragraph, .keep_all = TRUE) %>% filter(year %in% c(2011:2014,2017:2020)) %>% rename(text = lead_paragraph) %>% mutate(target = 'trump', year = as.integer(year)) # uppercase matters here
clinton_corpus <- corpus_nyt[grep('Clinton', corpus_nyt$lead_paragraph, fixed = TRUE, ignore.case = FALSE), c('lead_paragraph', 'year')] %>% distinct(lead_paragraph, .keep_all = TRUE) %>% filter(year %in% c(2011:2014,2017:2020)) %>% rename(text = lead_paragraph) %>% mutate(target = 'clinton', year = as.integer(year)) # uppercase matters here
sub_corpus <- rbind(trump_corpus, clinton_corpus)

# basic preprocessing of text
sub_corpus$text <- sub_corpus$text %>%
  gsub('Trump', 'toi', .) %>% # replace mentions of Trump with TOI (target of interest)
  gsub('Clinton', 'toi', .) %>% # replace mentions of Clinton with TOI (target of interest)
  gsub('LEAD:', '', .) %>% # remove header banner
  gsub("[^[:alpha:]]", " ", .) %>% # remove all non-alpha characters
  str_replace_all("\\b\\w{1,2}\\b", "") %>% # remove 1-2 letter words
  str_replace_all("^ +| +$|( ) +", "\\1") %>% # remove excess white space
  tolower() # lowercase

#---------------------------------
# conText regression
#---------------------------------

# add dummy variables distinguishing pre-/post-election years and trump/clinton mentions
sub_corpus <- sub_corpus %>% mutate(post_election = if_else(year>2014, 1L, 0L))
sub_corpus <- sub_corpus %>% mutate(trump = if_else(target == 'trump', 1L, 0L))
sub_corpus <- sub_corpus %>% mutate(interaction = trump*post_election)

# transform into quanteda corpus
sub_corpus <- corpus(sub_corpus$text, docvars = sub_corpus[,c("year", "target", "post_election", "trump", "interaction")])
toks <- tokens(sub_corpus)

# run regression
set.seed(2022L)
model1 <- conText(formula =  toi ~ trump + post_election + interaction, data = toks, pre_trained = pre_trained, transform = TRUE, transform_matrix = transform_matrix, bootstrap = TRUE, num_bootstraps = 1000, confidence_level = 0.95, stratify = TRUE, permute = TRUE, num_permutations = 100, window = 6, valuetype = 'fixed', case_insensitive = TRUE, hard_cut = FALSE, verbose = TRUE)

# save results
#saveRDS(model1, 'data/fg04_output.rds')

#---------------------------------
# visualize
#---------------------------------

# read results
model1 <- readRDS('data/fg04_output.rds')
model1@normed_coefficients

# coefficient plot
plot_tibble <- model1@normed_coefficients %>% mutate(coefficient = c("Trump", "Post_Election", "Trump x \n Post_Election")) %>% mutate(coefficient = factor(coefficient, levels = coefficient))
fg4 <- ggplot(plot_tibble, aes(x = coefficient, y = normed.estimate)) +
  geom_pointrange(aes(ymin = lower.ci, ymax = upper.ci), size = 1) +
  labs(y = expression(paste('Norm of ', hat(beta),'s'))) +
  geom_text(aes(label=c('***', '***', '***')), position=position_dodge(width=0.9), hjust=0.5, vjust = c(0, 0, 0), size = 8) +
  coord_flip() +
  ylim(0,0.15) +
  theme(axis.text.x = element_text(size=18, vjust = 0.5, margin = margin(t = 15, r = 0, b = 15, l = 0)),
        axis.text.y = element_text(size=18),
        axis.title.y = element_blank(),
        axis.title.x = element_text(size=20),
        plot.margin=unit(c(1,1,0,0),"cm"))

ggsave(filename = "fg04.pdf", plot = fg4, height = 10, width = 12, path = './figures/', dpi = 1000)

