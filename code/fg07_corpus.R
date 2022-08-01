#--------------------------------
# setup
# --------------------------------

# libraries
library(dplyr)
library(quanteda)

# set working directory to the location of the master "EmbeddingRegressionReplication" folder
setwd("")

# --------------------------------
# load data
# --------------------------------

# congressional record speeches with empire
cr_data <- readRDS("data/corpus_full_cr.rds") %>% select(speech, period_start, period_end) %>% distinct(speech, .keep_all = TRUE) %>% tidyr::drop_na()
cr_data <- cr_data[grepl('empire', cr_data$speech, ignore.case = TRUE),]

# parliamentary speeches
ps_data <- readRDS("data/corpus_full_ps.rds") %>% select(text, year) %>% distinct(text, .keep_all = TRUE) %>% tidyr::drop_na()
ps_data <- ps_data[grepl('empire', ps_data$text, ignore.case = TRUE),]

#---------------------------------
# build corpus
#---------------------------------

# congressional records
cr <- cr_data %>%
  filter(period_start >= min(ps_data$year) & period_end <= max(ps_data$year)) %>%
  mutate(period_end = period_end - 1,
         period = paste(period_start, period_end, sep = '-'),
         group = 'American',
         text = iconv(speech, from = "latin1", to = "UTF-8")) %>%
  select(text, period, group)

# parliamentary speeches
period_start <- as.integer(unique(cr_data$period_start))
period_end <- as.integer(unique(cr_data$period_end)) - 1
period_labels <- paste(period_start, period_end, sep = "-")

ps <- ps_data %>% mutate(period = NA, group = 'British')
for(j in 1:length(period_labels)){
  ps$period[ps$year >= period_start[j] & ps$year <= period_end[j]] <- period_labels[j]
}

ps <- ps %>% filter(!is.na(period)) %>% select(text, period, group)

# join both corpora
empire_corpus <- rbind(cr, ps)

# build quanteda corpus
empire_corpus <- corpus(tolower(empire_corpus$text), docvars = data.frame(period = empire_corpus$period, group = empire_corpus$group))
saveRDS(empire_corpus, "data/fg07_corpus.rds")

