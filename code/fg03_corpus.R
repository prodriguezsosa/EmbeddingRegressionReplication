# --------------------------------
# setup
# --------------------------------

# libraries
library(dplyr)

# set working directory to the location of the master "EmbeddingRegressionReplication" folder
setwd("")

# --------------------------------
# load data
# --------------------------------
nyt_data <- readRDS("data/nyt_data.rds")
nyt_data$lead_paragraph[is.na(nyt_data$lead_paragraph)] <- nyt_data$snippet[is.na(nyt_data$lead_paragraph)] # in 2018 lead_paragraph is missing

# --------------------------------
# prepare corpus
# --------------------------------

# keep documents with targets of interest (no need to process every document)
nyt_data <- nyt_data %>% 
  select(text = lead_paragraph, year) %>%
  mutate(
    text = trimws(gsub('LEAD:', '', text)), # remove header banner
    text = iconv(text, from = "latin1", to = "UTF-8"), # encoding
    target = case_when(grepl(paste0('\\<', "trump", '\\>'), text, ignore.case = FALSE) ~ "trump", 
                       grepl(paste0('\\<', "Trump", '\\>'), text, ignore.case = FALSE) ~ "Trump")) %>% 
  tidyr::drop_na() %>% 
  filter(!(target == "Trump" & year < 2017)) %>% # keep president Trump
  select(text, target)

# corpus
corpus_nyt <- quanteda::corpus(nyt_data$text, docvars = data.frame(target = nyt_data$target))

# save
saveRDS(corpus_nyt, "data/fg03_corpus.rds")
