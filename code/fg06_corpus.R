# --------------------------------
# setup
# --------------------------------
library(dplyr)
library(quanteda)

# set working directory to the location of the master "EmbeddingRegressionReplication" folder
setwd("")

# --------------------------------
# load data
# --------------------------------

# speeches
cr <- readRDS("data/corpus_daily.rds") %>% 
  select(speech, party, session_id, lastname, firstname, chamber, state, district) %>% 
  filter(session_id %in% 111:114 & party %in% c('D', 'R')) %>%
  mutate(speech = tolower(iconv(speech, from = "latin1", to = "UTF-8")), 
         district = as.numeric(district)) %>%
  distinct(speech, .keep_all = TRUE) %>% tidyr::drop_na()

# nominate scores
nominate <- list()
for(i in 111:114){
  nominate[[length(nominate) + 1]] <- read.csv(paste0("data/", "HS", i, "_members.csv")) %>%
    select(session_id = congress, chamber, state = state_abbrev, party = party_code, district = district_code, bioname, nominate_dim1) %>%
    filter(chamber !="President" & party %in% c(100,200)) %>%
    mutate(lastname = toupper(trimws(gsub(",.*$", "", bioname))), firstname = trimws(gsub(".*,", "", bioname)),
           chamber = if_else(chamber == "House", "H", "S"),
           party = if_else(party == 100, "D", "R")) %>% 
    select(lastname, session_id, chamber, state, party, bioname, nominate_dim1) %>% tidyr::drop_na() 
}

nominate <- bind_rows(nominate)      

# merge
cr_corpus <- left_join(cr, nominate, by = c('session_id', 'chamber', 'state', 'party', 'lastname')) %>% tidyr::drop_na()

# create quanteda corpus
cr_corpus <- corpus(cr_corpus$speech, docvars = data.frame(nominate_dim1 = cr_corpus$nominate_dim1))

# save corpus
saveRDS(list("cr_corpus" = cr_corpus, "nominate" = nominate), "data/fg06_data.rds")

