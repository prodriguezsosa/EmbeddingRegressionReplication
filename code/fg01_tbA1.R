#---------------------------------
# Setup
#---------------------------------

library(dplyr)
library(stringr)
library(quanteda)
library(pbapply)
library(readtext)
library(progress)
library(ggplot2)
library(stargazer)
library(tidyr)
library(conText)
library(text2vec)

# set working directory to the location of the master "EmbeddingRegressionReplication" folder
setwd("")

# paths
path_to_rodman_data <- "./data/rodman/original_replication_material/word2vec_time/data/"
path_to_rodman_codebook <- "./data/rodman/original_replication_material/word2vec_time/codebook/"

#---------------------------------
# RODMAN DATA (RODMAN REPLICATION FILES)
#---------------------------------
files <- as.list(list.files(path_to_rodman_data)) %>% .[grepl("processed",.)]
eras <- str_extract_all(files, pattern = "\\d+") %>% unlist
texts <- pblapply(files, function(x) readtext(paste0(path_to_rodman_data, x)) %>% select(text) %>% str_split("\n") %>% unlist() %>% tibble(text = ., era = str_extract(x, pattern = "\\d+"))) %>% bind_rows()
corpora <- corpus(texts$text, docvars = data.frame("era" = texts$era))

# Data from gold standard model of corpus
supervised <- read.csv(paste0(path_to_rodman_data, "supervised_category_means.csv")) %>%   #eras 2-6
  select(-X) %>%
  rename(prop = mean)

hand_coded <- read.csv(paste0(path_to_rodman_codebook, "hand_coded.csv")) %>%              #era 1
  select(code) %>%
  rename(category = code)

era1_count <- as.numeric(length(hand_coded[,1]))
hand_coded <- hand_coded %>% group_by(category) %>% 
  tally() %>%
  mutate(prop = n/era1_count) %>%
  select(category, prop) %>%
  mutate(era = 1) %>% 
  filter(category == 20 | 
           category == 40 |
           category == 60 |
           category == 61)
german <- c(41, 0, 1)
hand_coded <- rbind(hand_coded, german) %>% arrange(category) %>% select (era, category, prop)

gold_standard <- rbind(hand_coded, supervised) %>% arrange(category, era)

# Data from naive time word2vec model of corpus
naive <- read.csv(paste0(path_to_rodman_data, "naive_mean_output.csv"), check.names = FALSE, header = TRUE)
naive <- naive %>% mutate(category = c(20, 40, 41, 60, 61))
naive <- naive %>% gather(key = "year", value = "1855:2005", -category) %>% 
  rename("mean" = "1855:2005")
naive <- naive %>% arrange(category, year)

# Data from overlapping word2vec model of corpus
overlap <- read.csv(paste0(path_to_rodman_data, "overlap_mean_output.csv"), check.names = FALSE, header = TRUE)
overlap <- overlap %>% mutate(category = c(20, 40, 41, 60, 61))
overlap <- overlap %>% gather(key = "year", value = "1855:2005", -category) %>% 
  rename("mean" = "1855:2005")
overlap <- overlap %>% arrange(category, year)

# Data from aligned word2vec model of corpus
aligned <- read.csv(paste0(path_to_rodman_data, "aligned_mean_output.csv"), check.names = FALSE, header = TRUE)
aligned <- aligned %>% mutate(category = c(20, 40, 41, 60, 61))
aligned <- aligned %>% gather(key = "year", value = "1880:2005", -category) %>% 
  rename("mean" = "1880:2005")
aligned <- rbind(filter(naive, year == "1855"), aligned)
aligned <- aligned %>% arrange(category, year)

# Data from chronologically trained word2vec model of corpus
chrono <- read.csv(paste0(path_to_rodman_data, "chrono_mean_output.csv"), check.names = FALSE, header = TRUE)
chrono <- chrono %>% mutate(category = c("20", "40", "41", "60", "61", "social"))
chrono <- chrono %>% gather(key = "year", value = "1855:2005", -category) %>% 
  rename("mean" = "1855:2005") %>% filter(category != "social") %>% mutate(year = as.numeric(year))
chrono <- chrono %>% arrange(category, year)

#---------------------------------
# A LA CARTE EMBEDDINGS
#---------------------------------

# pre-trained embeddings & transformation matrix
pre_trained <- readRDS("data/stanford-glove/glove.rds")  # pre-trained
transform_matrix <- readRDS("data/stanford-glove/khodakA.rds")  # pre-trained

# parameters
window_size <- 10
targets <- c("equality", "gender", "treaty", "german", "race", "african_american", "social")

# tokenize corpus
rodman_toks <- tokens(corpora, remove_punct = F, remove_symbols = F, remove_numbers = F, remove_separators = T)

# build a tokenized corpus of contexts sorrounding the target
rodman_toks_target <- tokens_context(x = rodman_toks, pattern = targets, valuetype = "fixed", window = 10L, case_insensitive = FALSE, hard_cut = FALSE, verbose = TRUE)

# number of instances by era (TABLE A.1)
table(docvars(rodman_toks_target)$era, docvars(rodman_toks_target)$pattern)
table(docvars(rodman_toks)$era)

# build a document-feature-matrix
rodman_dfm <- dfm(rodman_toks_target, tolower = FALSE)

# build a document embedding matrix
rodman_dem <- dem(x = rodman_dfm, pre_trained = pre_trained, transform = TRUE, transform_matrix = transform_matrix, verbose = TRUE)

# group by patterns and era
rodman_dem_agg <- dem_group(rodman_dem, groups = paste(rodman_dem@docvars$pattern,rodman_dem@docvars$era, sep = "-"))

#---------------------------------
# COSINE DISTANCE W. EQUALITY
#---------------------------------
topics <- setdiff(targets, c("equality", "social"))

cosine_sim_topics <- vector("list", length(topics)) %>% setNames(topics)
for(topic in topics){
  cosine_sim_era <- vector("list", length(eras)) %>% setNames(eras)
  for(era in eras){
    cosine_sim_era[[era]] <- sim2(x = matrix(rodman_dem_agg[paste("equality", era, sep="-"),], nrow = 1, ncol = ncol(rodman_dem_agg)), y = matrix(rodman_dem_agg[paste(topic, era, sep="-"),], nrow = 1, ncol = ncol(rodman_dem_agg)), method = "cosine", norm = "l2")
    }
  cosine_sim_topics[[topic]] <- tibble(category = topic, year = as.numeric(eras), mean = unlist(cosine_sim_era))
}

alacarte <- do.call(rbind, cosine_sim_topics)
rownames(alacarte) <- NULL
alacarte$category <- recode_factor(alacarte$category, `gender` = "20", `treaty` = "40", `german` = "41", `race` = "60", `african_american` = "61")
alacarte <- alacarte %>% arrange(category, year)

#---------------------------------
# RODMAN FIG 3 EQUIVALENT PLOT
#---------------------------------

# Create data frame for figure 3 ALC
fig3_data <- gold_standard %>% rename("baseline" = "prop") %>% arrange(category, era)
fig3_data <- cbind(fig3_data, select(alacarte, mean)) 
fig3_data <- fig3_data %>% rename("alc_value" = "mean") %>%
  mutate(sq_alc_value = (alc_value*100)^2) 

z_baseline <- scale(fig3_data$baseline)
z_alc_value <- scale(fig3_data$alc_value)
z_sq_alc_value <- scale(fig3_data$sq_alc_value)

fig3_data <- cbind(fig3_data, z_baseline, z_alc_value, z_sq_alc_value)

# Create data frame for figure 3 Chrono

fig3_data <- cbind(fig3_data, select(chrono, mean)) 
fig3_data <- fig3_data %>% rename("chrono_value" = "mean") %>%
  mutate(sq_chrono_value = (chrono_value*100)^2) 

z_chrono_value <- scale(fig3_data$chrono_value)
z_sq_chrono_value <- scale(fig3_data$sq_chrono_value)

fig3_data <- cbind(fig3_data, z_chrono_value, z_sq_chrono_value)

# Rename the category variable for use as facet titles
fig3_data$category <- plyr::mapvalues(fig3_data$category,
                                      from = c("20", "40", "41", "60", "61"),
                                      to = c("Gender",
                                             "International Relations",
                                             "Germany",
                                             "Race",
                                             "African American"))

# Reorder the categories to change the order of plots
fig3_data$category <- factor(fig3_data$category,
                             c("Race",
                               "African American",
                               "International Relations",
                               "Germany",
                               "Gender"))

# plot
fg1 <- fig3_data %>% 
  select(era, category, z_alc_value, z_chrono_value, z_baseline) %>% 
  pivot_longer(
    cols = -c(era, category),
    names_to = "model",
    names_prefix = "z_",
    values_to = "value"
  ) %>% 
  mutate(model = recode(model, alc_value = "ALC", baseline = "GS", chrono_value = "CHR")) %>%
  ggplot(aes(x = era, y = value, color = category)) +
  geom_line(aes(linetype = model), size = 1.5) +
  facet_wrap(~category, nrow=3, ncol=2) +
  labs(x = "", y = "z-score normalized model output", linetype = "Model") +
  scale_x_continuous(breaks = seq(1, 7, 1),
                     labels = c("1855-\n1879 ", "1880-\n1904 ", "1905-\n1929 ", "1930-\n1954 ", 
                                "1955-\n1979 ", "1980-\n2004 ", "2005-\n2016 ")) +
  scale_linetype_manual(labels = c("ALC", "Chrono", "GS"), values=c("longdash", "dotdash", "solid")) +
  scale_color_manual(values=c("#fb6a4a", "#cb181d", "#008837","#5e3c99","#67a9cf"), guide="none") +
  theme(
    axis.text.x = element_text(size=18, vjust = 0.5, angle = 90),
    axis.text.y = element_text(size=18),
    axis.title.y = element_text(size=20, vjust = 2),
    axis.title.x = element_text(size=20, vjust = -2),
    strip.text = element_text(size = 18),
    legend.position = "top",
    legend.spacing.x = unit(0.5, 'cm'),
    legend.text=element_text(size=18),
    legend.title=element_blank(),
    legend.key=element_blank(),
    legend.key.size = unit(2, 'cm'),
    plot.margin=unit(c(1,1,1,1),"cm")
    )

ggsave(filename = "fg01.pdf", plot = fg1, height = 12, width = 10, path = './figures/', dpi = 1000)

#---------------------------------
# ANOVA
#---------------------------------
# Data for table 2, statistical comparison of baseline to word2vec models
table2_data <- fig3_data %>% rename(z_alacarte = z_alc_value,
                                    z_alacarte_sq = z_sq_alc_value,
                                    alacarte = alc_value,
                                    alacarte_sq = sq_alc_value) %>%
  mutate(naive = naive$mean,
         overlap = overlap$mean,
         aligned = aligned$mean,
         chrono = chrono$mean) %>%
  mutate(naive_sq = (naive*100)^2,
         overlap_sq = (overlap*100)^2,
         aligned_sq = (aligned*100)^2,
         chrono_sq = (chrono*100)^2) 

z_naive <- scale(table2_data$naive)
z_overlap <- scale(table2_data$overlap)
z_aligned <- scale(table2_data$aligned)
z_chrono <- scale(table2_data$chrono)

z_naive_sq <- scale(table2_data$naive_sq)
z_overlap_sq <- scale(table2_data$overlap_sq)
z_aligned_sq <- scale(table2_data$aligned_sq)
z_chrono_sq <- scale(table2_data$chrono_sq)

table2_data <- cbind(table2_data, z_naive, z_overlap, z_aligned, z_chrono,
                     z_naive_sq, z_overlap_sq, z_aligned_sq, z_chrono_sq)

table2_data <- table2_data %>% mutate(alacarte_dev = abs(z_baseline - z_alacarte),
                                      alacarte_sq_dev = abs(z_baseline - z_alacarte_sq),
                                      naive_dev = abs(z_baseline - z_naive),
                                      naive_sq_dev = abs(z_baseline - z_naive_sq),
                                      overlap_dev = abs(z_baseline - z_overlap),
                                      overlap_sq_dev = abs(z_baseline - z_overlap_sq),
                                      aligned_dev = abs(z_baseline - z_aligned),
                                      aligned_sq_dev = abs(z_baseline - z_aligned_sq),
                                      chrono_dev = abs(z_baseline - z_chrono),
                                      chrono_sq_dev = abs(z_baseline - z_chrono_sq))

# Calculating variance, squared variance, and correlation
dev_scores <- c(sum(table2_data$naive_dev), sum(table2_data$overlap_dev),
                sum(table2_data$chrono_dev), sum(table2_data$aligned_dev),
                sum(table2_data$alacarte_dev))

sq_dev_scores <- c(sum(table2_data$naive_sq_dev), sum(table2_data$overlap_sq_dev),
                   sum(table2_data$chrono_sq_dev), sum(table2_data$aligned_sq_dev),
                   sum(table2_data$alacarte_sq_dev))

naive_cor <- cor(table2_data$z_baseline, table2_data$z_naive)
overlap_cor <- cor(table2_data$z_baseline, table2_data$z_overlap)
chrono_cor <- cor(table2_data$z_baseline, table2_data$z_chrono)
aligned_cor <- cor(table2_data$z_baseline, table2_data$z_aligned)
alacarte_cor <- cor(table2_data$z_baseline, table2_data$z_alacarte)

# Table with model comparison statistics
comparison_stats <- data.frame("Model" = c("naive", "overlap", "chrono", "aligned", "alacarte"), 
                               "Deviance" = dev_scores, 
                               "Squared Deviance" = sq_dev_scores,
                               "Correlation" = c(naive_cor, overlap_cor, chrono_cor, aligned_cor, alacarte_cor))

stargazer(comparison_stats, digits = 3, summary = F, rownames = T)

#---------------------------------
# COMPARE NEAREST NEIGHBORS
#---------------------------------
# for this part you will need Python 3.6 and the gensim library
library(reticulate)
library(Matrix)

model_names <- list('model2_of_1855.model', 'model3_of_1880.model', 'model4_of_1905.model', 'model5_of_1930.model', 'model6_of_1955.model', 'model7_of_1980.model', 'model8_of_2005.model') %>% setNames(eras)
targets_nns_era <- vector('list', length = length(eras)) %>% setNames(eras)
pb <- progress_bar$new(total = length(eras))
for(era in eras){

# load rodman chronological model
gensim <- import("gensim") # import the gensim library
chrono_model = gensim$models$Word2Vec$load(paste0(path_to_rodman_data, model_names[[era]]))
embeds <- chrono_model$wv$syn0
rownames(embeds) <- chrono_model$wv$index2word

targets_nns <- vector('list', length = length(targets)) %>% setNames(targets)
for(target in targets){
  

  # alc nns
  alc_nns <- nns(x = matrix(rodman_dem_agg[paste(target, era, sep="-"),], nrow = 1, ncol = ncol(rodman_dem_agg)), pre_trained = pre_trained, N = 5, candidates = setdiff(rodman_dem_agg@features,target), as_list = FALSE)
  
  # rodman nns
  chrono_nns <- nns(x = matrix(embeds[target,], nrow = 1, ncol = ncol(embeds)), pre_trained = embeds, N = 5, candidates = setdiff(rownames(embeds), target), as_list = FALSE)
  
  # store
  targets_nns[[target]] <- tibble('CHR' = chrono_nns$feature, 'ALC' = alc_nns$feature)
}
targets_nns_era[[era]] <- targets_nns
rm(chrono_model, embeds)
pb$tick()
}

# check
targets_nns_era[['1855']]
targets_nns_era[['2005']]


