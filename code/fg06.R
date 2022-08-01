# --------------------------------
# setup
# --------------------------------
library(dplyr)
library(quanteda)
library(conText)
library(text2vec)
library(ggplot2)
library(directlabels)

# set working directory to the location of the master "EmbeddingRegressionReplication" folder
setwd("")

# --------------------------------
# load data
# --------------------------------

# pre-trained embeddings & transformation matrix
pre_trained <- readRDS("data/stanford-glove/glove.rds")  # pre-trained
transform_matrix <- readRDS("data/stanford-glove/khodakA.rds")  # pre-trained

# congressional record corpus and nominate scores
fg06_data <- readRDS("data/fg06_data.rds")
cr_corpus <- fg06_data[["cr_corpus"]]
nominate <- fg06_data[["nominate"]]
rm(fg06_data)

# --------------------------------
# pre-process and run conText
# --------------------------------

# tokenize
cr_toks <- tokens(cr_corpus, remove_punct = T, remove_symbols = T, remove_numbers = T, remove_separators = T)

# subset to features that occur 10 or more times and consist of at least 3 characters
feats <- dfm(cr_toks, verbose = TRUE) %>% dfm_trim(min_termfreq = 10) %>% featnames()
cr_toks <- tokens_select(cr_toks, pattern = feats, selection = "keep", padding = TRUE, min_nchar = 3)

# run conText regression
set.seed(2022L)
model1 <- conText(formula = immigration ~ nominate_dim1, data = cr_toks, pre_trained = pre_trained, 
                  transform = TRUE, transform_matrix = transform_matrix, 
                  bootstrap = TRUE, num_bootstraps = 100, confidence_level = 0.95,
                  permute = TRUE, num_permutations = 100, window = 6, case_insensitive = TRUE, 
                  verbose = TRUE)

# look at full range
percentiles <- quantile(docvars(cr_corpus)$nominate_dim1, probs = seq(0.05,0.95,0.05))
target_wvs <- lapply(percentiles, function(i) model1["(Intercept)",] + i*model1["nominate_dim1",])
target_nns <- lapply(target_wvs, function(i) find_nns(i, pre_trained = pre_trained, N = 10, candidates = model1@features))
nns_ranked <- lapply(target_wvs, function(i) {
  sim <- text2vec::sim2(x = pre_trained[model1@features,], y = matrix(i, nrow = 1), method = "cosine", norm = 'l2')[,1]
  sim <- sim[order(-sim)]
  return(sim)
})

# focuse on specific features
features <- c("bipartisan", "reform", "illegals", "enforce", "amend")
out <- vector("list", length(features)) %>% setNames(features)
for(i in features){
  rank <- lapply(nns_ranked, function(j) which(names(j) == i))
  value <- lapply(nns_ranked, function(j) j[which(names(j) == i)])
  result <- tibble(feature = i, percentile = names(unlist(rank)), rank = unname(unlist(rank)), value = unname(unlist(value)))
  out[[i]] <- result
}

# bind results
plot_df <- bind_rows(out) %>% 
  mutate(percentile = factor(percentile, levels = paste0(seq(5,95,5), "%")))

# feature count by group
docvars(cr_toks, "perc") <- cut(docvars(cr_toks, "nominate_dim1"), breaks = c(percentiles, 1), labels = names(percentiles), right = FALSE)
docvars(cr_toks)$perc[is.na(docvars(cr_toks)$perc)] <- '5%'
cr_dfm_select <- dfm_select(dfm(cr_toks, tolower = TRUE), pattern = features, selection = "keep")
target_freq <- quanteda.textstats::textstat_frequency(cr_dfm_select, groups = docvars(cr_dfm_select, 'perc')) %>% as.data.frame() %>% select(feature, frequency, percentile = group)

# add to plot df
plot_df <- left_join(plot_df, target_freq, by = c('feature', 'percentile'))
plot_df <- plot_df %>% mutate(frequency = if_else(is.na(frequency), 0, frequency)) %>% group_by(feature) %>% mutate(frequency = frequency/sum(frequency)) %>% ungroup()
plot_df <- plot_df %>% mutate(percentile = factor(percentile, levels = paste0(seq(5,95,5), "%")))

# median dem/rep
nominate$perc <- cut(nominate$nominate_dim1, breaks = c(percentiles, 1), labels = names(percentiles), right = FALSE)
nominate$perc[is.na(nominate$perc)] <- '5%' 
nominate$perc <- as.integer(gsub("%", "", nominate$perc))
med_dem <- median(nominate$perc[nominate$party == "D"])
med_rep <- median(nominate$perc[nominate$party == "R"])

# scientific numbering
# source: https://stackoverflow.com/questions/10762287/how-can-i-format-axis-labels-with-exponents-with-ggplot2-and-scales
scientific_10 <- function(x) {
  parse(text=gsub("e", " %*% 10^", scales::scientific_format()(x)))
}

# plot
fg6 <- ggplot(plot_df, aes(x = percentile, y = value, group = feature, linetype = feature, color = feature)) +
  geom_smooth(method = 'loess', formula = 'y ~ x', se = FALSE, size = 2) +
  geom_vline(xintercept = paste0(med_dem, "%"), linetype = "dotted", size = 1, color = "black") +
  geom_vline(xintercept = paste0(med_rep, "%"), linetype = "dotted", size = 1, color = "black") +
  geom_text(aes(x=paste0(med_dem, "%"), label="\nMedian Democrat", y=0.5), colour="black", angle=90, size=7) +
  geom_text(aes(x=paste0(med_rep, "%"), label="\nMedian Republican", y=0.5), colour="black", angle=90, size=7) +
  labs(y = "cosine similarity between \n predicted ALC embedding and feature", x = "percentile of DW-NOMINATE \n (higher values, more Conservative)") +
  geom_dl(aes(label = feature), method = list(dl.trans(x = x - 2.6, y = ifelse(plot_df$feature %in% c("enforce", "illegals", "amend"), y + 0.5, y)), "last.points", cex = 1.75)) +
  scale_color_grey(start = 0.2, end = 0.7) +
  scale_linetype_manual(values = c("longdash","dotted","dotdash","twodash", "dashed")) +
  theme(axis.text.x = element_text(size=18, angle = 90, hjust = 1, vjust = 0.5),
        axis.text.y = element_text(size=18, angle = 90, hjust = 0.5, vjust = 0.5),
        axis.title.y = element_text(size=20, vjust = 5),
        axis.title.x = element_text(size=20, vjust = -5),
        legend.position = "none",
        plot.margin = unit(c(1.5,1.5,1.5,1.5), "cm"))

ggsave(filename = "fg06.pdf", plot = fg6, height = 10, width = 15, path = './figures/', dpi = 1000)

