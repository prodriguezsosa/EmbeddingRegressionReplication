# --------------------------------
# setup
# --------------------------------

# libraries
library(dplyr)
library(ggplot2)
library(conText)
library(quanteda)

# set working directory to the location of the master "EmbeddingRegressionReplication" folder
setwd("")

# --------------------------------
# load data
# --------------------------------

# load corpus
empire_corpus <- readRDS("data/fg07_corpus.rds")

# pre-trained embeddings & transformation matrix (local)
pre_trained <- readRDS("data/word_vectors_6_300_5000.rds")  # cr + ps pre-trained embeddings
transform_matrix <- readRDS("data/A_local.rds")  # transformation matrix for cr + ps embeddings

# --------------------------------
# tokenize and get contexts
# --------------------------------

# tokenize
toks <- tokens(empire_corpus, remove_punct = T, remove_symbols = T, remove_numbers = T, remove_url = T, remove_separators = T)

# find contexts around empire
toks_empire <- tokens_context(toks, "empire", window = 6L, valuetype = "fixed", hard_cut = FALSE, rm_keyword = TRUE, verbose = TRUE)

#---------------------------------
# conText regression
#---------------------------------

# run regression for each period and user inner product
set.seed(2022L)
models <- lapply(unique(docvars(toks_empire, 'period')), function(j){
  conText(formula =  . ~ group, data = tokens_subset(toks_empire, period == j), pre_trained = pre_trained, transform = TRUE, transform_matrix = transform_matrix, bootstrap = TRUE, num_bootstraps = 100, confidence_level = 0.95, stratify = TRUE, permute = TRUE, num_permutations = 100, window = 6L, valuetype = 'fixed', case_insensitive = TRUE, hard_cut = FALSE, verbose = TRUE)
})

# save output
#saveRDS(models, 'data/fg07_models.rds')

#---------------------------------
# plot
#---------------------------------

# load results
models <- readRDS('data/fg07_models.rds')
period <- paste(seq(1935, 2009, 2), period_end = seq(1936, 2010, 2), sep = '-')
plot_tibble <- lapply(models, function(i) i@normed_coefficients) %>% do.call(rbind, .) %>% mutate(period = period)

# structural break points
library(strucchange)

## F statistics indicates one breakpoint
fs.norm <- Fstats(plot_tibble$normed.estimate ~ 1)
plot(fs.norm)
breakpoints(fs.norm)
lines(breakpoints(fs.norm))

cat('breakpoint in period', plot_tibble$period[fs.norm$breakpoint])

# plot
fg7 <- ggplot(plot_tibble) + 
  geom_line(aes(x = period, y = normed.estimate, group = 1), size = 2) +
  geom_line(aes(x = period, y = lower.ci, group = 1), color = 'gray50', size = 1, linetype = "dashed") +
  geom_line(aes(x = period, y = upper.ci, group = 1), color = 'gray50', size = 1, linetype = "dashed") +
  geom_vline(xintercept = plot_tibble$period[fs.norm$breakpoint], linetype = "dotted", color = 'red', size = 1) +
  geom_text(aes(x="1947-1948", label="\nStructural break", y=0.125), colour="black", angle=90, size=6) +
  xlab("") + 
  ylab(expression(paste('Norm of ', hat(beta)))) +
  scale_color_manual(values = c('no' = 'grey', 'yes' = 'blue')) +
  theme(axis.text.x = element_text(size=18, angle = 90, vjust = 0.5, hjust = 1),
        axis.text.y = element_text(size=18),
        axis.title.y = element_text(size=20, margin = margin(t = 0, r = 15, b = 0, l = 15)),
        legend.text=element_text(size=18),
        legend.position = "none",
        plot.margin=unit(c(1,1,0,0),"cm"))

ggsave(filename = "fg07.pdf", plot = fg7, height = 10, width = 15, path = './figures/', dpi = 1000)


#---------------------------------
# nearest neighbors
#---------------------------------

period <- paste(seq(1935, 2009, 2), period_end = seq(1936, 2010, 2), sep = '-')
pre <- c("1935-1936", "1937-1938", "1939-1940", "1941-1942", "1943-1944", "1945-1946")
post <- setdiff(period, c(pre, "1947-1948"))

# pre 1947 - 1948
toks_empire_pre <- tokens_subset(toks_empire, period %in% pre)
local_vocab_pre <- get_local_vocab(toks_empire_pre, pre_trained = pre_trained)
empire_pre_nns_ratio <- get_nns_ratio(x = toks_empire_pre, groups =  docvars(toks_empire_pre, 'group'), N = 20, numerator = 'American', candidates = local_vocab_pre, pre_trained = pre_trained, transform = TRUE, transform_matrix = transform_matrix, bootstrap = TRUE, num_bootstraps = 100, confidence_level = 0.95, permute = TRUE, num_permutations = 100, stem = FALSE, verbose = TRUE)
#saveRDS(empire_pre_nns_ratio, "data/fg08a_output.rds")

empire_pre_nns_ratio <- readRDS("data/fg08a_output.rds")
fg8a <- plot_nns_ratio(x = empire_pre_nns_ratio, horizontal = TRUE)
ggsave(filename = "fg08a.pdf", plot = fg8a, height = 4, width = 8, path = './figures/', dpi = 1000)


# post 1947 - 1948
toks_empire_post <- tokens_subset(toks_empire, period %in% post)
local_vocab_post <- get_local_vocab(toks_empire_post, pre_trained = pre_trained)
empire_post_nns_ratio <- get_nns_ratio(x = toks_empire_post, groups =  docvars(toks_empire_post, 'group'), N = 20, numerator = 'American', candidates = local_vocab_post, pre_trained = pre_trained, transform = TRUE, transform_matrix = transform_matrix, bootstrap = TRUE, num_bootstraps = 100, confidence_level = 0.95, permute = TRUE, num_permutations = 100, stem = FALSE, verbose = TRUE)
#saveRDS(empire_post_nns_ratio, "data/fg08b_output.rds")

empire_post_nns_ratio <- readRDS("data/fg08b_output.rds")
fg8b <- plot_nns_ratio(x = empire_post_nns_ratio, horizontal = TRUE)
ggsave(filename = "fg08b.pdf", plot = fg8b, height = 4, width = 8, path = './figures/', dpi = 1000)
