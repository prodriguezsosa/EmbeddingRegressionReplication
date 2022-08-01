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
# load corpus
# --------------------------------
cr <- readRDS("data/corpus_daily.rds") %>%
  filter(session_id %in% 111:114 & party %in% c('D', 'R')) %>%
  select(speech, party, gender) %>%
  mutate(speech = tolower(iconv(speech, from = "latin1", to = "UTF-8"))) %>%
  distinct(speech, .keep_all = TRUE) %>% tidyr::drop_na() %>%
  select(speech, party, gender)

# quanteda corpus
cr_corpus <- corpus(cr$speech, docvars = data.frame(party = cr$party, gender = cr$gender))

# pre-trained embeddings & transformation matrix
pre_trained <- readRDS("data/stanford-glove/glove.rds")  # pre-trained
transform_matrix <- readRDS("data/stanford-glove/khodakA.rds")  # pre-trained

# --------------------------------
# pre-processing
# --------------------------------
cr_toks <- tokens(cr_corpus, remove_punct = T, remove_symbols = T, remove_numbers = T, remove_url = T, remove_separators = T)

#---------------------------------
# conText regressions
#---------------------------------

# define target words
targets <- c('and', 'but', 'also', 'abortion', 'marriage', 'immigration')

# initialize model vectors
models <- vector('list', length = length(targets)) %>% setNames(targets)

# run conText regressions
set.seed(2022L)
for(target in targets){
  # too many instances of these stopwords, too slow. we use a sample instead.
  if(target %in% c('and','but','also')){
    model1 <- conText(formula =  as.formula(paste0(target, ' ~ party + gender')), data = tokens_sample(cr_toks, size = 0.05*length(cr_toks), replace = FALSE), pre_trained = pre_trained, transform = TRUE, transform_matrix = transform_matrix, bootstrap = TRUE, num_bootstraps = 100, stratify = TRUE, permute = TRUE, num_permutations = 100, window = 6L, valuetype = 'fixed', case_insensitive = TRUE, hard_cut = FALSE, verbose = TRUE)
  }else{
    model1 <- conText(formula =  as.formula(paste0(target, ' ~ party + gender')), data = cr_toks, pre_trained = pre_trained, transform = TRUE, transform_matrix = transform_matrix, bootstrap = TRUE, num_bootstraps = 100, stratify = TRUE, permute = TRUE, num_permutations = 100, window = 6L, valuetype = 'fixed', case_insensitive = TRUE, hard_cut = FALSE, verbose = TRUE)
  }
  models[[target]] <- model1
  cat("done with", target, "\n")
}

# save output
#saveRDS(models,'data/fg05_output.rds')

# load data
models <- readRDS('data/fg05_output.rds')
normed_cofficients <- lapply(targets, function(target) cbind(target = target, models[[target]]@normed_coefficients)) %>% bind_rows()
normed_cofficients <- normed_cofficients %>% mutate(target = factor(target, levels = c('also', 'but', 'and', 'abortion', 'marriage', 'immigration')),
                                        coefficient = factor(coefficient, levels = c('party_R', 'gender_M')))

#---------------------------------
# plot
#---------------------------------
fg5 <- ggplot(normed_cofficients, aes(coefficient, shape = coefficient)) +
  geom_pointrange(aes(x = target, y = normed.estimate,
                      ymin = lower.ci,
                      ymax = upper.ci), lwd = 1, position = position_dodge(width = 1/2), fill = "WHITE") +
  xlab('') +
  ylab(expression(paste('Norm of ', hat(beta),'s'))) +
  scale_shape_manual(values=c(2,20), labels = c("Republican", "Male")) +
  theme(axis.text.x = element_text(size=18, angle = 90, vjust = 0.5, hjust = 1),
        axis.text.y = element_text(size=18),
        axis.title.y = element_text(size=20, margin = margin(t = 0, r = 15, b = 0, l = 15)),
        legend.text=element_text(size=18),
        legend.title=element_blank(),
        legend.key=element_blank(),
        legend.position = "top",
        legend.spacing.x = unit(0.25, 'cm'),
        plot.margin=unit(c(1,1,0,0),"cm"))

ggsave(filename = "fg05.pdf", plot = fg5, height = 10, width = 12, path = './figures/', dpi = 1000)

#---------------------------------
# nearest neighbors using model
#---------------------------------

# just for immigration as a function of party
model1 <- conText(formula =  immigration ~ party, data = cr_toks, pre_trained = pre_trained, transform = TRUE, transform_matrix = transform_matrix, bootstrap = TRUE, num_bootstraps = 100, confidence_level = 0.95, stratify = TRUE, permute = TRUE, num_permutations = 100, window = 6L, valuetype = 'fixed', case_insensitive = TRUE, hard_cut = FALSE, verbose = FALSE)

# extract coefficients
immigrationR <- model1["(Intercept)",] + model1["party_R",]
immigrationD <- model1["(Intercept)",]

# nearest neighbors
nns(immigrationR, pre_trained = pre_trained, N = 10, candidates = model1@features, as_list = FALSE)
nns(immigrationD, pre_trained = pre_trained, N = 10, candidates = model1@features, as_list = FALSE)

#---------------------------------
# nearest contexts
#---------------------------------

# tokenize docs
immig_toks <- tokens_context(cr_toks, pattern = 'immigration', window = 6L, hard_cut = FALSE)

# build dfm
immig_dfm <- dfm(immig_toks)

# build dem
immig_dem <- dem(x = immig_dfm, pre_trained = pre_trained, transform = TRUE, transform_matrix = transform_matrix, verbose = TRUE)

# compute ALC embeddings for each party
immig_wv_party <- dem_group(immig_dem, groups = immig_dem@docvars$party)

# find nearest contexts (limit each party's ncs to their own contexts)
ncs(x = immig_wv_party["D",], contexts_dem = immig_dem[immig_dem@docvars$party == "D",], contexts = immig_toks[docvars(immig_toks, 'party') == "D",], N = 5, as_list = FALSE)
ncs(x = immig_wv_party["R",], contexts_dem = immig_dem[immig_dem@docvars$party == "R",], contexts = immig_toks[docvars(immig_toks, 'party') == "R",], N = 5, as_list = FALSE)
