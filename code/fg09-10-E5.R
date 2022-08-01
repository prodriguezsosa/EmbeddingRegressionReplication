# --------------------------------
# setup
# --------------------------------

# libraries
library(dplyr)
library(conText)
library(ggplot2)
library(quanteda)
library(text2vec)
library(tidyr)
library(zoo)

# set working directory to the location of the master "EmbeddingRegressionReplication" folder
setwd("")

# --------------------------------
# load data
# --------------------------------

transform_matrix <- readRDS("data/fg09_Amatrix.rds")
corpora <- readRDS("data/fg09_corpus.rds")
pre_trained <- readRDS("data/stanford-glove/glove.rds")
val_dict <- readRDS("data/fg09_warriner.rds")
pos_valence_words <- val_dict$positive
neg_valence_words <-val_dict$negative

# --------------------------------
# compute ALC embeddings
# --------------------------------

# define target terms
targets <- c("education", "nhs", "eu")
out <- list()
out_inner <- list()
out_count <- list()
for(i in names(corpora)){
  for(target in targets){
  
  #---------------------------------
  # build a (tokenized) corpus of contexts
  #---------------------------------
  
  # tokenize corpus
  toks <- tokens(corpora[[i]], remove_punct = T, remove_symbols = T, remove_numbers = T, remove_separators = T)
  
  # build a tokenized corpus of contexts sorrounding the target
  target_toks <- tokens_context(x = toks, pattern = target, valuetype = "fixed", window = 6L, hard_cut = FALSE, case_insensitive = TRUE)
  
  #---------------------------------
  # compute ALC embeddings
  #---------------------------------
  
  # build a document-feature-matrix
  target_dfm <- dfm(target_toks, tolower = TRUE)
  
  # build a document-embedding-matrix
  target_dem <- dem(x = target_dfm, pre_trained = pre_trained, transform = TRUE, transform_matrix = transform_matrix, verbose = TRUE)
  
  # num target by group
  freq <- target_dem@docvars %>% group_by(id_mp) %>% count(group) %>% ungroup() %>% group_by(group) %>% summarise(n = mean(n))
  
  # docids of docs that were embedded
  docids <- target_dem@Dimnames$docs
  
  # aggregate over grouping variable
  target_dem_grouped <- dem_group(target_dem, groups = target_dem@docvars$group)
  
  #---------------------------------
  # valence
  #---------------------------------

  # mean cosine similarity to positive valence terms
  group_pos_val <- sim2(target_dem_grouped, y = pre_trained[intersect(pos_valence_words, rownames(pre_trained)),], method = 'cosine', norm ="l2")
  group_pos_val <- rowMeans(group_pos_val)
  group_pos_val <- tibble(group = factor(names(group_pos_val)), pos_val = unname(group_pos_val))
  
  # mean cosine similarity to negative valence terms
  group_neg_val <- sim2(target_dem_grouped, y = pre_trained[intersect(neg_valence_words, rownames(pre_trained)),], method = 'cosine', norm = "l2")
  group_neg_val <- rowMeans(group_neg_val)
  group_neg_val <- tibble(group = factor(names(group_neg_val)), neg_val = unname(group_neg_val))

  # join results
  result <- left_join(group_pos_val, group_neg_val, by = "group") %>% mutate(val = (1)*pos_val + (-1)*neg_val) %>% select(group, val)
  rm(group_pos_val, group_neg_val)
  
  #---------------------------------
  # valence (using inner product)
  #---------------------------------
  
  # mean cosine similarity to positive valence terms
  group_pos_val_inner <- sim2(target_dem_grouped, y = pre_trained[intersect(pos_valence_words, rownames(pre_trained)),], method = 'cosine', norm = "none")
  group_pos_val_inner <- rowMeans(group_pos_val_inner)
  group_pos_val_inner <- tibble(group = factor(names(group_pos_val_inner)), pos_val = unname(group_pos_val_inner))
  
  # mean cosine similarity to negative valence terms
  group_neg_val_inner <- sim2(target_dem_grouped, y = pre_trained[intersect(neg_valence_words, rownames(pre_trained)),], method = 'cosine', norm = "none")
  group_neg_val_inner <- rowMeans(group_neg_val_inner)
  group_neg_val_inner <- tibble(group = factor(names(group_neg_val_inner)), neg_val = unname(group_neg_val_inner))
  
  # join results
  result_inner <- left_join(group_pos_val_inner, group_neg_val_inner, by = "group") %>% mutate(val = (1)*pos_val + (-1)*neg_val) %>% select(group, val)
  rm(group_pos_val_inner, group_neg_val_inner)
  
  #---------------------------------
  # count-based approach
  #---------------------------------
  count_dfm <- dfm_lookup(target_dfm, dictionary = val_dict)
  result_count <- cbind(convert(count_dfm, to = "data.frame"), docvars(count_dfm)) %>% group_by(group) %>% summarize(pos = mean(positive), neg = mean(negative)) %>% mutate(val = (1)*pos + (-1)*neg) %>% select(group, val)
  
  #---------------------------------
  # output
  #---------------------------------

  # other covs
  covs <- docvars(corpora[[i]]) %>% select(party, myear, group) %>% unique()
  result <- left_join(result, covs, by = 'group') %>% mutate(corpus = i)
  result <- left_join(result, freq, by = "group")
  result <- result %>% mutate(target = target) %>% select(corpus, target, party, myear, group, val, n)
  
  # inner product
  result_inner <- left_join(result_inner, covs, by = 'group') %>% mutate(corpus = i)
  result_inner <- left_join(result_inner, freq, by = "group")
  result_inner <- result_inner %>% mutate(target = target) %>% select(corpus, target, party, myear, group, val, n)
  
  # count based
  result_count <- left_join(result_count, covs, by = 'group') %>% mutate(corpus = i)
  result_count <- left_join(result_count, freq, by = "group")
  result_count <- result_count %>% mutate(target = target) %>% select(corpus, target, party, myear, group, val, n)
  
  # output
  out[[length(out) + 1]] <- result
  out_inner[[length(out_inner) + 1]] <- result_inner
  out_count[[length(out_count) + 1]] <- result_count
  rm(covs, freq, result, target_dem, target_dem_grouped, target_dfm, target_toks, toks, docids, count_dfm, result_count, result_inner)
  cat("done with target:", target, "\n")
  }
  cat("done with corpus:", i, "\n")
}

# save results
#saveRDS(out, "data/fgE5_output.rds")
#saveRDS(out_inner, "data/fg09_output.rds)
#saveRDS(out_count, "data/fg10_output.rds")

out <- readRDS("data/fgE5_output.rds")
out_inner <- readRDS("data/fg09_output.rds")
out_count <- readRDS("data/fg10_output.rds")

#---------------------------------
# Figure E5 (cosine similarity)
#---------------------------------

# collapse results into data.frame
out_df <- bind_rows(out) %>% 
  filter(corpus!="leaders") %>%
  group_by(party, target) %>%
  mutate(val = scale(val)[,1]) %>% # scale within party-target
  ungroup() %>%
  select(- group) %>%
  complete(expand(.,corpus, target, party), myear) %>%
  mutate(myear = as.factor(myear),
         corpus = factor(corpus, levels = c("cabinet", "backbenchers")),
         target = factor(target, levels = targets)) %>% 
  arrange(myear)


#---------------------------------
# loess fitted values
#---------------------------------

# empty column
out_df <- out_df %>% mutate(fit = NA)

# loop through corpus-target-party combinations
for(i in unique(out_df$corpus)){
  for(j in unique(out_df$target)){
    for(k in unique(out_df$party)){
      sub_data <- out_df[out_df$corpus == i & out_df$target == j & out_df$party == k,]
      out_df$fit[out_df$corpus == i & out_df$target == j & out_df$party == k] <- loess(val ~ as.numeric(myear),sub_data, control = loess.control(surface = "direct"), span = 0.15) %>% predict(as.numeric(sub_data$myear), se = FALSE)
    }
  }
}

#---------------------------------
# compute rolling correlation 
# (for conservatives)
#---------------------------------
roll_corr <- list()
myear <- unique(out_df$myear)
for(j in unique(out_df$target)){
    
    # sub df
    sub_df <- data.frame(myear = 1:length(out_df$myear[out_df$corpus == "cabinet" & out_df$party == "Conservative" & out_df$target == j]),
                         cabinet = out_df$fit[out_df$corpus == "cabinet" & out_df$party == "Conservative" & out_df$target == j],
                         backbenchers = out_df$fit[out_df$corpus == "backbenchers" & out_df$party == "Conservative" & out_df$target == j])
    
    # rolling correlation
    width <- 12 # window width
    cor_target <- rollapply(sub_df, width=width, function(x) cor(x[,2],x[,3], use = "complete.obs"), by.column=FALSE)
    roll_corr <- append(roll_corr, list(data.frame(corpus = "Correlation \n (rolling)", 
                                                   target = j,
                                                   party = "Correlation",
                                                   myear = myear[width:length(myear)], # start in myear[width], this is the average of the past 12 months
                                                   val = 3*cor_target, # this is a hacky way of zooming into the plot (i.e. since we can't independently set coord_cartesian for each facet below)
                                                   n = NA,
                                                   fit = NA)))

}

# combine to create plot df
plot_df <- rbind(out_df, do.call(rbind, roll_corr))

# large points fcn
# source: https://stackoverflow.com/questions/61096323/how-can-i-change-the-colour-line-size-and-shape-size-in-the-legend-independently
large_points <- function(data, params, size) {
  # Multiply by some number
  data$size <- data$size * 2
  draw_key_point(data = data, params = params, size = size)
}

# labeller
yaxis <- c(Conservative = "Valence\n(Conservative)", Labour = "Valence\n(Labour)", Correlation = "Correlation\n(Conservative)")

fgE5 <- ggplot(plot_df) + 
  geom_line(data = plot_df %>% filter(corpus!="Correlation \n (rolling)"), aes(x = myear, y = fit, group = corpus, color = corpus), size = 1.5) +
  geom_smooth(data = plot_df %>% filter(corpus=="Correlation \n (rolling)"), aes(x = myear, y = val, group = corpus, color = corpus), method = "loess", formula = 'y ~ x', se = FALSE, size = 1.5) +
  geom_point(aes(x = myear, y = val, group = corpus, shape = corpus), alpha =1/10, key_glyph = large_points) +
  geom_text(data = subset(plot_df, target == "eu" & party == "Conservative"), aes(x="Jun 2015", label="\nReferendum bill", y=-0.65), colour="black", angle=90, size=7) +
  geom_text(data = subset(plot_df, target == "eu" & party == "Conservative"), aes(x="May 2010", label="\nElection", y=-0.65), colour="black", angle=90, size=7) +
  geom_text(data = subset(plot_df, target == "education" & party == "Conservative"), aes(y = 0, label="Valence = 0\n", x= "Dec 2005"), colour="black", angle=0, size=7) +
  geom_text(data = subset(plot_df, target == "education" & party == "Correlation"), aes(y = 0, label="Correlation = 0\n", x= "Dec 2005"), colour="black", angle=0, size=7) +
  scale_x_discrete(breaks = unique(plot_df$myear)[seq(1,length(plot_df$myear),12)]) +
  geom_vline(xintercept = "May 2010", linetype = "dotted", color = "black", size = 0.5) +
  geom_vline(xintercept = "Jun 2015", linetype = "dotdash", color = "black", size = 0.5) +
  scale_size(guide = 'none') +
  scale_shape_manual(labels = c('Cabinet','Backbenchers', 'Correlation'),
                     values = c(21,24,23),
                     guide = "none") +
  scale_colour_manual(labels = c('Backbenchers','Cabinet', 'Correlation'),
                      values = c("black", "gray70", "gray35"),
                      guide=guide_legend(reverse=FALSE)) +
  scale_fill_manual(labels = c('Cabinet','Backbenchers', 'Correlation'),
                    values = c("black", "gray70", "gray35"),
                    guide="none") +
  geom_hline(yintercept = 0, linetype = "dashed", size = 0.5, color = "black") +
  ylab('Valence \n (scaled within party and target word)') +
  coord_cartesian(ylim = c(-3, 3)) +
  facet_grid(party ~ target, scales = 'free_y', switch = 'y', labeller = labeller(target = label_value, party = yaxis)) +
  theme(panel.grid.major = element_blank(), 
    panel.grid.minor = element_blank(),
    panel.background = element_rect(fill = 'grey100', colour = 'black'),
    axis.ticks.y = element_blank(),
    axis.text.x = element_text(size=18, angle = 90, hjust = 1, vjust = 0.5),
    axis.text.y = element_blank(),
    axis.title.y = element_blank(),
    axis.title.x = element_blank(),
    legend.text=element_text(size=18),
    legend.title=element_blank(),
    legend.key=element_blank(),
    legend.position = "top",
    legend.spacing.x = unit(0.25, 'cm'),
    strip.text = element_text(size = 20))

ggsave(filename = "fgE5.pdf", plot = fgE5, height = 10, width = 14, path = './figures/', dpi = 1000)


# -------------------------------------
# Figure 10 (count-based)
# -------------------------------------

# row bind results
plot_count_df <- bind_rows(out_count) %>% 
  group_by(party, target) %>%
  mutate(val = scale(val)[,1]) %>%
  ungroup() %>%
  mutate(myear = as.factor(myear),
         corpus = factor(corpus, levels = c("leaders", "cabinet", "backbenchers")),
         target = factor(target, levels = targets)) %>% 
  arrange(myear)

fg10 <- ggplot(subset(plot_count_df, corpus %in% c("cabinet", "backbenchers")), aes(x = myear, y = val, group = corpus, color = corpus)) + 
  geom_point(aes(color = corpus, shape = corpus, fill = corpus), alpha =1/10, key_glyph = large_points) +
  geom_smooth(method = "loess", span = 0.15, formula = 'y ~ x', se = FALSE) +
  geom_text(data = subset(plot_count_df, target == "eu" & party == "Conservative"), aes(x="Jun 2015", label="\nReferendum bill", y=-0.65), colour="black", angle=90, size=7) +
  geom_text(data = subset(plot_count_df, target == "eu" & party == "Conservative"), aes(x="May 2010", label="\nElection", y=-0.65), colour="black", angle=90, size=7) +
  geom_text(data = subset(plot_count_df, target == "education" & party == "Conservative"), aes(y = 0, label="Valence = 0\n", x= "Jul 2004"), colour="black", angle=0, size=7) +
  scale_x_discrete(breaks = unique(plot_count_df$myear)[seq(1,length(plot_count_df$myear),12)]) +
  geom_vline(xintercept = "May 2010", linetype = "dotted", color = "black", size = 0.5) +
  geom_vline(data = subset(plot_count_df, target == "eu" & corpus %in% c("cabinet", "backbenchers")), aes(xintercept = "Jun 2015"), linetype = "dotdash", color = "black", size = 0.5) +
  scale_size(guide = 'none') +
  scale_shape_manual(labels = c('Conservative','Labour'),
                     values = c(21,24),
                     guide = "none") +
  scale_colour_manual(labels = c('Cabinet','Backbenchers'),
                      #values = c("#0087DC", "#CC0000"),
                      values = c("gray60", "black"),
                      guide=guide_legend(reverse=TRUE)) +
  scale_fill_manual(labels = c('Conservative','Labour'),
                    #values = c("#0087DC", "#CC0000"),
                    values = c("black", "gray60"),
                    guide="none") +
  geom_hline(yintercept = 0, linetype = "dashed", size = 0.5, color = "black") +
  ylab('Valence \n (scaled within party and target word)') +
  coord_cartesian(ylim = c(-2, 1.5)) +
  facet_grid(party ~ target) +
  theme(
    panel.grid.major = element_blank(), 
    panel.grid.minor = element_blank(),
    panel.background = element_rect(fill = 'grey100', colour = 'black'),
    axis.ticks.y = element_blank(),
    axis.text.x = element_text(size=20, angle = 90, hjust = 1, vjust = 0.5),
    axis.text.y = element_blank(),
    axis.title.y = element_text(size=24),
    axis.title.x = element_blank(),
    legend.text=element_text(size=24),
    legend.title=element_blank(),
    legend.key=element_blank(),
    legend.position = "top",
    legend.spacing.x = unit(0.25, 'cm'),
    strip.text = element_text(size = 24))

ggsave(filename = "fg10.pdf", plot = fg10, height = 10, width = 14, path = './figures/', dpi = 1000)

# -------------------------------------
# Figure 9 (inner product)
# -------------------------------------

# collapse results into data.frame
out_df <- bind_rows(out_inner) %>% 
  filter(corpus!="leaders") %>%
  group_by(party, target) %>%
  mutate(val = scale(val)[,1]) %>% # scale within party-target
  ungroup() %>%
  select(- group) %>%
  complete(expand(.,corpus, target, party), myear) %>%
  mutate(myear = as.factor(myear),
         corpus = factor(corpus, levels = c("cabinet", "backbenchers")),
         target = factor(target, levels = targets)) %>% 
  arrange(myear)

#---------------------------------
# loess fitted values
#---------------------------------

# empty column
out_df <- out_df %>% mutate(fit = NA)

# loop through corpus-target-party compbinations
for(i in unique(out_df$corpus)){
  for(j in unique(out_df$target)){
    for(k in unique(out_df$party)){
      sub_data <- out_df[out_df$corpus == i & out_df$target == j & out_df$party == k,]
      out_df$fit[out_df$corpus == i & out_df$target == j & out_df$party == k] <- loess(val ~ as.numeric(myear),sub_data, control = loess.control(surface = "direct"), span = 0.15) %>% predict(as.numeric(sub_data$myear), se = FALSE)
    }
  }
}

#---------------------------------
# compute rolling correlation 
# (for conservatives)
#---------------------------------
roll_corr <- list()
myear <- unique(out_df$myear)
for(j in unique(out_df$target)){
  
  # sub df
  sub_df <- data.frame(myear = 1:length(out_df$myear[out_df$corpus == "cabinet" & out_df$party == "Conservative" & out_df$target == j]),
                       cabinet = out_df$fit[out_df$corpus == "cabinet" & out_df$party == "Conservative" & out_df$target == j],
                       backbenchers = out_df$fit[out_df$corpus == "backbenchers" & out_df$party == "Conservative" & out_df$target == j])
  
  # rolling correlation
  width <- 12 # window width
  cor_target <- rollapply(sub_df, width=width, function(x) cor(x[,2],x[,3], use = "complete.obs"), by.column=FALSE)
  roll_corr <- append(roll_corr, list(data.frame(corpus = "Correlation \n (rolling)", 
                                                 target = j,
                                                 party = "Correlation",
                                                 myear = myear[width:length(myear)], # start in myear[width], this is the average of the past 12 months
                                                 val = 3*cor_target, # this is a hacky way of zooming into the plot (i.e. since we can't independently set coord_cartesian for each facet below)
                                                 n = NA,
                                                 fit = NA)))
  
}

# combine to create plot df
plot_df <- rbind(out_df, do.call(rbind, roll_corr))

#---------------------------------
# plot
#---------------------------------

fg9 <- ggplot(plot_df) + 
  geom_line(data = plot_df %>% filter(corpus!="Correlation \n (rolling)"), aes(x = myear, y = fit, group = corpus, color = corpus), size = 1.5) +
  geom_smooth(data = plot_df %>% filter(corpus=="Correlation \n (rolling)"), aes(x = myear, y = val, group = corpus, color = corpus), method = "loess", formula = 'y ~ x', se = FALSE, size = 1.5) +
  geom_point(aes(x = myear, y = val, group = corpus, color = corpus, shape = corpus, fill = corpus), alpha =1/10, key_glyph = large_points) +
  geom_text(data = subset(plot_df, target == "eu" & party == "Conservative"), aes(x="Jun 2015", label="\nReferendum bill", y=-0.65), colour="black", angle=90, size=7) +
  geom_text(data = subset(plot_df, target == "eu" & party == "Conservative"), aes(x="May 2010", label="\nElection", y=-0.65), colour="black", angle=90, size=7) +
  geom_text(data = subset(plot_df, target == "education" & party == "Conservative"), aes(y = 0, label="Valence = 0\n", x= "Dec 2005"), colour="black", angle=0, size=7) +
  geom_text(data = subset(plot_df, target == "education" & party == "Correlation"), aes(y = 0, label="Correlation = 0\n", x= "Dec 2005"), colour="black", angle=0, size=7) +
  scale_x_discrete(breaks = unique(plot_df$myear)[seq(1,length(plot_df$myear),12)]) +
  geom_vline(xintercept = "May 2010", linetype = "dotted", color = "black", size = 0.5) +
  geom_vline(xintercept = "Jun 2015", linetype = "dotdash", color = "black", size = 0.5) +
  scale_size(guide = 'none') +
  scale_shape_manual(labels = c('Cabinet','Backbenchers', 'Correlation'),
                     values = c(21,24,23),
                     guide = "none") +
  scale_colour_manual(labels = c('Backbenchers', 'Cabinet', 'Correlation'),
                      values = c("black", "gray70", "gray35"),
                      guide=guide_legend(reverse=FALSE)) +
  scale_fill_manual(labels = c('Cabinet','Backbenchers', 'Correlation'),
                    values = c("black", "gray70", "gray35"),
                    guide="none") +
  geom_hline(yintercept = 0, linetype = "dashed", size = 0.5, color = "black") +
  ylab('Valence \n (scaled within party and target word)') +
  coord_cartesian(ylim = c(-3, 3)) +
  facet_grid(party ~ target, scales = 'free_y', switch = 'y', labeller = labeller(target = label_value, party = yaxis)) +
  theme(panel.grid.major = element_blank(), 
        panel.grid.minor = element_blank(),
        panel.background = element_rect(fill = 'grey100', colour = 'black'),
        axis.ticks.y = element_blank(),
        axis.text.x = element_text(size=20, angle = 90, hjust = 1, vjust = 0.5),
        axis.text.y = element_blank(),
        axis.title.y = element_blank(),
        axis.title.x = element_blank(),
        legend.text=element_text(size=24),
        legend.title=element_blank(),
        legend.key=element_blank(),
        legend.position = "top",
        legend.spacing.x = unit(0.25, 'cm'),
        strip.text = element_text(size = 24))

ggsave(filename = "fg09.pdf", plot = fg9, height = 10, width = 14, path = './figures/', dpi = 1000)

