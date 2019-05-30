library(tidyverse)
library(lattice)
library(caret)
library(tidyr)
library(ModelMetrics)
library(purrr)
library(rpart)
library(randomForest)
library(recipes)
library(varhandle)


setwd("~/2.Education/1.R/Harvardx/Captsone/ml-10m")
ratings <- read.table(text = gsub("::", "\t", readLines("ratings.dat")),
                      col.names = c("userId", "movieId", "rating", "timestamp"))
movies <- str_split_fixed(readLines("movies.dat"), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                           title = as.character(title),
                                           genres = as.character(genres))
movielens <- left_join(ratings, movies, by = "movieId")
# Validation set will be 10% of MovieLens data
set.seed(1)
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

##wrangling data
edx <- edx %>% extract(title, c("title_noY", "year"), regex = "^(.*) \\(([0-9 \\-]*)\\)$", remove = F)
edx$timestamp <- as_datetime(edx$timestamp)
edx$year <- as.numeric(edx$year)
edx <- edx %>% mutate(year_rat = year(timestamp))

validation <- validation %>% extract(title, c("title_noY", "year"), regex = "^(.*) \\(([0-9 \\-]*)\\)$", remove = F)
validation$timestamp <- as_datetime(validation$timestamp)
validation$year <- as.numeric(validation$year)
validation <- validation %>% mutate(year_rat = year(timestamp))


### Adjusting the columns to numeric values
genr <- data.frame(unique(edx$genres))
vec <- c()
n <- as.numeric(length(genr))
for (i in 1:n){
  vec[i] <- i  
  }
genr <- genr %>% mutate(genreId = vec)
names(genr) <- c("genres", "genresId")

edx <- edx %>% left_join(genr, by = "genres")
validation <- validation %>% left_join(genr, by = "genres")

### running a PCA
ind <- c(1:3,6,7,9,10)
edx_prc <- edx[ind]
edxpca <- prcomp(edx_prc)
edxpca$rotation

## running a correlation matrix
ind <- c(1,2,6,7,9,10)
edx_cor <- edx[ind]
cor(edx_cor)

#############################################################################
###First method: linnear regression
names(edx)
str(edx)
fit_lm <- lm(rating ~ userId+movieId+timestamp+genresId, data = edx)
edx_hat_lm <- predict(fit_lm, newdata=validation)
rmse(validation$rating,edx_hat_lm)
##rmse_reported: 1.06

#############################################################################
###Second method: knn

ks <- seq(3, 700003, 100000)
accurac <- map_df(ks, function(k){
  fit_knn <- knn3(rating ~ userId+movieId+timestamp+genresId, data = edx, k = 1000000)
  edx_hat_knn <- predict(fit_knn, validation, type = "class")
  rmse_acc <- rmse(validation$rating,edx_hat_knn)
  })
max(accurac)
ks[is.max(accurac)]

 
#############################################################################
###Third method: classification tree
fit_rt <- rpart(rating ~ year + year_rev , data = edx)
edx_hat_rt <- predict(fit_rt, validation)
rmse_acc <- rmse(validation$rating,edx_hat_rt)
##rmse_reported: 0.9571388

#############################################################################
###Fourth method: random forest tree
fit_rf <- randomForest(rating ~ . , data = edx)

#############################################################################
###Fifth method: neural net
max = apply(edx , 2 , max)
min = apply(edx, 2 , min)
scaled = as.data.frame(scale(data, center = min, scale = max - min))

