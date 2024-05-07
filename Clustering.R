install.packages("readxl")
install.packages("cluster")
install.packages("stats")
install.packages("data.table")
install.packages("dplyr")
install.packages("ggplot2")
install.packages("factoextra")
install.packages("NbClust")
install.packages("fpc")

# Load Required Libraries
library(readxl)
library(cluster)
library(stats)
library(dplyr)
library(data.table)
library(NbClust)
library(fpc)
library(ggplot2)
library(factoextra)

#Reading the dataset from excel file
dataset <- read_excel("C:/Users/Mohamed Sulaiman/Desktop/ML Exam/Whitewine_v6.xlsx")

#Ensure if the dataset is read properly
head(dataset, 10)

#Obtain the summary of the dataset
sum(is.na(dataset))
summary(is.na(dataset))

#Selecting first 10 attributes for analysis
dataset <- select(dataset, 1:11)

#Scaling the data for normalization
dataset_scaled <- scale(dataset)
print(dataset_scaled)

#Identifying outliers in the dataset
outliers = c()
for (i in 1:11) {
  lower_quantile <- quantile(dataset_scaled[, i], probs = 0.25)
  upper_quantile <- quantile(dataset_scaled[, i], probs = 0.75)
  inter_quantile_range <- upper_quantile - lower_quantile
  lower_outliers = which(dataset_scaled[, i] < lower_quantile - 1.5 * inter_quantile_range)
  upper_outliers = which(dataset_scaled[, i] > upper_quantile + 1.5 * inter_quantile_range)
  outliers = c(outliers, upper_outliers, lower_outliers)
}
outliers <- unique(outliers)

#Cleaning the dataset by removing outliers
dataset_cleaned <- dataset_scaled[-outliers, ]


#Conduct four autmoated tests after normalizing the dataset
#==========================================================
  #1. NBClust Method
  nb_results <- NbClust(dataset_cleaned, distance = "euclidean", min.nc = 2, max.nc = 10, method = "kmeans")
  print(nb_results)

#2. Elbow Method
elbow_results <- fviz_nbclust(dataset_cleaned, kmeans, method = "wss")
print(elbow_results)

#3. Gap Statistics Method
gap_results <- fviz_nbclust(dataset_cleaned, kmeans, method = "gap_stat")
print(gap_results)

#4.Silhouette Method
silhouette_results <- fviz_nbclust(dataset_cleaned, kmeans, method = "silhouette")
print(silhouette_results)

#===============================================================================
# Assign optimal number of clusters from above tests to k
k <- 2
kmeans_model <- kmeans(dataset_cleaned, centers = k)

# Get cluster centers
cluster_centers <- kmeans_model$centers

# Assign clusters
cluster_assignments <- kmeans_model$cluster

# Calculate internal evaluation metrics
BSS <- kmeans_model$betweenss
WSS <- kmeans_model$tot.withinss
TSS <- BSS + WSS

# Calculate ratios
ratio_BSS_WSS <- BSS / WSS
ratio_BSS_TSS <- BSS / TSS


# Display the results
print("Cluster Centers: ")
print(cluster_centers)

print("Cluster Assignments: ")
print(cluster_assignments)

cat("Between-Cluster Sum of Squares (BSS): ", BSS, "\n")

cat("Within-Cluster Sum of Squares (WSS): ", WSS, "\n")

cat("Ratio BSS/TSS: ", ratio_BSS_TSS, "\n")

cat("Ratio BSS/WSS: ", ratio_BSS_WSS, "\n")

# Perform Silhouette Test

# Calculate silhouette widths
silhouette_width <- silhouette(cluster_assignments, dist(dataset_cleaned))

# Plot silhouette plot
silhouette_plot <- fviz_silhouette(silhouette_width)

# Print average silhouette width score
avg_silhouette_width <- mean(silhouette_width[, "sil_width"])
cat("Average Silhouette Width:", avg_silhouette_width, "\n")

# Display silhouette plot
print(silhouette_plot)

# Visualize the clustering results using a cluster plot without numbers
cluster_plot <- fviz_cluster(kmeans_model, data = dataset_cleaned, geom = "point")

# Show the cluster plot
print(cluster_plot)

#========================== SECOND SUBTASK =====================================

# Performing PCA on the cleaned dataset
pca_result <- prcomp(dataset_cleaned, center = TRUE, scale. = TRUE)

# Calculating cumulative explained variance
cumulative_variance <- cumsum(pca_result$sdev^2 / sum(pca_result$sdev^2)) * 100
print(cumulative_variance)

# Selecting principal components for cumulative variance > 85%
pca_selected <- pca_result$x[, cumulative_variance <= 85]
print(pca_selected)

#Conducting four automated tests on PCA 
## Elbow Method 
elbow_graph <- fviz_nbclust(pca_selected, kmeans, method = "wss") +
  labs(title = "Determining Optimal Cluster Count using Elbow Method")
print(elbow_graph)

## Silhouette Method for Optimal Cluster Identification
silhouette_graph <- fviz_nbclust(pca_selected, kmeans, method = "silhouette") +
  labs(title = "Optimal Cluster Count using Silhouette Method")
print(silhouette_graph)

## NbClust for Multiple Cluster Validation Techniques
nb_results <- NbClust(pca_selected, min.nc = 2, max.nc = 10, method = "kmeans")
barplot(table(nb_results$Best.nc), xlab = "Number of Clusters", ylab = "Frequency",
        main = "NbClust Evaluation for PCA-Reduced Data")

# Using Gap Statistics
optimal_clusters <- as.numeric(names(which.max(table(nb_results$Best.nc))))
set.seed(123)
gap_stat <- clusGap(pca_selected, FUN = kmeans, nstart = 25, K.max = 10, B = 50)
plot(gap_stat, main = "Visualization of Gap Statistic for Cluster Validation")

BSS <- sum(pca_kmeans_result$size * apply(pca_kmeans_result$centers, 1, function(center) sum((center - colMeans(dataset_cleaned)) ^ 2)))
WSS <- pca_kmeans_result$tot.withinss
TSS <- BSS + WSS
ratio_BSS_TSS <- BSS / TSS

print(paste("BSS:", BSS))
print(paste("WSS:", WSS))
print(paste("Ratio BSS/TSS:", ratio_BSS_TSS))

# Print eigenvalues
cat("\nEigenvalues:\n")
print(pca_result$sdev^2)

# Print eigenvectors
cat("\nEigenvectors:\n")
print(pca_result$rotation)

optimal_k <- 2
kmeans_result <- kmeans(pca_selected, centers = optimal_k)
print(kmeans_result)

# Calculate Between-Cluster Sum of Squares (BSS) and Within-Cluster Sum of Squares (WSS)
BSS <- sum(kmeans_result$size * dist(kmeans_result$centers)^2)  # Calculate BSS
WSS <- kmeans_result$tot.withinss  # Calculate WSS

# Calculate the ratio of BSS to TSS (Total Sum of Squares)
TSS <- BSS + WSS
ratio_BSS_TSS <- BSS / TSS

# Print BSS, WSS, and BSS/TSS ratio
cat("BSS:", BSS, "\n")
cat("WSS:", WSS, "\n")
cat("Ratio BSS/TSS:", ratio_BSS_TSS, "\n")

#===============================================================================
# Cluster plot after PCA analysis

# Visualize the clustering results using a cluster plot without numbers
cluster_plot <- fviz_cluster(kmeans_result, data = pca_selected, geom = "point")

# Show the cluster plot
print(cluster_plot)

#===============================================================================
# Calinski-Harabasz Index
cluster_stats <- cluster.stats(dist(pca_selected), kmeans_result$cluster)
ch_index <- cluster_stats$ch
cat("Calinski-Harabasz Index:", ch_index, "\n")


