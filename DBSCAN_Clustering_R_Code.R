install.packages("mlbench")
install.packages("dbscan")
install.packages("ggplot2")


# Step 1: Load the necessary libraries
library(mlbench)
library(dbscan)
library(ggplot2)

# Step 2: Generate spiral data using the correct function name
# This function creates two intertwined spirals.
spirals_data <- mlbench.spirals(n = 300, cycles = 1, sd = 0.05)

# Step 3: Standardize the data
X_scaled <- scale(spirals_data$x)

# Step 4: Apply DBSCAN
# We might need to adjust 'eps' for the new shape
db <- dbscan(X_scaled, eps = 0.2, minPts = 5)

# Step 5: Create a data frame for plotting
plot_data <- data.frame(
  Feature1 = X_scaled[, 1],
  Feature2 = X_scaled[, 2],
  Cluster = as.factor(db$cluster)
)

# Step 6: Plot the results
ggplot(plot_data, aes(x = Feature1, y = Feature2, color = Cluster)) +
  geom_point(size = 3) +
  labs(title = "DBSCAN Clustering",
       x = "Feature 1",
       y = "Feature 2") +
  theme_minimal() +
  scale_color_discrete(name = "Cluster")



#Visualizing Geometric Intuition

install.packages("ggforce")

# Load the required libraries
library(ggplot2)
library(ggforce) # For drawing circles

# Create data frames for the points
core_points <- data.frame(x = c(2, 3, 2.5), y = c(2, 2, 2.8), label = "Core Point")
border_points <- data.frame(x = 3.5, y = 2.2, label = "Border Point")
noise_points <- data.frame(x = 5, y = 5, label = "Noise Point")

# Epsilon neighborhood radius
epsilon <- 1.0

# Plotting with ggplot2
ggplot() +
  # Draw the epsilon circles around core points using ggforce::geom_circle
  geom_circle(data = core_points, aes(x0 = x, y0 = y, r = epsilon),
              fill = "green", alpha = 0.2, linetype = "dashed", color = "darkgreen") +
  
  # Plot the core points
  geom_point(data = core_points, aes(x = x, y = y, color = label), size = 5) +
  
  # Plot the border point
  geom_point(data = border_points, aes(x = x, y = y, color = label), size = 5) +
  
  # Plot the noise point
  geom_point(data = noise_points, aes(x = x, y = y, color = label), size = 6, shape = 4, stroke = 1.5) +
  
  # Set plot limits and aspect ratio
  coord_fixed(xlim = c(0, 6), ylim = c(0, 6)) +
  
  # Customize colors and labels
  scale_color_manual(name = "Point Type",
                     values = c("Core Point" = "darkgreen", "Border Point" = "orange", "Noise Point" = "red")) +
  
  # Add titles and theme
  labs(title = "DBSCAN Geometric Intuition in R", x = NULL, y = NULL) +
  theme_minimal(base_size = 14)

