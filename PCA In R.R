
install.packages("caTools")
library(caTools)
install.packages("caret")
library(caret)
install.packages("ggplot2")
library(ggplot2) # For advanced plotting


df <- data.frame(
  Height = c(170, 165, 180, 175, 160, 172, 168, 177, 162, 158),
  Weight = c(65, 59, 75, 68, 55, 70, 62, 74, 58, 54),
  Age = c(30, 25, 35, 28, 22, 32, 27, 33, 24, 21),
  # factor() ব্যবহার করে লেবেল দেওয়া হলো (0 = Female, 1 = Male)
  Gender = factor(c(1, 0, 1, 1, 0, 1, 0, 1, 0, 0), levels = c(0, 1), labels = c("Female", "Male"))
)

print("Original DataFrame:")
print(df)


X <- df[, c("Height", "Weight", "Age")]
y <- df$Gender



X_scaled <- scale(X)


pca_result <- prcomp(X_scaled, center = TRUE, scale. = TRUE)

X_pca <- pca_result$x[, 1:2]


library(caTools)
set.seed(42) # random_state=42 এর সমতুল্য
split <- sample.split(y, SplitRatio = 0.7) # 70% training data

X_train <- subset(X_pca, split == TRUE)
X_test <- subset(X_pca, split == FALSE)
y_train <- subset(y, split == TRUE)
y_test <- subset(y, split == FALSE)


# glm (Generalized Linear Model) ফাংশন এবং family="binomial" ব্যবহার করা হয়
model <- glm(y_train ~ ., data = as.data.frame(X_train), family = "binomial")

# প্রেডিকশন তৈরি করা
predicted_probs <- predict(model, newdata = as.data.frame(X_test), type = "response")
# সম্ভাবনার উপর ভিত্তি করে ক্লাস লেবেল নির্ধারণ করা
y_pred <- ifelse(predicted_probs > 0.5, "Male", "Female")
y_pred <- factor(y_pred, levels = c("Female", "Male"))




conf_matrix <- table(Actual = y_test, Predicted = y_pred)
print("Confusion Matrix:")
print(conf_matrix)



heatmap(conf_matrix, Rowv = NA, Colv = NA, 
        col = cm.colors(256), scale="column", margins=c(5,10),
        main = "Confusion Matrix Heatmap")




par(mfrow=c(1, 2)) # ১টি রো এবং ২টি কলামে প্লট দেখানোর জন্য

# Plot Before PCA
plot(X_scaled[, 1], X_scaled[, 2], 
     col = as.numeric(y) + 1, # রঙ নির্ধারণ
     pch = 19, # Symbol type
     xlab = "Original Feature 1 (Height)", 
     ylab = "Original Feature 2 (Weight)",
     main = "Before PCA")
legend("bottomright", legend = levels(y), col = c(2,3), pch = 19)


# Plot After PCA
plot(X_pca[, 1], X_pca[, 2],
     col = as.numeric(y) + 1, # রঙ নির্ধারণ
     pch = 19, # Symbol type
     xlab = "Principal Component 1", 
     ylab = "Principal Component 2",
     main = "After PCA")
legend("bottomright", legend = levels(y), col = c(2,3), pch = 19)

# গ্রাফিক্স ডিভাইস রিসেট করা
par(mfrow=c(1,1))

