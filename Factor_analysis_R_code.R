
install.packages("psych")
library(psych)
# install.packages("ggplot2")
library(ggplot2)


# R-এর bfi ডেটাসেটটি factor_analyzer এর ডেটাসেটের মতই
# For demonstration, let's use a sample dataset from the library
data(bfi)
# কলামগুলো বাদ দিন এবং NA ভ্যালু পরিষ্কার করুন
df <- bfi[, 1:25]
df <- na.omit(df)

# Bartlett's test প্রয়োগ করুন
# আউটপুটের মধ্যে Bartlett's test এর ফলাফল পাওয়া যাবে
cortest.bartlett(df)

# KMO test প্রয়োগ করুন
KMO(df)

# ফ্যাক্টর সংখ্যা নির্ধারণ করতে Scree Plot তৈরি করুন
# fa() ফাংশন নিজে থেকেই scree plot তৈরি করতে পারে
# অথবা, eigenvalues বের করে ম্যানুয়ালি প্লট করা যায়
fa_model_initial <- fa(df, rotate = "varimax")
print(fa_model_initial$e.values) # Eigenvalues দেখা
plot(fa_model_initial$e.values, type = "b", main = "Scree Plot", xlab = "Factors", ylab = "Eigenvalue")

# নির্দিষ্ট সংখ্যক ফ্যাক্টর (যেমন ৬) দিয়ে ফ্যাক্টর অ্যানালাইসিস করুন
fa_model <- fa(df, nfactors = 6, rotate = "varimax")

# ফ্যাক্টর লোডিং দেখুন
print("Factor Loadings:")
print(fa_model$loadings)

# প্রতিটি ফ্যাক্টরের Variance দেখুন
# summary(fa_model) এর আউটপুটে SS loadings, Proportion Var, Cumulative Var আকারে Variance দেখা যায়
print("Factor Variance:")
print(fa_model$Vaccounted)
