

# --- Using FactoMineR package ---
install.packages("FactoMineR")
install.packages("ca")
install.packages("factoextra")


library(ca)
library(factoextra)
library(FactoMineR)

data("USArrests")
result = CA(USArrests)



fviz_ca_biplot(result, repel = TRUE)


res.ca <- ca(USArrests, graph = FALSE)

# --- Extract Eigenvalues and Create a Scree Plot ---
# extract eigenvalues
eig <- get_eigenvalue(res.ca)
eig

# visualize eigenvalues
fviz_eig(res.ca)

# --- Row and column profiles ---
row.profiles <- get_ca_row(res.ca)
row.profiles

col.profiles <- get_ca_col(res.ca)
col.profiles

# --- Visualization of Row and Column Profiles ---
fviz_ca_row(res.ca)
fviz_ca_col(res.ca)

# --- Create a Biplot ---
# We can also create a biplot to visualize the relationship between the rows and columns of the analysis.
# And from the previous example, we know that fviz_ca_biplot() function creates a scatter plot of the rows and columns of the analysis
# with arrows indicating the strength and direction of the relationship between them.
fviz_ca_biplot(res.ca)
