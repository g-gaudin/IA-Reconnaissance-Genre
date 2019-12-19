# Title     : packageInstalleR
# Objective : Install all required R packages
# Created by: Valentin ZELIONII
# Created on: 16/12/2019

warbleRURL <- "https://cran.r-project.org/src/contrib/Archive/warbleR/warbleR_1.1.0.tar.gz"
install.packages(warbleRURL,repos=NULL, type="source")
install.packages("Rraven",repos = "http://cran.us.r-project.org")
install.packages("devtools",repos = "http://cran.us.r-project.org")