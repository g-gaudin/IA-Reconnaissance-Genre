# Title     : packageCheckeR
# Objective : Check if R packages are ready to go
# Created by: Valentin ZELIONII
# Created on: 17/12/2019

args <- commandArgs(trailingOnly = TRUE)
path <- args[[1]][1]
setwd(path)
getwd()

require("warbleR")
require("Rraven")
require("devtools")

writeLines("List of loaded R packages:")
(.packages())
writeLines("All available warbleR functions:")
X <- c("warbleR", "Rraven")
invisible(lapply(X, library, character.only = TRUE))
ls("package:warbleR")