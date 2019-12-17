# Title     : voiceAnalyzeR
# Objective : Analyze voice samples and save their parameters
# Created by: Valentin ZELIONII
# Created on: 17/12/2019

args <- commandArgs(trailingOnly = TRUE)
path <- args[[1]][1]
setwd(path)
getwd()

require("warbleR")
require("Rraven")
require("devtools")

checkwavs()
wavs <- list.files(pattern="wav$")
lspec(flist = wavs, ovlp = 90, it = "tiff")
res <- autodetec(flist = wavs, bp = c(0.08, 0.3), mindur = 0.1, ssmooth = 600,res = 300, set = TRUE, redo = TRUE)
params <- specan(res, bp = c(0.08, 0.3), threshold = 15)
write.csv(params, "results.csv", row.names = FALSE)

