subDir <- paste("Test", Sys.Date(), sep = "_");
patch <- paste0(patch, sym)
if(!file.exists(patch)){dir.create(file.path(patch))}
mainDir <- paste(patch, tf, sep = "/")
if(!file.exists(mainDir)){dir.create(file.path(mainDir))}
if(!file.exists(file.path(mainDir, subDir))){
	dir.create(file.path(mainDir, subDir))
}
setwd(file.path(mainDir, subDir));