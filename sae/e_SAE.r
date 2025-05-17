Test(dt,X)
if(exists('Acc') & exists('K') & exists('Kmax')) {
	flag1 <- TRUE 
	swr <- TRUE
	flag<-0
	alert1<-FALSE
	}else{alert1 <- TRUE}
	#----------------------------
if(swr) {save.image(file = fileS); swr <- FALSE}
	