	#-----StartServer---------------------------
	library('svSocket', quietly = T) 
	s <- startSocketServer(port = port, server.name = sym)
	#------SetDir-------V
	source('C:/RData/SAE_SetDir.r')
	#---------LoadModel---------------------------
	model <- paste(mainDir, 'SAE.model', sep='/' );
	if(file.exists(model)){load(model)};
	alert <- ifelse(exists('SAE') & exists('prepr'), 0, 1);

#-----------Function-----------------------------	
Test<-function(dt,x){
	  dt.x <- tail(predict(prepr, dt[ ,-ncol(dt)]), 500);
	  dt.y <- tail(dt[ ,ncol(dt)], 500);
	  pr.sae <- nn.predict(SAE, dt.x);
	  pr <- ifelse(pr.sae > mean(pr.sae), 1, 0);
	  Acc <<- unname(confusionMatrix(dt.y, pr)$overall[1]);
	  new.data <- predict(prepr, tail(x, 500));
	  pr.sae <- nn.predict(SAE, new.data);
	  if(dec == 1) {sig <<- ifelse(pr.sae > mean(pr.sae), -1, 1)}
	  if(dec == 2) {sig <<- ifelse(pr.sae > 0.6, -1, ifelse(pr.sae<0.4, 1, 0))}
	  bal<-cumsum(tail(CO, 500) * sig)
	  K <<- tail(bal/length(bal) * 10^Dig, 1);
	  Kmax <<- max(bal)/which.max(bal) * 10^Dig;
	  if(exists('Acc') & exists('K') & exists('Kmax')) {
					return(TRUE)
				  }else {return(FALSE)}
	}
#------------SaveImage-------------------------------
if(swr) {save.image(file = fS); swr <- FALSE}	
	
