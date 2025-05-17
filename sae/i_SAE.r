
if(first )
{
	library("TTR", quietly = T);
	#-------connection-----------------------
	if(serv){
	  library("svSocket", quietly=T);
	  con <- socketConnection(host = 'localhost', port = port, blocking = FALSE);
	}
	source('C:/RData/i_SAE_fun.r');
   #------SetDir-------V
	source('C:/RData/SAE_SetDir.r')
	#--------------------
	first = F;
}
#1---- -------------------
	price <- pr.OHLC(Open, High, Low, Close);
	rm(list = c("Open","High","Low","Close"));
#1a----
	X <- In();
	zz1 <- ZZ(ch1);
	zz2 <- ZZ(ch2);
#1b--------------------------
	Y <- Out(zz2); 
#1c------
	dt <- Clearing(X, Y);
	rm(Y);
	
#2-------------------------------------------------------------------------
	if(serv == TRUE	){
		evalServer(con, dt, dt); 
		evalServer(con, X, X);
		evalServer(con, CO, price[ , 'CO'])
		evalServer(con, "source('C:/RData/e_SAE.r')")
		#rm(list=Cs(price, dt, X, Y))
		#evalServer(con, 'rm(list=Cs(CO, dt, X))');
	}
#3-------------------------------------------
	
#4-------------------------------------------------------------------		
	if(swr) {save.image(basename(fR2))}
