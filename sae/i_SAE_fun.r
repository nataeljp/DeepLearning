#---------------------------------------------
pr.OHLC <- function(o,  h,  l,  c){
  price <- cbind(Open = rev(o), High = rev(h), Low = rev(l), Close = rev(c))
  Med <- (price[ ,2] + price[ ,3])/2
  CO <- price[ ,4] - price[ ,1]
  price <- cbind(price, Med, CO)
}

#-----------------------------------------------------
In <- function(p = 16){
	adx <- ADX(price, n = p)
	ar <- aroon(price[ ,c('High', 'Low')], n = p)[ ,'oscillator']
	cci <- CCI(price[ ,2:4], n = p)
	chv <- chaikinVolatility(price[ ,2:4], n = p)
	cmo <- CMO(price[ ,'Med'], n = p)
	macd <- MACD(price[ ,'Med'], 12, 26, 9)[ ,'macd']
	osma <- macd - MACD(price[ ,'Med'],12, 26, 9)[ ,'signal']
	rsi <- RSI(price[ ,'Med'], n = p)
	stoh <- stoch(price[ ,2:4], 14, 3, 3)
	smi <- SMI(price[ ,2:4],n = p, nFast = 2, nSlow = 25, nSig = 9)
	vol <- volatility(price[ ,1:4], n = p, calc="yang.zhang", N=96)
	In <- cbind(adx, ar, cci, chv, cmo, macd, osma, rsi, stoh, smi, vol)
	return(In)
}
#--------------------------------------------------
ZZ <- function(ch = 0.0037, pr = price[ ,'Med']){
	zz <- ZigZag(pr, change = ch, percent = F, retrace = F, lastExtreme = T)
	n <- 1:length(zz)
	#Неопределенные значения заменим на последние известные
	for(i in n) { if(is.na(zz[i])) zz[i] = zz[i-1]}
	return(zz)
}
#---------------------------------------------------
Out <- function(zz){
	#Определим скорость изменения ЗигЗага
	dz <- c(diff(zz), NA)
	#Перекодируем скорость в сигналы; 0 - Buy; 1 - Sell
	sig <- ifelse(dz > 0, 0, ifelse(dz < 0, 1, NA))
	return(sig)
}
#----------------------------------
Clearing <- function(x, y){
	dt <- cbind(x, y)
	dt <- na.omit(dt)
	return(dt)  
}
#---------------------------------------
Cs <- function (...)
  as.character(sys.call())[-1]
#-------------------------------------------

GetRes <- function(){
	z <- evalServer(con, flag1)
	if(z){sig <<- evalServer(con, sig)}
	return(z)
}

