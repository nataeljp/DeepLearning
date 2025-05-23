//+------------------------------------------------------------------+
//|                                                        e_SAE.mq4 |
//|                                   Copyright 2014,Vlad Perervenko |
//|                                                 v_minkov@mail.ru |
//+------------------------------------------------------------------+
#property copyright "Copyright 2014,Vlad Perervenko"
#property link      "v_minkov@mail.ru"
#property version   "1.00"
#property strict

/**
* This code is released under Gnu General Public License (GPL) V3
* If you need a commercial license then send me an email.
*/

/**
* For this to use you need the following:
*  - install R  (www.r-project.org)
*  - install mt4Rb7.mqh and mt4Rb7.dll
*  - set RPATH below to point to your R installation
*  - (optional) download and run DebugView.exe
*  - (optional) set RDEBUG to 2 to view more debug info in DebugView
*/
//Specify the path to the location of R on your computer
// set this so that it points to your R installation. Do NOT remove the --no-save
#define RPATH "C:/Program Files/R/R-3.1.1/bin/x64/Rterm.exe --no-save"
//--- input parameters
input double Lots          = 0.1;
input double TakeProfit    = 50.0;
input double StopLoss      = 25.0;
input int    magic         = 54321;
input int    cor           = 2;     //Angle of the text printing
input int    port          = 8888;  //Server port
input int    dec           = 1;     // 1-mean, 2 - 60/40
input color  cvet          = Brown;
#include <mt4Rb7.mqh>

int  k=10,sig=0;
string fileName="",op="ERR";
double Acc=0.0,K=0.0,Kmax=0.0,TP=0.0,SL=0.0,TS=0.0;
bool saveWS=true;
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int init()
  {
   if(!IsDllsAllowed())
     {
      MessageBox("You need to turn on \'Allow DLL imports\'");
     }
   if(IsTradeAllowed() == false) Alert("Trading forbidden!");
//--- create timer
   EventSetTimer(5);
//--- Start Rterm -----------------
   StartR(RPATH);
   Rx("rm(list = ls()); first <- TRUE; swr <- FALSE; alert1 <- TRUE");
   if(saveWS)
     {
      string terminalDataPath = TerminalInfoString(TERMINAL_DATA_PATH);
      StringReplace(terminalDataPath, "\\", "/");
      string patch = StringConcatenate(terminalDataPath, "/MQL4/Files/");
      string tf = GetNameTF(Period());
      fileName = createFileName();
      Rx("swr <- TRUE");
      Ri("port", port);
      Rs("patch", patch);
      Rs("fS", fileName);
      Rs("sym", Symbol());
      Rs("tf", tf);
     }
   if(Digits == 5 || Digits == 3) k = 10; else  k = 1;
   TP = TakeProfit * k;
   SL = StopLoss * k;
   Ri("Dig", Digits);
   Ri("dec", dec);
   Rx("source('C:/RData/e_SAE_init.r')");
   RExecuteAsync(hR, "library('deepnet', quietly = T); library('caret', quietly = T)");

//---

   return(INIT_SUCCEEDED);
  }
//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void deinit()
  {
//--- destroy timer
   EventKillTimer();
   ObjectDelete("res");
//---- close Server -------
   Rx("stopSocketServer(port = 8888)");
   StopR();
  }
//+------------------------------------------------------------------+
//| Timer function                                                   |
//+------------------------------------------------------------------+
void OnTimer()
  {
   if(!RIsRunning(hR))
     {
      Alert("Rterm crashed!");
      return;
     }
//----------------------------------------------------------------------
   sig = GetRes();
   if(sig == 1) op = "BUY";
   else if(sig == -1) op = "SELL";
   else if(sig == 0) op = "CLOSE";
   else op = "ERR";
   string text = StringConcatenate("OP = ", op, ";  ", "Acc = ", DoubleToStr(Acc,2), ";  ",
                                 "K = ", DoubleToStr(K,0), ";  ",
                                 "Kmax = ", DoubleToStr(Kmax,0));
   SetLabelText("res", text, cvet, 50, 30, cor, 12);
//----------------------------------------------
   CheckForClose(op, magic);
   CheckForOpen(op, magic);
   return;
  }
//+------------------------------------------------------------------+

//+------------------------------------------------------------------+
//===== Function ====================================================
//----------------------------------------------------------------------
//----------------------------------------------------------------------
int GetRes()
  {
   if(Rgb("alert1"))
     {
      Alert("No calculation results!" + Symbol());
      sig = 0;
      return(0);
     }
   if(Rgb("flag1"))
     {
      sig = Rgi("tail(sig, 1)");
      Acc = ND(Rgd("Acc"), 2);
      K = ND(Rgd("K"), 1);
      Kmax = ND(Rgd("Kmax"), 1);
      return(sig);
     }
   return(sig = 0);
  }
//----------------------------------------------------------------------
double ND(double A,int d=-1)
  {
   if(d == -1) d = Digits;
   return(NormalizeDouble(A, d));
  }
//----------------------------------------------
string createFileName()
  {
   string name = WindowExpertName() + "_" +
                 Symbol() + "_" +
                 IntegerToString(Period()) +
                 ".RData";
   return(name);
  }
//+----------------------------------------------------------------------------+
//|  Authro    : Kim Igor V. aka KimIV,  http://www.kimiv.ru                   |
//+----------------------------------------------------------------------------+
//|  Version   : 01.09.2005                                                    |
//|  Description : Returns the timeframe name                                  |
//+----------------------------------------------------------------------------+
//|  Parameters:                                                               |
//|    TimeFrame - timeframe (number of seconds)      (0 - current timeframe)  |
//+----------------------------------------------------------------------------+
string GetNameTF(int TimeFrame = 0)
  {
   if(TimeFrame == 0) TimeFrame = Period();
   switch(TimeFrame)
     {
      case PERIOD_M1:  return("M1");
      case PERIOD_M5:  return("M5");
      case PERIOD_M15: return("M15");
      case PERIOD_M30: return("M30");
      case PERIOD_H1:  return("H1");
      case PERIOD_H4:  return("H4");
      case PERIOD_D1:  return("Daily");
      case PERIOD_W1:  return("Weekly");
      case PERIOD_MN1: return("Monthly");
      default:         return("Unknown Period");
     }
  }
//+----------------------------------------------------------------------------+
//|  Authro    : Kim Igor V. aka KimIV,  http://www.kimiv.ru                   |
//+----------------------------------------------------------------------------+
//|  Version   : 12.10.2007                                                    |
//|  Description : Placing a text label, object OBJ_LABEL.                   |
//+----------------------------------------------------------------------------+
//|  Parameters:                                                               |
//|    nm - object name                                                        |
//|    tx - text                                                               |
//|    cl - label color                                                        |
//|    xd - X component in pixels                                              |
//|    yd - Y component in pixels                                              |
//|    cr - number of the bound corner (0 - top left )                         |
//|                                     1 - top right                          |
//|                                     2 - bottom left                        |
//|                                     3 - bottom right )                     |
//|    fs - font size                  (9 - by default  )
//     font- default font "pas" "Arial", "akt" -  "Arial Black"                         |
//+----------------------------------------------------------------------------+
void SetLabelText(string nm, string tx, color cl, int xd, int yd,
                  int cr = 0, int fs = 9, string font = "pas", int window = 0)
  {
   if(ObjectFind(nm) < 0)
      ObjectCreate(nm, OBJ_LABEL, window, 0, 0);
   if(font == "pas") font = "Arial";
   if(font == "akt") font = "Arial Black";
   ObjectSetText(nm, tx, fs, font);
   ObjectSet(nm, OBJPROP_COLOR, cl);
   ObjectSet(nm, OBJPROP_XDISTANCE, xd);
   ObjectSet(nm, OBJPROP_YDISTANCE, yd);
   ObjectSet(nm, OBJPROP_CORNER, cr);

  }
//+------------------------------------------------------------------+
//+------------------------------------------------------------------+
//| Calculate open positions                                         |
//+------------------------------------------------------------------+
int CalculateCurrentOrders(int mag)
  {
   int buys = 0, sells = 0;
//----
   for(int i = 0; i < OrdersTotal(); i++)
     {
      if(OrderSelect(i,SELECT_BY_POS,MODE_TRADES) == false) continue;
      if(OrderSymbol() == Symbol() && OrderMagicNumber() == mag)
        {
         if(OrderType() == OP_BUY) buys++;
         if(OrderType() == OP_SELL) sells++;
        }
     }
//----  
   if(buys > 0) return(buys);
   else       return(sells);
  }
//----------------------------------------------------------------------
//+------------------------------------------------------------------+
//| Check for close order conditions                                 |
//+------------------------------------------------------------------+

bool CheckForClose(string op, int mag)
  {

//----
   for(int i = 0; i < OrdersTotal(); i++)
     {
      if(OrderSelect(i,SELECT_BY_POS,MODE_TRADES) == false) continue;
      if(OrderMagicNumber() != mag || OrderSymbol() != Symbol()) continue;
      //---- check order type 
      if((OrderType() == OP_BUY) && (op == "SELL" || op == "CLOSE" || op == "ERR"))
        {
         if(!OrderClose(OrderTicket(), OrderLots(), Bid, 3, White))
           {
            Alert("Failed to close position"+Symbol());
            return(false);
           }
         else return(true);
        }
      if(( OrderType() == OP_SELL) && (op == "BUY" || op == "CLOSE" || op == "ERR"))
        {
         if(!OrderClose(OrderTicket(), OrderLots(), Ask, 3, White))
           {
            Alert("Failed to close position"+Symbol());
            return(false);
           }
         else return(true);
        }
     }
//----
   return(false);
  }
//+------------------------------------------------------------------+
//| Check for open order conditions                                  |
//+------------------------------------------------------------------+
bool CheckForOpen(string op, int mag)
  {
   int ticket = -1;
   int pos = CalculateCurrentOrders(magic);//Are there open positions?
   if(pos > 0) return(true);             //If permissible number is exceeded, exit
//---- sell conditions
   if(op == "SELL")
     {
      RefreshRates();
      double tp = ND((Bid - TP * Point));
      double sl = ND((Bid + SL * Point));
      ticket=OrderSend(Symbol(), OP_SELL, Lots, Bid, 3, sl, tp,
                       "SAE", mag, 0, Red);
      if(ticket > 0)
        {
         if(OrderSelect(ticket, SELECT_BY_TICKET, MODE_TRADES))
           {
            Print("SELL order opened : ", OrderOpenPrice());
            return(true);
           }
           } else {
         Print("Error opening SELL order : ", GetLastError());
         Alert("Failed to open the SELL position " + Symbol());
         return(false);
        }
     }

//---- buy conditions
   if(op == "BUY")
     {
      RefreshRates();
      double tp = ND((Ask + TP * Point));
      double sl = ND((Ask - SL * Point));
      ticket = OrderSend(Symbol(), OP_BUY, Lots, Ask, 3, sl, tp,
                       "SAE", mag, 0, Blue);
      if(ticket > 0)
        {
         if(OrderSelect(ticket, SELECT_BY_TICKET, MODE_TRADES))
           {
            Print("BUY order opened : ", OrderOpenPrice());
            return(true);
           }
           } else {
         Print("Error opening BUY order : ", GetLastError());
         Alert("Failed to open the BUY position " + Symbol());
         return(false);
        }
     }

//----
   return(false);
  }
//+------------------------------------------------------------------+
