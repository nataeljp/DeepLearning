//+------------------------------------------------------------------+
//|                                                        R_sae.mq4 |
//|                                   Copyright 2014,Vlad Perervenko |
//|                                                 v_minkov@mail.ru |
//+------------------------------------------------------------------+

#property copyright "Copyright 2014,Vlad Perervenko"
#property link      "v_minkov@mail.ru"
#property version   "1.00"
#property strict
#property indicator_chart_window
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

#property indicator_buffers 4
#property indicator_color1 Blue
#property indicator_width1 2
#property indicator_color2 Brown
#property indicator_width2 2
#property indicator_color3 Blue
#property indicator_width3 2
#property indicator_color4 Red
#property indicator_width4 2

#include <mt4Rb7.mqh>

extern int     back = 700;
extern int     ch1  = 75;
extern int     ch2  = 25;
extern bool    send = false;//Send to server ?
extern int     port = 8888; //Server port

double ZZ1[], zz1[], ZZ2[], zz2[], UP[], DN[], sig[];
double op[], hi[], lo[], cl[];
int k=0;
string  fileName, patch;
bool first=true, saveWS=true;
//+------------------------------------------------------------------+
//| Custom indicator initialization function                         |
//+------------------------------------------------------------------+
void init()
  {
   if(!IsDllsAllowed()) 
     {
      MessageBox("You need to turn on \'Allow DLL imports\'");
     }
//---- indicators
   SetIndexBuffer(0,ZZ1);
   SetIndexStyle(0,DRAW_LINE);
   SetIndexBuffer(1,ZZ2);
   SetIndexStyle(1,DRAW_LINE);

   SetIndexBuffer(2,UP);
   SetIndexStyle(2,DRAW_ARROW);
   SetIndexArrow(2,233);
   SetIndexBuffer(3,DN);
   SetIndexStyle(3,DRAW_ARROW);
   SetIndexArrow(3,234);

//----Launch Rterm---------------------------------------------------
   StartR(RPATH);
//----Load parameters for Rterma---------------------------------------
   string terminalDataPath = TerminalInfoString(TERMINAL_DATA_PATH);
   StringReplace(terminalDataPath, "\\", "/");
   patch = StringConcatenate(terminalDataPath, "/MQL4/Files/");
   string tf = GetNameTF(Period());
//Clear the working space and set the initial values
   Rx("rm(list = ls()); first<-TRUE; swr<- FALSE; serv<-FALSE");
//pass the file name and the path of saving the R working space to R
   if(saveWS)
     {
      Rx("swr <- TRUE");
      fileName = createFileName();
      Rs("patch", patch);
      Rs("fR2", fileName);
      Rs("sym", Symbol());
      Rs("tf", tf);
     }
   if(send) Rx("serv <- TRUE");
//------------------ 
   ArrayResize(op, back);
   ArrayResize(hi, back);
   ArrayResize(lo, back);
   ArrayResize(cl, back);
   if(Digits == 5 || Digits == 3) k = 10; else  k = 1;
   double chan1 = ch1 * Point * k;
   double chan2 = ch2 * Point * k;
//Pass constants to R
   Ri("Dig", Digits);
   Ri("port", port);
   Rd("ch1", chan1);
   Rd("ch2", chan2);
 
  }
//+------------------------------------------------------------------+
//| Custom indicator deinitialization function                       |
//+------------------------------------------------------------------+
void deinit()
  {
//----
   Rx("if(serv) close(con)");
   StopR();
//----

  }
//+------------------------------------------------------------------+
//| Custom indicator iteration function                              |
//+------------------------------------------------------------------+
int start()
  {
   int i = 0, j = 0;
   static datetime LastTime = 0;
   static bool get_zz = false, get_sig = false;
//--------Check the work of Rterm--------------------------------------
   if(!RIsRunning(hR))
     {
      Alert("Rterm crashed!");
      return(-1);
     }

//-----------Calculate data-------------------------------------------
   if(LastTime != Time[0] && !RIsBusy(hR))
     { //.. If the bar is new and R is not occupied
      for(i = 1, j = 0; i <= back; i++)
        {             //Populate a new set 
         op[j] = Open[i];
         hi[j] = High[i];
         lo[j] = Low[i];
         cl[j] = Close[i];
         j++;
        }
      //----------Pass data to Rterm-----------------------------------------
      Rv("Open", op);
      Rv("High", hi);
      Rv("Low", lo);
      Rv("Close", cl);

      //-----------Load and exit without waiting for the completion----------
      RExecuteAsync(hR, "source('C:/RData/i_SAE.r')");
      LastTime = Time[0];
      get_zz = true;
      get_sig = true;

     }//.

//---------Obtain the ZZ result--------------------------------------------
   if(get_zz && RIsRunning(hR) && !RIsBusy(hR))
     {//..
      int len = Rgi("length(zz1)");//Calculate the length of the vector
      ArrayResize(zz1, len);
      ArrayResize(zz2, len);
      Rgv("rev(zz1)", zz1);         // Reverse and print 
      Rgv("rev(zz2)", zz2);
      for(i = 0; i < len; i++)
        {
         ZZ1[i+1] = ND(zz1[i]);      //Transfer to the buffer
         ZZ2[i+1] = ND(zz2[i]);
        }
      get_zz = false;
     }//.
//-------Obtain the forecast result----------------------
   if(RIsRunning(hR) && !RIsBusy(hR) && send && get_sig)
     {
      if(Rgb("res <- GetRes()")) 
        {       //If the result is ready
         int len = Rgi("length(sig)"); //Define the length of the vector
         ArrayResize(sig, len);
         Rgv("rev(sig)", sig);         // Reverse and print 
         for(i = 0; i < len-1; i++)
           {        //Transfer to the buffer
            if(sig[i] != sig[i+1])
              {
               if(sig[i] == 1)
                 {
                  UP[i] = Low[i];
                  DN[i] = NULL;
                 }
               if(sig[i] == -1)
                 {
                  UP[i] = NULL;
                  DN[i] = High[i];
                 }
                 } else {
               UP[i] = NULL;
               DN[i] = NULL;
              }
           }
         get_sig = false;
        }
     }

//-------------------------------------------------------
   return(0);
  }
//+------------------------------------------------------------------+
//============= FUNCTIONS ============================================
//----------------------------------------------------------------------
double ND(double A, int d = -1)
  {
   if(d == -1) d = Digits;
   return(NormalizeDouble(A, d));
  }
//----------------------------------------------------------------------
string createFileName() 
  {
 string Name = WindowExpertName() + "_" +
               Symbol() + "_" +
               IntegerToString(Period()) +
               ".RData";
   return(Name);
  }
//----------------------------------------------------------------------------

//+----------------------------------------------------------------------------+
//|  Authro    : Kim Igor V. aka KimIV,  http://www.kimiv.ru                   |
//+----------------------------------------------------------------------------+
//|  Version   : 01.09.2005                                                    |
//|  Description : Returns the timeframe name                                  |
//+----------------------------------------------------------------------------+
//|  Parameters:                                                               |
//|    TimeFrame - timeframe (number of seconds)      (0 - current timeframe)  |
//+----------------------------------------------------------------------------+
string GetNameTF(int TimeFrame=0) 
  {
   if(TimeFrame==0) TimeFrame=Period();
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
void SetLabelText(string nm,string tx,color col,int xd,int yd,
                  int cr=0,int fs=9,string font="pas",int window=0)
  {
   if(ObjectFind(nm)<0)
      ObjectCreate(nm,OBJ_LABEL,window,0,0);
   if(font=="pas") font="Arial";
   if(font=="akt") font="Arial Black";
   ObjectSetText(nm,tx,fs,font);
   ObjectSet(nm,OBJPROP_COLOR,col);
   ObjectSet(nm,OBJPROP_XDISTANCE,xd);
   ObjectSet(nm,OBJPROP_YDISTANCE,yd);
   ObjectSet(nm,OBJPROP_CORNER,cr);

  }
//+------------------------------------------------------------------+
