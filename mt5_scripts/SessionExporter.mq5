//+------------------------------------------------------------------+
//| SessionExporter.mq5 - Export trading sessions to JSON daily      |
//| Runs as EA on any chart, writes session times for all symbols    |
//| Output: C:\tradebots\data\sessions\{broker}_sessions.json       |
//+------------------------------------------------------------------+
#property copyright "Sovereign Trading System"
#property version   "1.0"
#property strict

input string OutputDir = "C:\\tradebots\\data\\sessions";  // Output directory
input string BrokerTag = "bf";                              // Broker tag (bf or ftmo)
input int    UpdateHour = 0;                                // Hour to update (0 = midnight)
input int    UpdateMinute = 5;                              // Minute to update

datetime lastExportDate = 0;

//+------------------------------------------------------------------+
int OnInit()
{
    Print("[SessionExporter] Starting for broker: ", BrokerTag);
    ExportSessions();  // Export immediately on startup
    EventSetTimer(60);  // Check every 60 seconds
    return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
    EventKillTimer();
}

//+------------------------------------------------------------------+
void OnTimer()
{
    MqlDateTime dt;
    TimeCurrent(dt);

    // Export at configured time, once per day
    if (dt.hour == UpdateHour && dt.min == UpdateMinute)
    {
        datetime today = StringToTime(TimeToString(TimeCurrent(), TIME_DATE));
        if (today != lastExportDate)
        {
            ExportSessions();
            lastExportDate = today;
        }
    }
}

//+------------------------------------------------------------------+
void OnTick()
{
    // Also check on tick in case timer missed
    MqlDateTime dt;
    TimeCurrent(dt);

    datetime today = StringToTime(TimeToString(TimeCurrent(), TIME_DATE));
    if (today != lastExportDate && dt.hour >= UpdateHour)
    {
        ExportSessions();
        lastExportDate = today;
    }
}

//+------------------------------------------------------------------+
void ExportSessions()
{
    string filename = OutputDir + "\\" + BrokerTag + "_sessions.json";

    // Ensure directory exists
    // MQL5 can only write to Files/ sandbox, so we use a different approach
    // Write to common data folder
    string commonFile = BrokerTag + "_sessions.json";

    int handle = FileOpen(commonFile, FILE_WRITE | FILE_TXT | FILE_COMMON | FILE_ANSI);
    if (handle == INVALID_HANDLE)
    {
        Print("[SessionExporter] ERROR: Cannot open file: ", commonFile);
        return;
    }

    FileWriteString(handle, "{\n");
    FileWriteString(handle, "  \"broker\": \"" + BrokerTag + "\",\n");
    FileWriteString(handle, "  \"exported\": \"" + TimeToString(TimeCurrent(), TIME_DATE | TIME_MINUTES) + "\",\n");
    FileWriteString(handle, "  \"server_time\": \"" + TimeToString(TimeCurrent()) + "\",\n");
    FileWriteString(handle, "  \"gmt_offset\": " + IntegerToString(TimeGMTOffset()) + ",\n");
    FileWriteString(handle, "  \"symbols\": {\n");

    int totalSymbols = SymbolsTotal(false);  // All symbols, not just Market Watch
    bool firstSymbol = true;
    int exported = 0;

    for (int i = 0; i < totalSymbols; i++)
    {
        string sym = SymbolName(i, false);
        if (sym == "") continue;

        // Check if symbol has sessions
        bool hasSessions = false;
        for (int day = 0; day < 7; day++)
        {
            datetime from, to;
            if (SymbolInfoSessionTrade(sym, (ENUM_DAY_OF_WEEK)day, 0, from, to))
            {
                hasSessions = true;
                break;
            }
        }
        if (!hasSessions) continue;

        if (!firstSymbol)
            FileWriteString(handle, ",\n");
        firstSymbol = false;

        FileWriteString(handle, "    \"" + sym + "\": {\n");
        FileWriteString(handle, "      \"trade_mode\": " + IntegerToString(SymbolInfoInteger(sym, SYMBOL_TRADE_MODE)) + ",\n");

        // Days: 0=Sunday, 1=Monday, ..., 6=Saturday
        string dayNames[] = {"sunday", "monday", "tuesday", "wednesday", "thursday", "friday", "saturday"};

        bool firstDay = true;
        FileWriteString(handle, "      \"sessions\": {\n");

        for (int day = 0; day < 7; day++)
        {
            string daySessions = "";
            int sessionIdx = 0;
            datetime from, to;
            bool firstSession = true;

            while (SymbolInfoSessionTrade(sym, (ENUM_DAY_OF_WEEK)day, sessionIdx, from, to))
            {
                // Convert datetime to HH:MM format
                // from/to are seconds since midnight
                int fromMin = (int)(from % 86400) / 60;
                int toMin = (int)(to % 86400) / 60;

                string fromStr = StringFormat("%02d:%02d", fromMin / 60, fromMin % 60);
                string toStr = StringFormat("%02d:%02d", toMin / 60, toMin % 60);

                if (!firstSession) daySessions += ", ";
                daySessions += "[\"" + fromStr + "\", \"" + toStr + "\"]";
                firstSession = false;
                sessionIdx++;
            }

            if (daySessions != "")
            {
                if (!firstDay) FileWriteString(handle, ",\n");
                firstDay = false;
                FileWriteString(handle, "        \"" + dayNames[day] + "\": [" + daySessions + "]");
            }
        }

        FileWriteString(handle, "\n      }\n");
        FileWriteString(handle, "    }");
        exported++;
    }

    FileWriteString(handle, "\n  }\n");
    FileWriteString(handle, "}\n");

    FileClose(handle);

    // Also copy to tradebots data dir using shell
    string src = TerminalInfoString(TERMINAL_COMMONDATA_PATH) + "\\Files\\" + commonFile;
    string dst = OutputDir + "\\" + commonFile;

    Print("[SessionExporter] Exported ", exported, " symbols to ", commonFile);
    Print("[SessionExporter] Common path: ", src);

    // Write a copy script that Python can run
    string copyScript = BrokerTag + "_copy_sessions.bat";
    int batHandle = FileOpen(copyScript, FILE_WRITE | FILE_TXT | FILE_COMMON | FILE_ANSI);
    if (batHandle != INVALID_HANDLE)
    {
        FileWriteString(batHandle, "@echo off\n");
        FileWriteString(batHandle, "if not exist \"" + OutputDir + "\" mkdir \"" + OutputDir + "\"\n");
        FileWriteString(batHandle, "copy /Y \"" + src + "\" \"" + OutputDir + "\\" + commonFile + "\"\n");
        FileClose(batHandle);
    }
}
//+------------------------------------------------------------------+
