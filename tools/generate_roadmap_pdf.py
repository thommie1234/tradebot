"""Generate Sovereign Bot Roadmap PDF — 100 improvement ideas."""
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.colors import HexColor
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, HRFlowable,
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER
from datetime import datetime

OUTPUT = "/home/tradebot/tradebots/sovereign_roadmap_100.pdf"

# ── Colour palette ────────────────────────────────────────────────
DARK   = HexColor("#1a1a2e")
ACCENT = HexColor("#e94560")
BLUE   = HexColor("#0f3460")
LIGHT  = HexColor("#f5f5f5")
WHITE  = HexColor("#ffffff")
GREY   = HexColor("#666666")
GREEN  = HexColor("#27ae60")
ORANGE = HexColor("#f39c12")
RED    = HexColor("#e74c3c")

# ── Styles ────────────────────────────────────────────────────────
styles = getSampleStyleSheet()

title_style = ParagraphStyle(
    "Title", parent=styles["Title"],
    fontSize=26, leading=32, textColor=DARK,
    spaceAfter=4*mm, alignment=TA_CENTER,
)
subtitle_style = ParagraphStyle(
    "Subtitle", parent=styles["Normal"],
    fontSize=11, leading=14, textColor=GREY,
    spaceAfter=8*mm, alignment=TA_CENTER,
)
cat_style = ParagraphStyle(
    "Category", parent=styles["Heading1"],
    fontSize=16, leading=20, textColor=ACCENT,
    spaceBefore=8*mm, spaceAfter=3*mm,
    borderWidth=0, borderPadding=0,
)
item_title_style = ParagraphStyle(
    "ItemTitle", parent=styles["Normal"],
    fontSize=10, leading=13, textColor=DARK,
    fontName="Helvetica-Bold",
)
item_desc_style = ParagraphStyle(
    "ItemDesc", parent=styles["Normal"],
    fontSize=9, leading=12, textColor=GREY,
)
priority_high = ParagraphStyle("PH", parent=styles["Normal"], fontSize=8, textColor=RED, fontName="Helvetica-Bold", alignment=TA_CENTER)
priority_med  = ParagraphStyle("PM", parent=styles["Normal"], fontSize=8, textColor=ORANGE, fontName="Helvetica-Bold", alignment=TA_CENTER)
priority_low  = ParagraphStyle("PL", parent=styles["Normal"], fontSize=8, textColor=GREEN, fontName="Helvetica-Bold", alignment=TA_CENTER)

# ── The 100 ideas ────────────────────────────────────────────────
# (category, title, description, priority, complexity)
# priority: HIGH / MED / LOW
# complexity: 1-5 (1=easy, 5=hard)

IDEAS = [
    # ─── 1. ML & MODELLING ──────────────────────────────────────
    ("ML & Modelling", "LightGBM ensemble naast XGBoost",
     "Train een LightGBM model op dezelfde features en combineer de probabiliteiten. Twee onafhankelijke boosting-implementaties vangen elkaars zwakke plekken op.", "HIGH", 2),

    ("ML & Modelling", "CatBoost als derde ensemble-lid",
     "CatBoost heeft native ordered target encoding en is robuust tegen overfitting bij weinig data. Voeg toe als derde stem in het ensemble.", "MED", 2),

    ("ML & Modelling", "Stacked generalization (meta-learner)",
     "Train een logistic regression meta-model op de out-of-fold predictions van XGBoost, LightGBM, CatBoost en TabNet. Leert optimale weging automatisch.", "HIGH", 3),

    ("ML & Modelling", "Temporal Fusion Transformer (TFT)",
     "Google's TFT architectuur combineert LSTM met attention en static covariates. Geschikt voor tijdreeksen met bekende toekomstige inputs (bijv. sessietijden).", "LOW", 5),

    ("ML & Modelling", "Online learning / incremental updates",
     "Update het model incrementeel met elke nieuwe bar in plaats van periodieke volledige retrain. River of Vowpal Wabbit kunnen streaming gradient updates doen.", "MED", 4),

    ("ML & Modelling", "Conformal prediction intervals",
     "Voeg conformal prediction toe voor calibrated confidence intervals. In plaats van een punt-schatting krijg je een betrouwbaarheidsinterval — trade alleen als het interval smal genoeg is.", "MED", 3),

    ("ML & Modelling", "Feature importance drift monitoring",
     "Track SHAP values per feature over tijd. Als de top-5 features verschuiven, is het model waarschijnlijk aan het degraderen — trigger automatische retrain.", "HIGH", 2),

    ("ML & Modelling", "Adversarial validation",
     "Train een classifier om train vs. recent data te onderscheiden. Als AUC > 0.6 is er een distributieshift — het model is onbetrouwbaar voor live trading.", "MED", 2),

    ("ML & Modelling", "Target engineering: risk-adjusted returns",
     "Vervang binaire labels door Sharpe-gebaseerde labels: target = (return / vol) > threshold. Dit leert het model risico-gecorrigeerde signalen te genereren.", "HIGH", 3),

    ("ML & Modelling", "Multi-timeframe features (M15 + H4)",
     "Voeg features toe van M15 (korte-termijn momentum) en H4 (trend context). Multi-timeframe alignment is een sterke predictor.", "HIGH", 3),

    ("ML & Modelling", "Purged group time-series CV",
     "Verbeter cross-validatie met purging EN embargoing op groepsniveau (per dag/week). Voorkomt subtiele lookahead bias bij overlappende labels.", "MED", 2),

    ("ML & Modelling", "Bayesian hyperparameter optimization",
     "Vervang Optuna's TPE door Gaussian Process-based BO (bijv. scikit-optimize) voor beter sample-efficiënte hyperparameter search.", "LOW", 2),

    ("ML & Modelling", "Quantile regression voor TP/SL",
     "Train een quantile regression model (XGBoost quantile objective) dat de 10e en 90e percentiel van returns voorspelt. Gebruik voor dynamische TP/SL.", "HIGH", 3),

    ("ML & Modelling", "Survival analysis voor trade duration",
     "Train een Cox proportional hazards model om te voorspellen hoe lang een trade open blijft. Sluit trades die langer open staan dan verwacht.", "LOW", 4),

    ("ML & Modelling", "Autoencoders voor anomaly detection",
     "Train een autoencoder op 'normale' marktcondities. Hoge reconstructie-error = abnormale markt → pauzeer trading automatisch.", "MED", 3),

    # ─── 2. FEATURE ENGINEERING ────────────────────────────────
    ("Feature Engineering", "Order flow imbalance (OFI)",
     "Bereken de cumulatieve delta tussen bid/ask volume changes. OFI is een krachtige korte-termijn predictor van prijsbewegingen.", "HIGH", 3),

    ("Feature Engineering", "VWAP distance",
     "Afstand van prijs tot Volume Weighted Average Price. Institutionele traders gebruiken VWAP als benchmark — afwijkingen creëren mean-reversion signalen.", "MED", 1),

    ("Feature Engineering", "Microstructure noise ratio",
     "Meet de verhouding tussen tick-by-tick noise en werkelijke prijsbeweging. Hoge noise ratio = vermijd trades (slechte fills).", "MED", 3),

    ("Feature Engineering", "Cross-asset correlatie features",
     "Voeg rolling correlaties toe tussen gerelateerde assets (bijv. BTC-ETH, EUR-GBP). Decorrelatie = regime switch signaal.", "HIGH", 2),

    ("Feature Engineering", "Funding rate (crypto)",
     "Integreer perpetual futures funding rates als feature. Extreme funding = overextended posities = mean-reversion kans.", "MED", 2),

    ("Feature Engineering", "Implied volatility (options data)",
     "Als je opties-data kunt halen, is IV rank/percentile een sterke predictor voor verwachte bewegingen en richting.", "LOW", 4),

    ("Feature Engineering", "Session volume profile",
     "Bereken de volume-at-price distributie per sessie (Azië/Europa/VS). POC (Point of Control) shifts geven richting aan.", "MED", 3),

    ("Feature Engineering", "Economic calendar features",
     "Voeg binary features toe voor naderende high-impact events (NFP, CPI, FOMC). Verlaag confidence automatisch rond events.", "HIGH", 2),

    ("Feature Engineering", "Realized volatility vs implied spread",
     "Verschil tussen realized volatility en bid-ask spread. Als RV >> spread is de markt efficiënt geprijsd; als spread >> RV betaal je te veel.", "LOW", 2),

    ("Feature Engineering", "Fractal dimension (Higuchi/Katz)",
     "Meet de fractal dimensie van de prijsreeks. D ≈ 1.0 = trending, D ≈ 1.5 = random walk, D ≈ 2.0 = mean-reverting.", "MED", 2),

    # ─── 3. RISK MANAGEMENT ────────────────────────────────────
    ("Risk Management", "Dynamic Kelly fraction",
     "Bereken de Kelly fraction per trade op basis van het model's calibrated probability en de verwachte payoff. Gebruik half-Kelly voor veiligheid.", "HIGH", 2),

    ("Risk Management", "Conditional Value at Risk (CVaR)",
     "Vervang VaR door CVaR (Expected Shortfall) voor risk budgeting. CVaR pakt tail risks beter op dan gewone VaR.", "MED", 3),

    ("Risk Management", "Intraday drawdown circuit breaker",
     "Stop automatisch alle trading als de intraday drawdown X% bereikt, zelfs als FTMO limiet nog niet geraakt is. Voorkomt tilt-trading.", "HIGH", 1),

    ("Risk Management", "Correlation-based position limits",
     "Bereken realtime correlaties tussen open posities. Als de portfolio-correlatie boven 0.7 komt, weiger nieuwe trades in dezelfde richting.", "HIGH", 2),

    ("Risk Management", "Volatility regime position scaling",
     "Schaal positiegrootte inversief met marktvol. In hoge-vol regimes (VIX > 25) halveer max risk; in lage-vol verdubbel.", "MED", 2),

    ("Risk Management", "Time-based stop loss",
     "Sluit trades die na X uren niet in profit zijn. Dead capital in de markt is opportunity cost — forceer rotatie.", "MED", 1),

    ("Risk Management", "Trailing stop op basis van ATR",
     "Implementeer een ATR-based trailing stop die meebeweegt met de markt. Beschermt profits zonder te vroeg te sluiten.", "HIGH", 2),

    ("Risk Management", "Weekend gap protection",
     "Sluit alle posities vrijdagavond of verklein lot sizes naar 25%. Weekend gaps kunnen door SL heen schieten.", "HIGH", 1),

    ("Risk Management", "Anti-martingale position sizing",
     "Verhoog lot size na winnende trades, verlaag na verliezende trades. Rit de winners, bescherm het kapitaal bij drawdowns.", "MED", 2),

    ("Risk Management", "Max sector exposure limits",
     "Beperk totale exposure per sector (crypto, forex, commodities) tot bijv. 2% van account. Voorkomt gecorreleerde blow-ups.", "HIGH", 1),

    ("Risk Management", "Stress testing module",
     "Simuleer historische crash-scenario's (COVID maart 2020, crypto crash mei 2021) op je huidige portfolio. Rapporteer verwacht verlies.", "MED", 3),

    # ─── 4. EXECUTION & ORDER MANAGEMENT ──────────────────────
    ("Execution", "Smart order routing met spread check",
     "Controleer de spread op meerdere timeframes voordat je een order plaatst. Als de huidige spread > 2x gemiddelde spread, wacht.", "HIGH", 1),

    ("Execution", "Partial fill handling",
     "Detecteer partial fills en beheer de resterende order. Nu wordt een partial fill mogelijk genegeerd.", "MED", 2),

    ("Execution", "Iceberg orders voor grote posities",
     "Split grote orders in kleinere chunks met random timing. Vermindert market impact bij illiquide symbolen.", "LOW", 3),

    ("Execution", "Slippage tracking en analyse",
     "Log het verschil tussen verwachte en werkelijke fill prijs per trade. Gebruik dit om execution quality te meten en symbolen met hoge slippage te vermijden.", "HIGH", 1),

    ("Execution", "Limit order entry in plaats van market",
     "Plaats limit orders net boven/onder de huidige prijs in plaats van market orders. Bespaart spread-kosten bij niet-urgente entries.", "MED", 2),

    ("Execution", "Re-entry logica na SL",
     "Als een trade SL raakt maar het signaal nog steeds geldig is op de volgende bar, overweeg re-entry met kleinere positie.", "MED", 2),

    ("Execution", "Breakeven stop management",
     "Verplaats SL naar breakeven zodra de trade 1R in profit is. Elimineert risico terwijl de trade nog kan doorlopen naar TP.", "HIGH", 1),

    ("Execution", "Multi-leg entry (scale-in)",
     "Verdeel de entry over 2-3 prijsniveaus. Eerste entry op signaal, tweede bij pullback. Verbetert gemiddelde entry prijs.", "MED", 3),

    ("Execution", "Session-aware execution timing",
     "Voer crypto trades uit bij opening Azië sessie (hoge liquiditeit), forex bij London open. Vermijd illiquide uren.", "MED", 2),

    # ─── 5. DATA & INFRASTRUCTURE ─────────────────────────────
    ("Infrastructure", "Redis caching voor features",
     "Cache berekende features in Redis met TTL. Voorkomt herberekening bij meerdere scans en versnelt inference.", "MED", 2),

    ("Infrastructure", "PostgreSQL voor audit trail",
     "Migreer van SQLite naar PostgreSQL. Betere concurrency, geen file locking issues, en je kunt queries draaien terwijl de bot loopt.", "MED", 3),

    ("Infrastructure", "Grafana dashboard",
     "Zet een Grafana dashboard op met realtime metrics: equity curve, drawdown, win rate, open posities, model confidence distributies.", "HIGH", 2),

    ("Infrastructure", "Prometheus metrics export",
     "Exporteer bot metrics (trades/uur, latency, FD count, CPU/GPU temp) naar Prometheus. Stel alerts in voor anomalieën.", "MED", 2),

    ("Infrastructure", "Docker containerization",
     "Containerize de bot + MT5 bridge + sentiment engine in Docker. Maakt deployment reproduceerbaar en rollback makkelijk.", "MED", 3),

    ("Infrastructure", "Automated backup naar S3/Backblaze",
     "Dagelijkse backup van modellen, configs, audit DB en trade logs naar cloud storage. Beschermt tegen disk failure.", "HIGH", 1),

    ("Infrastructure", "Config hot-reload zonder restart",
     "Implementeer SIGHUP handler die config herlaadt zonder de bot te herstarten. Nuttig voor threshold-aanpassingen mid-sessie.", "MED", 2),

    ("Infrastructure", "Tick data pipeline naar Parquet/DuckDB",
     "Stream tick data continu naar gecomprimeerde Parquet files met DuckDB. Maakt historische analyse en backtesting 10x sneller.", "MED", 2),

    # ─── 6. BACKTESTING & RESEARCH ────────────────────────────
    ("Backtesting", "Event-driven backtester",
     "Bouw een event-driven backtester die exact de live order flow simuleert: fills, partial fills, slippage, commissie. Realistischer dan vectorized.", "HIGH", 4),

    ("Backtesting", "Walk-forward Monte Carlo",
     "Combineer walk-forward validatie met Monte Carlo simulatie. Genereer 1000 equity curves door trade-volgorde te shuffelen — meet robuustheid.", "MED", 3),

    ("Backtesting", "Transaction cost modelling",
     "Model realistische transactiekosten: spread (variabel per sessie), commissie, swap, en slippage. Veel strategieën overleven dit niet.", "HIGH", 2),

    ("Backtesting", "Regime-conditional backtesting",
     "Backtest de strategie apart per marktregime (trending/ranging/volatile). Identificeer in welke regimes het model het best presteert.", "MED", 2),

    ("Backtesting", "Paper trading mode",
     "Voeg een paper trading mode toe die live data gebruikt maar orders simuleert. Perfect om nieuwe features te valideren zonder risico.", "HIGH", 2),

    ("Backtesting", "Performance attribution",
     "Splits P&L uit per feature-groep: hoeveel komt van momentum, hoeveel van mean-reversion, hoeveel van sentiment? Identificeert alpha-bronnen.", "MED", 3),

    ("Backtesting", "Backtest van sentiment impact",
     "Backtest de strategie met en zonder sentiment-adjustment. Meet of sentiment daadwerkelijk alpha toevoegt of alleen noise introduceert.", "HIGH", 2),


    # ─── 7. MONITORING & ALERTING ─────────────────────────────
    ("Monitoring", "Telegram bot integratie",
     "Stuur trade alerts, dagelijkse samenvattingen en error notificaties naar Telegram. Sneller dan Discord voor mobiel.", "MED", 1),

    ("Monitoring", "Email digest (dagelijks/wekelijks)",
     "Automatische email met equity curve, top winners/losers, model confidence distributie en risk metrics.", "LOW", 2),

    ("Monitoring", "Mobile app dashboard",
     "Simpele web-app (Flask/FastAPI) met realtime portfolio view, accessible via telefoon. Equity, open trades, P&L.", "MED", 3),

    ("Monitoring", "Anomaly detection op trade patterns",
     "Detecteer abnormale patronen: te veel trades in korte tijd, ongewone lot sizes, trades buiten sessietijden. Alert op mogelijke bugs.", "MED", 2),

    ("Monitoring", "Model staleness alert",
     "Alert als een model langer dan X dagen niet hertraind is of als de gemiddelde confidence daalt onder een threshold.", "HIGH", 1),

    ("Monitoring", "Execution quality scorecard",
     "Wekelijkse scorecard: gemiddelde slippage, fill rate, spread betaald vs. gemiddeld, en timing accuracy.", "MED", 2),

    ("Monitoring", "Risk dashboard met heatmap",
     "Visuele heatmap van correlaties tussen open posities, sector exposure, en drawdown proximity tot FTMO limieten.", "MED", 2),

    ("Monitoring", "Heartbeat naar externe uptime monitor",
     "Push heartbeat naar UptimeRobot of Healthchecks.io. Als de heartbeat stopt, krijg je SMS/call.", "HIGH", 1),

    # ─── 8. SENTIMENT & ALTERNATIVE DATA ──────────────────────
    ("Alternative Data", "Twitter/X sentiment scraping",
     "Scrape crypto Twitter voor real-time sentiment. Gebruik FinBERT of een fine-tuned LLM voor classificatie.", "MED", 3),

    ("Alternative Data", "On-chain data (crypto)",
     "Integreer on-chain metrics: exchange inflows/outflows, whale movements, funding rates. Glassnode of CryptoQuant API.", "MED", 3),

    ("Alternative Data", "COT rapport features (futures)",
     "Commitment of Traders data toont positioning van commercials vs. speculanten. Extreme positionering = reversal signaal.", "MED", 2),

    ("Alternative Data", "Google Trends als feature",
     "Google Trends search volume voor crypto/forex termen. Spikes in zoekvolume correleren met volatiliteit en reversals.", "LOW", 1),

    ("Alternative Data", "Fear & Greed Index integratie",
     "Crypto Fear & Greed Index als daily feature. Extreme fear = koopkans, extreme greed = verkoopsignaal (contrarian).", "MED", 1),

    ("Alternative Data", "Earnings calendar (indices)",
     "Voor index-trades (JP225): integreer earnings calendar. Grote earnings (Toyota, Sony) bewegen de Nikkei.", "LOW", 2),

    ("Alternative Data", "Central bank speech NLP",
     "Analyseer ECB/Fed speeches met NLP voor hawkish/dovish sentiment. Directe impact op forex pairs.", "LOW", 4),

    # ─── 9. PORTFOLIO & STRATEGY ──────────────────────────────
    ("Portfolio & Strategy", "Mean-reversion strategie als complement",
     "Voeg een aparte mean-reversion strategie toe naast de momentum-based ML. Twee oncorreleerde strategieën smoothen de equity curve.", "HIGH", 3),

    ("Portfolio & Strategy", "Pairs trading module",
     "Identificeer co-integreerde paren (bijv. BTC-ETH) en trade de spread. Marktneutraal en complementair aan directional trades.", "MED", 4),

    ("Portfolio & Strategy", "Dynamic strategy allocation",
     "Alloceer kapitaal dynamisch tussen strategieën op basis van recent performance. Bayesian optimization van de allocatie-weights.", "MED", 4),

    ("Portfolio & Strategy", "Multi-timeframe strategie",
     "H4 voor trend-richting, H1 voor entry timing, M15 voor exacte entry. Layered timeframe approach verbetert hit rate.", "HIGH", 3),

    ("Portfolio & Strategy", "Sector rotation model",
     "Track welke sectoren outperformen en roteer kapitaal. Crypto in risk-on, forex majors in risk-off.", "MED", 3),

    ("Portfolio & Strategy", "Volatility targeting",
     "Schaal de totale portfolio exposure zodat de gerealiseerde volatiliteit constant is (bijv. 10% annualized). Voorkomt drawdown-spikes.", "HIGH", 2),

    ("Portfolio & Strategy", "Risk parity allocatie",
     "Verdeel risk budget gelijk over alle posities (inversief met volatiliteit). Voorkomt dat één volatile positie het portfolio domineert.", "MED", 2),


    # ─── 10. AUTOMATION & OPS ─────────────────────────────────
    ("Automation", "Sunday ritual automatisering",
     "Volledig geautomatiseerde weekly pipeline: data download → retrain → backtest → promote → deploy. Geen handmatige interventie.", "HIGH", 2),

    ("Automation", "Auto-scaling retrain frequency",
     "Train vaker als model performance daalt, minder vaak als het stabiel is. Bespaar compute bij stabiele modellen.", "MED", 2),

    ("Automation", "Canary deployment voor nieuwe modellen",
     "Deploy nieuwe modellen eerst op 1 symbool met klein kapitaal. Als performance OK is na 20 trades, rol uit naar alle symbolen.", "MED", 3),

    ("Automation", "Git-based model versioning",
     "Gebruik DVC (Data Version Control) of MLflow voor model versioning met git. Maakt rollback naar elk eerder model trivial.", "MED", 2),

    ("Automation", "Automated hyperparameter reoptimization",
     "Draai Optuna automatisch wanneer model decay gedetecteerd wordt. Sluit de retrain → optimize → deploy loop.", "MED", 3),

    ("Automation", "Self-healing: auto-restart bij crash",
     "Verbeter systemd service met watchdog pattern: als de bot langer dan 5 min geen heartbeat stuurt, kill en restart.", "HIGH", 1),

    ("Automation", "Log rotation en cleanup",
     "Automatische log rotation (logrotate config) en cleanup van oude model versions (keep laatste 5 per symbool).", "MED", 1),


    # ─── 11. ADVANCED ML TECHNIQUES ───────────────────────────
    ("Advanced ML", "Reinforcement learning voor order execution",
     "Train een RL agent (PPO/SAC) voor optimale order execution: wanneer limit vs. market, hoe groot, wanneer annuleren.", "LOW", 5),

    ("Advanced ML", "Graph Neural Networks voor asset relaties",
     "Model de relaties tussen assets als een graph. GNNs kunnen lead-lag relaties en contagion patterns leren.", "LOW", 5),

    ("Advanced ML", "Transfer learning tussen symbolen",
     "Pre-train een basis-model op alle symbolen, fine-tune per symbool. Symbolen met weinig data profiteren van gedeelde kennis.", "MED", 3),

    ("Advanced ML", "Calibrated probability outputs",
     "Pas Platt scaling of isotonic regression toe op de model outputs. Zorgt dat 'proba=0.7' ook echt 70% kans betekent.", "HIGH", 2),

    ("Advanced ML", "Mixture of Experts (MoE)",
     "Train aparte expert-modellen per regime (trending/ranging/volatile). Een gating network selecteert het juiste expert-model.", "MED", 4),

    ("Advanced ML", "Noise-aware training",
     "Gebruik label smoothing of confident learning (cleanlab) om met noisy labels om te gaan. Financiële labels zijn inherent noisy.", "MED", 2),


    # ─── 12. COMPLIANCE & REPORTING ───────────────────────────
    ("Compliance", "FTMO drawdown predictor",
     "Voorspel de kans dat je de dagelijkse of totale FTMO drawdown limiet raakt op basis van huidige open posities en volatiliteit.", "HIGH", 2),

    ("Compliance", "Automated FTMO rule checker",
     "Continue check op alle FTMO regels: max drawdown, min trading days, max lot size, weekend exposure. Alert voor je in overtreding bent.", "HIGH", 1),

    ("Compliance", "Trade journal PDF generator",
     "Automatische wekelijkse trade journal als PDF: elke trade met screenshot, entry/exit rationale, en lesson learned.", "MED", 2),

    ("Compliance", "Tax reporting export",
     "Exporteer alle trades in het juiste formaat voor belastingaangifte. Per land verschillend (NL: box 3, vermogensrendementsheffing).", "LOW", 2),

    ("Compliance", "Profit target tracker",
     "Track voortgang naar FTMO profit target (10%). Visuele progress bar en ETA op basis van huidige win rate.", "MED", 1),

]

assert len(IDEAS) == 100, f"Expected 100 ideas, got {len(IDEAS)}"

# ── Build PDF ─────────────────────────────────────────────────────
doc = SimpleDocTemplate(
    OUTPUT, pagesize=A4,
    topMargin=20*mm, bottomMargin=15*mm,
    leftMargin=15*mm, rightMargin=15*mm,
    title="Sovereign Bot — 100-Point Roadmap",
    author="Claude Code",
)

story = []

# Title page
story.append(Spacer(1, 30*mm))
story.append(Paragraph("SOVEREIGN BOT", title_style))
story.append(Paragraph("100-Point Improvement Roadmap", ParagraphStyle(
    "Sub", parent=styles["Heading2"], fontSize=18, textColor=BLUE,
    alignment=TA_CENTER, spaceAfter=6*mm,
)))
story.append(Paragraph(
    f"Gegenereerd op {datetime.now().strftime('%d %B %Y')} &bull; "
    f"Huidige stack: XGBoost + TabNet + LinUCB RL + 39 features + Sentiment",
    subtitle_style,
))
story.append(Spacer(1, 10*mm))
story.append(HRFlowable(width="80%", thickness=1, color=ACCENT))
story.append(Spacer(1, 6*mm))

# Legend
legend_data = [
    [Paragraph("<b>Prioriteit</b>", item_title_style),
     Paragraph("<b>Complexiteit</b>", item_title_style),
     Paragraph("<b>Beschrijving</b>", item_title_style)],
    [Paragraph("HIGH", priority_high),
     Paragraph("1-2", item_desc_style),
     Paragraph("Directe impact op P&L of stabiliteit", item_desc_style)],
    [Paragraph("MED", priority_med),
     Paragraph("3", item_desc_style),
     Paragraph("Verbetert edge maar niet urgent", item_desc_style)],
    [Paragraph("LOW", priority_low),
     Paragraph("4-5", item_desc_style),
     Paragraph("Nice-to-have of research project", item_desc_style)],
]
legend_table = Table(legend_data, colWidths=[30*mm, 30*mm, 100*mm])
legend_table.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (-1, 0), LIGHT),
    ('GRID', (0, 0), (-1, -1), 0.5, HexColor("#dddddd")),
    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ('TOPPADDING', (0, 0), (-1, -1), 3),
    ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
]))
story.append(legend_table)

# Summary stats
story.append(Spacer(1, 6*mm))
high_count = sum(1 for _, _, _, p, _ in IDEAS if p == "HIGH")
med_count = sum(1 for _, _, _, p, _ in IDEAS if p == "MED")
low_count = sum(1 for _, _, _, p, _ in IDEAS if p == "LOW")
story.append(Paragraph(
    f"<b>{high_count}</b> high priority &bull; <b>{med_count}</b> medium &bull; <b>{low_count}</b> low &bull; "
    f"<b>12</b> categorieën",
    ParagraphStyle("Stats", parent=styles["Normal"], fontSize=10, textColor=GREY, alignment=TA_CENTER),
))

story.append(PageBreak())

# Group by category
from collections import OrderedDict
categories = OrderedDict()
for cat, title, desc, prio, comp in IDEAS:
    categories.setdefault(cat, []).append((title, desc, prio, comp))

item_num = 0
for cat_name, items in categories.items():
    story.append(Paragraph(cat_name.upper(), cat_style))
    story.append(HRFlowable(width="100%", thickness=0.5, color=ACCENT))
    story.append(Spacer(1, 2*mm))

    for title, desc, prio, comp in items:
        item_num += 1
        prio_style = {"HIGH": priority_high, "MED": priority_med, "LOW": priority_low}[prio]
        complexity_dots = "●" * comp + "○" * (5 - comp)

        row = [[
            Paragraph(f"<b>{item_num:>3}.</b>", ParagraphStyle("Num", parent=styles["Normal"],
                      fontSize=10, textColor=BLUE, fontName="Helvetica-Bold")),
            Paragraph(f"<b>{title}</b>", item_title_style),
            Paragraph(prio, prio_style),
            Paragraph(complexity_dots, ParagraphStyle("Dots", parent=styles["Normal"],
                      fontSize=8, textColor=GREY, alignment=TA_CENTER)),
        ], [
            Paragraph("", item_desc_style),
            Paragraph(desc, item_desc_style),
            Paragraph("", item_desc_style),
            Paragraph("", item_desc_style),
        ]]

        t = Table(row, colWidths=[12*mm, 120*mm, 18*mm, 20*mm])
        t.setStyle(TableStyle([
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('SPAN', (1, 1), (3, 1)),  # Description spans full width
            ('TOPPADDING', (0, 0), (-1, -1), 1),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 0),
            ('BOTTOMPADDING', (0, 1), (-1, 1), 3),
            ('LINEBELOW', (0, 1), (-1, 1), 0.3, HexColor("#eeeeee")),
        ]))
        story.append(t)

    story.append(Spacer(1, 4*mm))

# Final page
story.append(PageBreak())
story.append(Spacer(1, 40*mm))
story.append(Paragraph("PRIORITERING", ParagraphStyle(
    "FinalTitle", parent=styles["Heading1"], fontSize=20, textColor=DARK, alignment=TA_CENTER,
)))
story.append(Spacer(1, 6*mm))
story.append(Paragraph(
    "Begin met de <b>HIGH priority, lage complexiteit</b> items — "
    "die leveren de meeste waarde met de minste effort. "
    "Focus op items die direct je edge verbeteren of risico verlagen. "
    "De advanced ML en infrastructure items zijn langetermijnprojecten "
    "die je kunt plannen voor de weekenden.",
    ParagraphStyle("FinalText", parent=styles["Normal"], fontSize=11, leading=16,
                   textColor=GREY, alignment=TA_CENTER),
))
story.append(Spacer(1, 10*mm))

# Top 10 quick wins
story.append(Paragraph("TOP 10 QUICK WINS", ParagraphStyle(
    "QW", parent=styles["Heading2"], fontSize=14, textColor=ACCENT, alignment=TA_CENTER,
    spaceAfter=4*mm,
)))

quick_wins = [
    (i+1, t, d) for i, (_, t, d, p, c) in enumerate(IDEAS)
    if p == "HIGH" and c <= 2
][:10]

for rank, (num, title, _) in enumerate(quick_wins, 1):
    story.append(Paragraph(
        f"<b>{rank}.</b> #{num} — {title}",
        ParagraphStyle("QWItem", parent=styles["Normal"], fontSize=10, textColor=DARK,
                       leftIndent=20*mm, spaceAfter=2*mm),
    ))

doc.build(story)
print(f"PDF generated: {OUTPUT}")
