# app.py
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import pandas_ta as pd_ta # Assurez-vous que pandas_ta est importé si vous l'utilisez comme pd.ta

# --- Section 1: Fonctions de Calcul de l'Indicateur ---
# (Coller ici la fonction Python qui implémente la logique de votre indicateur)
# Cette fonction prendra df, et tous les paramètres, et retournera
# (bull_confluences, bear_confluences, rating, direction_potentielle)

def calculate_canadian_confluence(df, hmaLength, adxThreshold, rsiLength, adxLength, ichimokuLength, len1, len2):
    # S'assurer que le DataFrame a assez de lignes pour les calculs
    min_rows_needed = max(hmaLength, adxLength, rsiLength, ichimokuLength, 26, 52, len1, len2) + 50 # Marge de sécurité
    if len(df) < min_rows_needed:
        return 0, 0, 0, "Pas assez de données"

    # Copier le df pour éviter les SettingWithCopyWarning
    df_calc = df.copy()

    # 1. HMA
    hma_series = df_calc.ta.hma(length=hmaLength)
    if hma_series is None or len(hma_series) < 2 or hma_series.isna().all(): return 0,0,0, "Erreur HMA"
    hma_slope = 1 if hma_series.iloc[-1] > hma_series.iloc[-2] else -1

    # 2. Heiken Ashi (Standard)
    ha_df = df_calc.ta.ha()
    if ha_df is None or 'HA_close' not in ha_df.columns or 'HA_open' not in ha_df.columns or ha_df['HA_close'].isna().all():
        return 0,0,0, "Erreur HA"
    ha_signal = 1 if ha_df['HA_close'].iloc[-1] > ha_df['HA_open'].iloc[-1] else -1

    # 3. Smoothed Heiken Ashi
    ohlc_ema = pd.DataFrame(index=df_calc.index)
    ohlc_ema['Open'] = df_calc['Open'].ewm(span=len1, adjust=False).mean()
    ohlc_ema['High'] = df_calc['High'].ewm(span=len1, adjust=False).mean()
    ohlc_ema['Low'] = df_calc['Low'].ewm(span=len1, adjust=False).mean()
    ohlc_ema['Close'] = df_calc['Close'].ewm(span=len1, adjust=False).mean()

    ha_on_ema = pd.DataFrame(index=ohlc_ema.index)
    ha_on_ema['haclose'] = (ohlc_ema['Open'] + ohlc_ema['High'] + ohlc_ema['Low'] + ohlc_ema['Close']) / 4
    ha_on_ema['haopen'] = np.nan
    
    first_valid_idx = ohlc_ema.first_valid_index()
    if first_valid_idx is not None and not ohlc_ema.loc[first_valid_idx, ['Open', 'Close']].isna().any():
        ha_on_ema.loc[first_valid_idx, 'haopen'] = (ohlc_ema.loc[first_valid_idx, 'Open'] + ohlc_ema.loc[first_valid_idx, 'Close']) / 2
        for i in range(ohlc_ema.index.get_loc(first_valid_idx) + 1, len(ohlc_ema)):
            prev_actual_idx = ohlc_ema.index[i-1]
            curr_actual_idx = ohlc_ema.index[i]
            if not pd.isna(ha_on_ema.loc[prev_actual_idx, 'haopen']) and not pd.isna(ha_on_ema.loc[prev_actual_idx, 'haclose']):
                 ha_on_ema.loc[curr_actual_idx, 'haopen'] = (ha_on_ema.loc[prev_actual_idx, 'haopen'] + ha_on_ema.loc[prev_actual_idx, 'haclose']) / 2
            elif not ohlc_ema.loc[curr_actual_idx, ['Open', 'Close']].isna().any(): # Fallback if previous was NaN
                 ha_on_ema.loc[curr_actual_idx, 'haopen'] = (ohlc_ema.loc[curr_actual_idx, 'Open'] + ohlc_ema.loc[curr_actual_idx, 'Close']) / 2


    smooth_ha_open = ha_on_ema['haopen'].ewm(span=len2, adjust=False).mean()
    smooth_ha_close = ha_on_ema['haclose'].ewm(span=len2, adjust=False).mean()
    if smooth_ha_open.isna().all() or smooth_ha_close.isna().all(): return 0,0,0, "Erreur HA Lissé"
    smoothed_ha_signal = 1 if smooth_ha_close.iloc[-1] > smooth_ha_open.iloc[-1] else -1


    # 4. RSI
    hlc4 = (df_calc['Open'] + df_calc['High'] + df_calc['Low'] + df_calc['Close']) / 4
    rsi_series = pd_ta.rsi(close=hlc4, length=rsiLength)
    if rsi_series is None or rsi_series.isna().all(): return 0,0,0, "Erreur RSI"
    rsi_signal = 1 if rsi_series.iloc[-1] > 50 else -1

    # 5. ADX
    adx_df = df_calc.ta.adx(length=adxLength)
    if adx_df is None or f'ADX_{adxLength}' not in adx_df.columns or adx_df[f'ADX_{adxLength}'].isna().all():
        return 0,0,0, "Erreur ADX"
    adx_val = adx_df[f'ADX_{adxLength}'].iloc[-1]
    adx_has_momentum = adx_val >= adxThreshold

    # 6. Ichimoku Cloud
    tenkan_period = ichimokuLength
    kijun_period = 26
    senkou_span_b_period = 52

    high_roll_tenkan = df_calc['High'].rolling(window=tenkan_period).max()
    low_roll_tenkan = df_calc['Low'].rolling(window=tenkan_period).min()
    tenkan = (high_roll_tenkan + low_roll_tenkan) / 2

    high_roll_kijun = df_calc['High'].rolling(window=kijun_period).max()
    low_roll_kijun = df_calc['Low'].rolling(window=kijun_period).min()
    kijun = (high_roll_kijun + low_roll_kijun) / 2
    
    senkou_a_current = (tenkan + kijun) / 2
    
    high_roll_senkou_b = df_calc['High'].rolling(window=senkou_span_b_period).max()
    low_roll_senkou_b = df_calc['Low'].rolling(window=senkou_span_b_period).min()
    senkou_b_current = (high_roll_senkou_b + low_roll_senkou_b) / 2
    
    cloud_top_current = pd.concat([senkou_a_current, senkou_b_current], axis=1).max(axis=1)
    cloud_bottom_current = pd.concat([senkou_a_current, senkou_b_current], axis=1).min(axis=1)

    if df_calc['Close'].iloc[-1] == np.nan or cloud_top_current.iloc[-1] == np.nan or cloud_bottom_current.iloc[-1] == np.nan :
        return 0,0,0, "Erreur Ichimoku data"

    current_close_val = df_calc['Close'].iloc[-1]
    current_cloud_top_val = cloud_top_current.iloc[-1]
    current_cloud_bottom_val = cloud_bottom_current.iloc[-1]

    ichimoku_signal = 0
    if current_close_val > current_cloud_top_val: ichimoku_signal = 1
    elif current_close_val < current_cloud_bottom_val: ichimoku_signal = -1
    
    # Confluences
    bullConfluences = 0
    bearConfluences = 0
    if hma_slope == 1: bullConfluences += 1
    if ha_signal == 1: bullConfluences += 1
    if smoothed_ha_signal == 1: bullConfluences += 1
    if rsi_signal == 1: bullConfluences += 1
    if adx_has_momentum: bullConfluences += 1
    if ichimoku_signal == 1: bullConfluences += 1

    if hma_slope == -1: bearConfluences += 1
    if ha_signal == -1: bearConfluences += 1
    if smoothed_ha_signal == -1: bearConfluences += 1 # Pine: o2 > c2 ? -1 (bear) : 1 (bull). Python: smooth_ha_close > smooth_ha_open ? 1 (bull) : -1 (bear)
    if rsi_signal == -1: bearConfluences += 1
    if adx_has_momentum: bearConfluences += 1
    if ichimoku_signal == -1: bearConfluences += 1
    
    rating = 0
    direction = "NEUTRE"

    # Déterminer la direction principale et la note
    # Un signal est généré s'il y a au moins 5 étoiles pour ce côté
    final_bull_rating = 0
    final_bear_rating = 0

    if bullConfluences >= 5:
        final_bull_rating = bullConfluences
    
    if bearConfluences >= 5:
        final_bear_rating = bearConfluences

    return final_bull_rating, final_bear_rating, df_calc['Close'].iloc[-1]


# --- Section 2: Configuration de l'Application Streamlit ---
st.set_page_config(layout="wide")
st.title("🚀 Canadian Confluence Premium Scanner")

# Paramètres dans la sidebar
st.sidebar.header("⚙️ Paramètres de l'Indicateur")
hmaLength_input = st.sidebar.number_input("HMA Length", min_value=1, value=20)
adxThreshold_input = st.sidebar.number_input("ADX Threshold", min_value=1, value=20)
rsiLength_input = st.sidebar.number_input("RSI Length", min_value=1, value=10)
adxLength_input = st.sidebar.number_input("ADX Length", min_value=1, value=14)
ichimokuLength_input = st.sidebar.number_input("Ichimoku: Tenkan Length", min_value=1, value=9) # Kijun=26, SenkouB=52 sont fixes
len1_input = st.sidebar.number_input("Smoothed HA Length 1 (EMA OHLC)", min_value=1, value=10)
len2_input = st.sidebar.number_input("Smoothed HA Length 2 (EMA sur HA)", min_value=1, value=10)

st.sidebar.header("🕒 Paramètres du Scan")
timeframe_options = ["15m", "30m", "1h", "4h", "1d"] # "1wk", "1mo"
# Pour yfinance, les intervalles < 1h nécessitent des données des 60 derniers jours.
# "1m": max 7 jours, "1h": max 730 jours
timeframe_input = st.sidebar.selectbox("Unité de Temps", timeframe_options, index=2) # Default to 1h

# Liste des actifs (personnalisez avec les tickers yfinance corrects)
assets_forex = ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X", "USDCAD=X", "NZDUSD=X", "USDCHF=X"]
assets_commodities = ["GC=F"] # Or
assets_indices = ["^DJI", "^NDX", "^GSPC"] # NDX pour Nasdaq 100, DJI pour US30, GSPC pour S&P 500
# ^IXIC est le Nasdaq Composite, ^NDX est le Nasdaq 100. Choisissez celui que vous préférez.
all_assets = assets_forex + assets_commodities + assets_indices

# Pour déterminer la période de téléchargement des données
# Ex: pour du 1h, télécharger 730 jours (max pour yfinance) pour avoir assez de barres
# Pour du 15m, télécharger 60 jours.
# Il faut assez de barres pour que tous les indicateurs se calculent (max lookback est 52 pour Senkou B)
# + quelques barres supplémentaires. Disons 200 barres.
# yfinance period: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max
# Si interval < 1h, period ne doit pas dépasser 60d.
# Pour 1h et plus, on peut prendre plus.
# On va demander un nombre de bougies suffisant.
num_candles_needed = max(hmaLength_input, adxLength_input, rsiLength_input, ichimokuLength_input, 26, 52, len1_input, len2_input) + 100 # Marge

# Placeholder pour les résultats
results_placeholder = st.empty()

if st.sidebar.button(" Lancer le Scan ", use_container_width=True):
    st.toast(f"Scan en cours pour {len(all_assets)} actifs sur {timeframe_input}...")
    premium_signals = []

    progress_bar = st.progress(0)
    total_assets = len(all_assets)

    for i, asset_ticker in enumerate(all_assets):
        try:
            # Déterminer la période de téléchargement pour yfinance
            # Si l'intervalle est inférieur à 1 jour, la période ne peut pas être trop longue.
            # Pour les intervalles intrajournaliers (<1d), la période ne peut pas dépasser 60 jours.
            # Pour 1h, yfinance permet jusqu'à 730 jours.
            # Pour 1d, on peut prendre "2y" par exemple.
            # Nous allons essayer de récupérer 'num_candles_needed'.
            # yfinance ne permet pas de spécifier un nombre de bougies directement avec 'period'.
            # On va prendre une période assez large et la tronquer.
            
            current_period = "2y" # Default pour 1d et plus
            if timeframe_input in ["1m", "2m", "5m", "15m", "30m"]:
                current_period = "59d" # Max pour ces intervalles
            elif timeframe_input == "1h":
                current_period = "729d" # Max pour 1h
            
            data = yf.download(asset_ticker, period=current_period, interval=timeframe_input, progress=False, auto_adjust=True)
            
            if data.empty or len(data) < num_candles_needed:
                st.warning(f"Pas assez de données pour {asset_ticker} sur {timeframe_input} (besoin de {num_candles_needed}, obtenu {len(data)}).")
                continue
            
            # Garder seulement les N dernières bougies nécessaires + un peu plus pour être sûr
            data_subset = data.iloc[-num_candles_needed:]

            bull_rating, bear_rating, current_price = calculate_canadian_confluence(
                data_subset, hmaLength_input, adxThreshold_input, rsiLength_input, 
                adxLength_input, ichimokuLength_input, len1_input, len2_input
            )

            if isinstance(bull_rating, str): # Erreur retournée
                st.error(f"Erreur de calcul pour {asset_ticker}: {bull_rating}")
                continue

            asset_name = asset_ticker.replace("=X", "") # Pour un affichage plus propre

            if bull_rating >= 5:
                stars = "⭐" * bull_rating
                premium_signals.append({
                    "Actif": asset_name, 
                    "Signal": f"{stars} ACHAT", 
                    "Note": bull_rating,
                    "Prix Actuel": f"{current_price:.4f}" if asset_ticker not in assets_indices else f"{current_price:.2f}"
                })
            
            if bear_rating >= 5:
                stars = "⭐" * bear_rating
                premium_signals.append({
                    "Actif": asset_name, 
                    "Signal": f"{stars} VENTE", 
                    "Note": bear_rating,
                    "Prix Actuel": f"{current_price:.4f}" if asset_ticker not in assets_indices else f"{current_price:.2f}"
                })

        except Exception as e:
            st.error(f"Erreur lors du traitement de {asset_ticker}: {e}")
        
        progress_bar.progress((i + 1) / total_assets)

    st.toast("Scan terminé!", icon="✅")
    if premium_signals:
        results_df = pd.DataFrame(premium_signals)
        results_df = results_df.sort_values(by="Note", ascending=False)
        results_placeholder.dataframe(results_df, use_container_width=True)
    else:
        results_placeholder.info("Aucun signal 5 ou 6 étoiles détecté avec les paramètres actuels.")

else:
    results_placeholder.info("Cliquez sur 'Lancer le Scan' pour commencer.")


st.sidebar.markdown("---")
st.sidebar.markdown("Développé avec l'aide de Gemini.")
