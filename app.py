import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import pandas_ta as pd_ta
import alpaca_trade_api as tradeapi # Importation de la bibliothèque Alpaca
from alpaca_trade_api.rest import TimeFrame, TimeFrameUnit # Pour spécifier les timeframes Alpaca
from datetime import datetime, timedelta # Pour gérer les dates

# --- Section 1: Fonction de Calcul de l'Indicateur Canadian Confluence ---
# (Votre fonction calculate_canadian_confluence reste la même)
def calculate_canadian_confluence(df, hmaLength, adxThreshold, rsiLength, adxLength, ichimokuLength, len1_smooth_ha, len2_smooth_ha):
    min_rows_needed = max(hmaLength, adxLength, rsiLength, ichimokuLength, 26, 52, len1_smooth_ha, len2_smooth_ha) + 50
    if len(df) < min_rows_needed:
        return "Erreur", f"Pas assez de données ({len(df)}/{min_rows_needed})", df['Close'].iloc[-1] if not df.empty and 'Close' in df.columns and not df['Close'].empty else 0

    df_calc = df.copy()
    hma_slope_signal, ha_signal, smoothed_ha_signal, rsi_signal, adx_has_momentum_signal, ichimoku_signal = 0,0,0,0,0,0
    current_price = df_calc['Close'].iloc[-1] if not df_calc.empty and 'Close' in df_calc.columns and not df_calc['Close'].empty else 0

    try:
        hma_series = df_calc.ta.hma(length=hmaLength, append=False)
        if hma_series is not None and not hma_series.isna().all() and len(hma_series) >= 2:
            if hma_series.iloc[-1] > hma_series.iloc[-2]: hma_slope_signal = 1
            elif hma_series.iloc[-1] < hma_series.iloc[-2]: hma_slope_signal = -1
        else: return "Erreur HMA", "Calcul impossible (série vide ou <2 valeurs)", current_price
    except Exception as e: return "Erreur HMA", str(e), current_price

    try:
        ha_df = df_calc.ta.ha(append=False)
        if ha_df is not None and 'HA_close' in ha_df.columns and 'HA_open' in ha_df.columns and not ha_df['HA_close'].isna().all() and len(ha_df) > 0:
            if ha_df['HA_close'].iloc[-1] > ha_df['HA_open'].iloc[-1]: ha_signal = 1
            elif ha_df['HA_close'].iloc[-1] < ha_df['HA_open'].iloc[-1]: ha_signal = -1
        else: return "Erreur HA", "Calcul impossible (série vide ou colonnes manquantes)", current_price
    except Exception as e: return "Erreur HA", str(e), current_price

    try:
        ohlc_ema = pd.DataFrame(index=df_calc.index)
        ohlc_ema['Open'] = df_calc['Open'].ewm(span=len1_smooth_ha, adjust=False).mean()
        ohlc_ema['High'] = df_calc['High'].ewm(span=len1_smooth_ha, adjust=False).mean()
        ohlc_ema['Low'] = df_calc['Low'].ewm(span=len1_smooth_ha, adjust=False).mean()
        ohlc_ema['Close'] = df_calc['Close'].ewm(span=len1_smooth_ha, adjust=False).mean()

        ha_on_ema = pd.DataFrame(index=ohlc_ema.index)
        ha_on_ema['haclose_s1'] = (ohlc_ema['Open'] + ohlc_ema['High'] + ohlc_ema['Low'] + ohlc_ema['Close']) / 4
        ha_on_ema['haopen_s1'] = np.nan
        first_valid_idx = ohlc_ema.first_valid_index()
        if first_valid_idx is not None and not ohlc_ema.loc[first_valid_idx, ['Open', 'Close']].isna().any():
            ha_on_ema.loc[first_valid_idx, 'haopen_s1'] = (ohlc_ema.loc[first_valid_idx, 'Open'] + ohlc_ema.loc[first_valid_idx, 'Close']) / 2
        start_loop_idx = ha_on_ema['haopen_s1'].first_valid_index()
        if start_loop_idx is not None:
            start_loop_iloc = ha_on_ema.index.get_loc(start_loop_idx)
            for i in range(start_loop_iloc + 1, len(ha_on_ema)):
                prev_actual_idx, curr_actual_idx = ha_on_ema.index[i-1], ha_on_ema.index[i]
                if not pd.isna(ha_on_ema.loc[prev_actual_idx, 'haopen_s1']) and not pd.isna(ha_on_ema.loc[prev_actual_idx, 'haclose_s1']):
                    ha_on_ema.loc[curr_actual_idx, 'haopen_s1'] = (ha_on_ema.loc[prev_actual_idx, 'haopen_s1'] + ha_on_ema.loc[prev_actual_idx, 'haclose_s1']) / 2
                elif not ohlc_ema.loc[curr_actual_idx, ['Open', 'Close']].isna().any():
                    ha_on_ema.loc[curr_actual_idx, 'haopen_s1'] = (ohlc_ema.loc[curr_actual_idx, 'Open'] + ohlc_ema.loc[curr_actual_idx, 'Close']) / 2
        ha_on_ema.dropna(subset=['haopen_s1', 'haclose_s1'], inplace=True)
        if ha_on_ema.empty:
            return "Erreur HA Lissé", "Données HA_on_EMA vides après dropna", current_price
        smooth_ha_open = ha_on_ema['haopen_s1'].ewm(span=len2_smooth_ha, adjust=False).mean()
        smooth_ha_close = ha_on_ema['haclose_s1'].ewm(span=len2_smooth_ha, adjust=False).mean()
        if not smooth_ha_open.empty and not smooth_ha_close.empty and not smooth_ha_open.isna().all() and not smooth_ha_close.isna().all():
            if smooth_ha_close.iloc[-1] > smooth_ha_open.iloc[-1]: smoothed_ha_signal = 1
            elif smooth_ha_close.iloc[-1] < smooth_ha_open.iloc[-1]: smoothed_ha_signal = -1
        else: return "Erreur HA Lissé", "Calcul EMA impossible (séries vides/NaN)", current_price
    except Exception as e: return "Erreur HA Lissé", str(e), current_price

    try:
        hlc4 = (df_calc['Open'] + df_calc['High'] + df_calc['Low'] + df_calc['Close']) / 4
        rsi_series = pd_ta.rsi(close=hlc4, length=rsiLength, append=False)
        if rsi_series is not None and not rsi_series.isna().all() and len(rsi_series) > 0:
            if rsi_series.iloc[-1] > 50: rsi_signal = 1
            elif rsi_series.iloc[-1] < 50: rsi_signal = -1
        else: return "Erreur RSI", "Calcul impossible (série vide)", current_price
    except Exception as e: return "Erreur RSI", str(e), current_price

    try:
        adx_df = df_calc.ta.adx(length=adxLength, append=False)
        adx_col_name = f'ADX_{adxLength}'
        if adx_df is not None and adx_col_name in adx_df.columns and not adx_df[adx_col_name].isna().all() and len(adx_df) > 0:
            adx_val = adx_df[adx_col_name].iloc[-1]
            if adx_val >= adxThreshold: adx_has_momentum_signal = 1
        else: return "Erreur ADX", f"Calcul impossible (colonne {adx_col_name} manquante ou série vide)", current_price
    except Exception as e: return "Erreur ADX", str(e), current_price

    try:
        tenkan_period, kijun_period, senkou_b_period = ichimokuLength, 26, 52
        tenkan_sen = (df_calc['High'].rolling(window=tenkan_period).max() + df_calc['Low'].rolling(window=tenkan_period).min()) / 2
        kijun_sen = (df_calc['High'].rolling(window=kijun_period).max() + df_calc['Low'].rolling(window=kijun_period).min()) / 2
        senkou_a_current = (tenkan_sen + kijun_sen) / 2
        senkou_b_current = (df_calc['High'].rolling(window=senkou_b_period).max() + df_calc['Low'].rolling(window=senkou_b_period).min()) / 2
        if tenkan_sen.empty or kijun_sen.empty or senkou_a_current.empty or senkou_b_current.empty or \
           tenkan_sen.isna().all() or kijun_sen.isna().all() or senkou_a_current.isna().all() or senkou_b_current.isna().all():
             return "Erreur Ichimoku", "Calcul des lignes Ichimoku impossible (données vides ou NaN)", current_price
        cloud_top_current = pd.concat([senkou_a_current, senkou_b_current], axis=1).max(axis=1)
        cloud_bottom_current = pd.concat([senkou_a_current, senkou_b_current], axis=1).min(axis=1)
        if cloud_top_current.empty or cloud_bottom_current.empty or cloud_top_current.isna().all() or cloud_bottom_current.isna().all():
            return "Erreur Ichimoku", "Calcul du cloud impossible (données vides ou NaN)", current_price
        current_close_val = df_calc['Close'].iloc[-1]
        current_cloud_top_val = cloud_top_current.iloc[-1]
        current_cloud_bottom_val = cloud_bottom_current.iloc[-1]
        if pd.isna(current_close_val) or pd.isna(current_cloud_top_val) or pd.isna(current_cloud_bottom_val):
             return "Erreur Ichimoku", "Données cloud manquantes (NaN à la dernière ligne)", current_price
        if current_close_val > current_cloud_top_val: ichimoku_signal = 1
        elif current_close_val < current_cloud_bottom_val: ichimoku_signal = -1
    except Exception as e: return "Erreur Ichimoku", str(e), current_price
    
    bullConfluences, bearConfluences = 0, 0
    primary_bull = hma_slope_signal == 1 or ha_signal == 1 or smoothed_ha_signal == 1 or rsi_signal == 1 or ichimoku_signal == 1
    primary_bear = hma_slope_signal == -1 or ha_signal == -1 or smoothed_ha_signal == -1 or rsi_signal == -1 or ichimoku_signal == -1
    if hma_slope_signal == 1: bullConfluences += 1
    if ha_signal == 1: bullConfluences += 1
    if smoothed_ha_signal == 1: bullConfluences += 1
    if rsi_signal == 1: bullConfluences += 1
    if ichimoku_signal == 1: bullConfluences +=1 
    if adx_has_momentum_signal == 1 and primary_bull : bullConfluences += 1
    if hma_slope_signal == -1: bearConfluences += 1
    if ha_signal == -1: bearConfluences += 1
    if smoothed_ha_signal == -1: bearConfluences += 1
    if rsi_signal == -1: bearConfluences += 1
    if ichimoku_signal == -1: bearConfluences +=1
    if adx_has_momentum_signal == 1 and primary_bear: bearConfluences += 1
    return bullConfluences, bearConfluences, current_price

# --- Configuration de l'API Alpaca ---
# Utilisez les secrets Streamlit pour stocker vos clés API en production/déploiement
try:
    API_KEY = st.secrets["ALPACA_API_KEY"]
    API_SECRET = st.secrets["ALPACA_SECRET_KEY"]
    BASE_URL = st.secrets["ALPACA_BASE_URL"]
except KeyError: # Au cas où les secrets ne sont pas configurés (pour test local sans secrets)
    st.error("Erreur: Veuillez configurer les secrets ALPACA_API_KEY, ALPACA_SECRET_KEY, et ALPACA_BASE_URL.")
    # Pour test local SANS secrets Streamlit, vous pourriez mettre vos clés ici TEMPORAIREMENT
    # MAIS NE JAMAIS COMMITTER CELA DANS UN REPOSITORY PUBLIC
    # API_KEY = "VOTRE_CLE_API_PAPER_ICI_POUR_TEST_LOCAL"
    # API_SECRET = "VOTRE_CLE_SECRETE_PAPER_ICI_POUR_TEST_LOCAL"
    # BASE_URL = "https://paper-api.alpaca.markets"
    # st.stop() # Arrête l'exécution si les secrets ne sont pas là
    # Pour l'instant, on va laisser continuer avec des valeurs vides si les secrets manquent,
    # mais la connexion échouera. L'idéal est de gérer ça proprement.
    API_KEY = None
    API_SECRET = None
    BASE_URL = None


alpaca_api = None
if API_KEY and API_SECRET and BASE_URL:
    try:
        alpaca_api = tradeapi.REST(API_KEY, API_SECRET, base_url=BASE_URL, api_version='v2')
        # Vérifier la connexion
        account_info = alpaca_api.get_account()
        st.sidebar.success(f"Connecté à Alpaca (Paper: {account_info.paper_only})")
    except Exception as e:
        st.sidebar.error(f"Erreur connexion Alpaca: {e}")
        alpaca_api = None # S'assurer que l'api n'est pas utilisable
else:
    st.sidebar.warning("Clés API Alpaca non configurées via st.secrets.")


# --- Section 2: Configuration de l'Application Streamlit ---
st.set_page_config(layout="wide")
st.title("🚀 Canadian Confluence Premium Scanner (Alpaca Enhanced)")
st.markdown("Analyse les actifs pour des signaux de 5 ou 6 étoiles basés sur l'indicateur Canadian Confluence.")

# ... (vos inputs Streamlit restent les mêmes) ...
st.sidebar.header("⚙️ Paramètres de l'Indicateur")
hmaLength_input = st.sidebar.number_input("HMA Length (Pine: 20)", min_value=1, value=20, step=1, key="hma_len")
adxThreshold_input = st.sidebar.number_input("ADX Threshold (Pine: 20)", min_value=1, value=20, step=1, key="adx_thresh")
rsiLength_input = st.sidebar.number_input("RSI Length (Pine: 10)", min_value=1, value=10, step=1, key="rsi_len")
adxLength_input = st.sidebar.number_input("ADX Length (Pine: 14)", min_value=1, value=14, step=1, key="adx_len")
ichimokuLength_input = st.sidebar.number_input("Ichimoku: Tenkan Length (Pine: 9)", min_value=1, value=9, step=1, key="ichi_tenkan")
len1_input = st.sidebar.number_input("Smoothed HA Length 1 (EMA OHLC, Pine: 10)", min_value=1, value=10, step=1, key="sha_len1")
len2_input = st.sidebar.number_input("Smoothed HA Length 2 (EMA sur HA, Pine: 10)", min_value=1, value=10, step=1, key="sha_len2")

st.sidebar.header("🕒 Paramètres du Scan")
timeframe_options_map = {
    "1 minute": TimeFrame(1, TimeFrameUnit.Minute),
    "5 minutes": TimeFrame(5, TimeFrameUnit.Minute),
    "15 minutes": TimeFrame(15, TimeFrameUnit.Minute),
    "30 minutes": TimeFrame(30, TimeFrameUnit.Minute),
    "1 heure": TimeFrame(1, TimeFrameUnit.Hour),
    "4 heures": TimeFrame(4, TimeFrameUnit.Hour), # Alpaca peut ne pas supporter 4H directement, on prendra 1D ou devra agréger
    "1 jour": TimeFrame(1, TimeFrameUnit.Day),
    # "1 semaine": TimeFrame(1, TimeFrameUnit.Week) # Moins courant pour l'API bars, vérifier la doc
}
timeframe_yf_map = {
    "1 minute": "1m", "5 minutes": "5m", "15 minutes": "15m", "30 minutes": "30m",
    "1 heure": "1h", "4 heures": "4h", "1 jour": "1d", "1 semaine": "1wk"
}
timeframe_display = st.sidebar.selectbox("Unité de Temps", list(timeframe_options_map.keys()), index=4, key="timeframe_select")
alpaca_tf_obj = timeframe_options_map[timeframe_display]
timeframe_yf_str = timeframe_yf_map[timeframe_display] # Pour yfinance

assets_forex_input_default = "EUR/USD,GBP/USD,USD/JPY,AUD/USD,USD/CAD,NZD/USD,USD/CHF" # Notation Alpaca pour Forex
assets_commodities_input_default = "XAU/USD,XAG/USD" # Notation Alpaca pour Or, Argent (spot)
assets_indices_input_default = "^GSPC,^DJI,^IXIC" # Indices, on utilisera yfinance pour ceux-là pour l'instant

assets_forex_input = st.sidebar.text_area("Paires Forex (Alpaca: ex EUR/USD)", assets_forex_input_default, key="forex_assets")
assets_commodities_input = st.sidebar.text_area("Métaux Spot (Alpaca: ex XAU/USD)", assets_commodities_input_default, key="commodities_assets")
assets_indices_input = st.sidebar.text_area("Indices (yfinance: ex ^GSPC)", assets_indices_input_default, key="indices_assets")

results_placeholder = st.empty()
status_placeholder = st.empty()

# Helper function pour déterminer la source de données et le ticker
def get_data_source_and_ticker(asset_input):
    asset_input = asset_input.strip().upper()
    if asset_input.startswith("^"): # Indices yfinance
        return "yfinance", asset_input
    elif "/" in asset_input: # Forex ou Métaux Spot Alpaca (ex: EUR/USD, XAU/USD)
        return "alpaca", asset_input
    # Par défaut, on pourrait essayer yfinance pour les actions, ou spécifier un préfixe/suffixe
    # Pour l'instant, on se concentre sur les distinctions claires
    # On pourrait ajouter une catégorie "Actions US (Alpaca)"
    else: # On suppose yfinance pour les autres (comme les contrats à terme =F)
        return "yfinance", asset_input

if st.sidebar.button("🚀 Lancer le Scan", use_container_width=True, type="primary", key="scan_button_main"):
    # ... (logique de construction de all_assets) ...
    all_asset_tickers_str = f"{assets_forex_input},{assets_commodities_input},{assets_indices_input}"
    all_assets_raw = [ticker.strip() for ticker in all_asset_tickers_str.split(',') if ticker.strip()]


    if not all_assets_raw:
        status_placeholder.warning("Veuillez entrer au moins un ticker d'actif.")
        results_placeholder.empty()
    else:
        results_placeholder.empty()
        status_placeholder.info(f"Scan en cours pour {len(all_assets_raw)} actifs sur {timeframe_display}...")
        st.toast(f"Scan lancé pour {len(all_assets_raw)} actifs...", icon="⏳")
        premium_signals = []
        error_logs = []
        progress_bar_container = st.empty()
        progress_bar = progress_bar_container.progress(0)
        total_assets_count = len(all_assets_raw)

        max_lookback_param = max(hmaLength_input, adxLength_input, rsiLength_input, ichimokuLength_input, 26, 52, len1_input, len2_input)
        # Pour Alpaca, on va calculer start_date et end_date
        end_dt = datetime.now() # Ou pd.Timestamp.now(tz='America/New_York').tz_localize(None) pour Alpaca
        # Calculer start_dt basé sur le nombre de bougies nécessaires
        # Cela dépend du timeframe. Si c'est journalier, max_lookback_param jours. Si horaire, heures, etc.
        # Pour simplifier, on prend une période un peu large et on la filtre après.
        # Exemple: pour des barres de 1H, on veut max_lookback_param + 150 barres.
        # Si 1 bougie = 1 heure, alors on a besoin de (max_lookback_param + 150) heures en arrière.
        if alpaca_tf_obj.unit == TimeFrameUnit.Minute:
            delta_days = ((max_lookback_param + 200) * alpaca_tf_obj.amount) / (60*24) + 5 # +5 jours de marge
        elif alpaca_tf_obj.unit == TimeFrameUnit.Hour:
            delta_days = ((max_lookback_param + 200) * alpaca_tf_obj.amount) / 24 + 10 # +10 jours de marge
        else: # Day
            delta_days = (max_lookback_param + 200) + 30 # +30 jours de marge (pour les weekends etc)
        start_dt = end_dt - timedelta(days=max(delta_days, 10)) # Au moins 10 jours de données

        start_iso = start_dt.isoformat()
        end_iso = end_dt.isoformat()


        # --- MAPPING DES NOMS (GARDER À JOUR) ---
        asset_name_display_mapping = {
            "XAU/USD": "Or (XAU/USD)",
            "XAG/USD": "Argent (XAG/USD)",
            # yfinance tickers
            "^GSPC": "S&P 500", "^DJI": "US30 (Dow Jones)", "^IXIC": "NAS100 (Nasdaq)",
            "^FTSE": "FTSE 100 (UK)", "^GDAXI": "DAX 40 (Allemagne)", "^FCHI": "CAC 40 (France)",
            "^N225": "Nikkei 225 (Japon)",
            "CL=F": "Pétrole (WTI)", "NG=F": "Gaz Naturel" # Si vous les ajoutez via yfinance
        }
        # Pour les paires Forex Alpaca, on peut les formater si besoin, ou laisser tel quel
        # ex: "EUR/USD": "EUR / USD"
        # --- FIN MAPPING ---

        for i, asset_input_str in enumerate(all_assets_raw):
            data_source, asset_ticker_to_use = get_data_source_and_ticker(asset_input_str)

            st.sidebar.info(f"Début: {asset_ticker_to_use} ({data_source}, {i+1}/{total_assets_count})")
            print(f"DEBUG: Début: {asset_ticker_to_use} via {data_source}")

            current_asset_status = f"Traitement de {asset_ticker_to_use} ({i+1}/{total_assets_count})..."
            status_placeholder.text(current_asset_status)
            
            asset_name_display = asset_ticker_to_use # Default
            if asset_ticker_to_use in asset_name_display_mapping:
                asset_name_display = asset_name_display_mapping[asset_ticker_to_use]
            elif "/" in asset_ticker_to_use : # Pour les autres paires forex non mappées
                asset_name_display = asset_ticker_to_use # Garde EUR/USD etc.
            else: # Fallback pour yfinance
                name_temp = asset_ticker_to_use.replace("=X", "").replace(".SI", "").replace("=F", "").replace("^", "")
                asset_name_display = name_temp

            data = pd.DataFrame() # Initialiser data

            try:
                if data_source == "alpaca" and alpaca_api:
                    if alpaca_tf_obj == TimeFrame(4, TimeFrameUnit.Hour): # Gestion spéciale pour 4H
                         st.warning("Alpaca ne supporte pas 4H directement. Utilisation de 1 Jour et scan sauté pour cet actif.")
                         error_logs.append({"Actif": asset_name_display, "Erreur": "Timeframe 4H non supporté directement par Alpaca pour les barres."})
                         continue # Passer à l'actif suivant

                    # Pour Forex et Métaux (qui sont traités comme des "crypto" paires sur Alpaca Data API v2)
                    # ou utiliser get_forex_bars si disponible et adapté
                    print(f"DEBUG: Alpaca request for {asset_ticker_to_use} with TF {alpaca_tf_obj}, S:{start_iso}, E:{end_iso}")
                    bars_df = alpaca_api.get_bars(
                        asset_ticker_to_use,
                        alpaca_tf_obj,
                        start=start_iso, # Format 'YYYY-MM-DDTHH:MM:SSZ' ou juste 'YYYY-MM-DD'
                        end=end_iso,
                        adjustment='raw' # Ou 'split', 'dividend'
                    ).df
                    
                    # Alpaca retourne des données avec index fuseau horaire UTC. pandas-ta préfère sans.
                    if not bars_df.empty:
                        bars_df.index = bars_df.index.tz_convert('UTC').tz_localize(None)
                        # S'assurer que les colonnes sont bien nommées (Alpaca utilise ohlcv en minuscules)
                        bars_df = bars_df.rename(columns={
                            'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'
                        })
                        data = bars_df

                elif data_source == "yfinance":
                    # Logique yfinance existante
                    yf_period_calc = "2y" # Default, ajuster si besoin
                    if timeframe_yf_str == "1m": yf_period_calc = "7d"
                    elif timeframe_yf_str in ["5m", "15m", "30m"]: yf_period_calc = "60d"
                    elif timeframe_yf_str == "1h": yf_period_calc = "730d"
                    
                    print(f"DEBUG: yfinance request for {asset_ticker_to_use} with P:{yf_period_calc}, I:{timeframe_yf_str}")
                    yf_data = yf.download(asset_ticker_to_use, period=yf_period_calc, interval=timeframe_yf_str,
                                          progress=False, auto_adjust=True, timeout=20)
                    if isinstance(yf_data.columns, pd.MultiIndex):
                        yf_data.columns = yf_data.columns.get_level_values(0)
                    data = yf_data
                else:
                    if not alpaca_api and data_source == "alpaca":
                         error_logs.append({"Actif": asset_name_display, "Erreur": "API Alpaca non initialisée."})
                    else:
                        error_logs.append({"Actif": asset_name_display, "Erreur": f"Source de données inconnue: {data_source}"})
                    continue

                # --- Vérification des données et calculs (commun aux deux sources) ---
                if data.empty or len(data) < max_lookback_param + 5: # +5 pour buffer
                    msg = f"Pas assez de données pour {asset_name_display} (obtenues: {len(data)}, requis min: {max_lookback_param+5})."
                    error_logs.append({"Actif": asset_name_display, "Erreur": msg, "Détail": f"Source: {data_source}"})
                    continue

                # S'assurer qu'on ne prend que les dernières N bougies nécessaires pour les calculs
                # candles_to_fetch = max_lookback_param + 150 (défini plus haut)
                data_for_indicator = data.iloc[-(max_lookback_param + 150):] # Prendre les dernières
                if len(data_for_indicator) < max_lookback_param + 5:
                    msg = f"Données insuffisantes après filtrage pour {asset_name_display} ({len(data_for_indicator)}/{max_lookback_param+5})."
                    error_logs.append({"Actif": asset_name_display, "Erreur": msg})
                    continue
                
                # ... (votre logique de calcul Canadian Confluence, signaux, etc. reste ici) ...
                bull_rating, bear_rating, current_price = calculate_canadian_confluence(
                    data_for_indicator, hmaLength_input, adxThreshold_input, rsiLength_input,
                    adxLength_input, ichimokuLength_input, len1_input, len2_input
                )
                if isinstance(bull_rating, str):
                    error_logs.append({"Actif": asset_name_display, "Erreur Calcul": f"{bull_rating}", "Message": f"{bear_rating}"})
                    continue

                signal_text, final_rating_display, signal_type = "NEUTRE", 0, "NEUTRE"
                if bull_rating >= 5 and bull_rating >= bear_rating:
                    stars, signal_text, final_rating_display, signal_type = "⭐" * bull_rating, f"{'⭐' * bull_rating} ACHAT ({bull_rating}c)", bull_rating, "ACHAT"
                elif bear_rating >= 5:
                    stars, signal_text, final_rating_display, signal_type = "⭐" * bear_rating, f"{'⭐' * bear_rating} VENTE ({bear_rating}c)", bear_rating, "VENTE"

                if final_rating_display >= 5:
                    if current_price is None or pd.isna(current_price): price_str = "N/A"
                    elif abs(current_price) < 0.01 and abs(current_price) > 0: price_str = f"{current_price:.6f}"
                    elif abs(current_price) < 10: price_str = f"{current_price:.4f}"
                    else: price_str = f"{current_price:.2f}"
                    premium_signals.append({
                        "Actif": asset_name_display, "Signal": signal_text,
                        "Prix Actuel": price_str, "_raw_rating": max(bull_rating, bear_rating),
                        "_signal_type": signal_type
                    })

            except tradeapi.rest.APIError as अल्प_erreur: # Erreur spécifique à l'API Alpaca
                st.sidebar.warning(f"Alpaca API error for {asset_name_display}: {अल्प_erreur}")
                error_logs.append({"Actif": asset_name_display, "Erreur API Alpaca": str(अल्प_erreur)})
            except Exception as e:
                st.sidebar.error(f"Erreur générale pour {asset_name_display}: {e}")
                error_logs.append({"Actif": asset_name_display, "Erreur Générale": str(e)})
            
            st.sidebar.info(f"Fin: {asset_ticker_to_use} ({data_source})")
            print(f"DEBUG: Fin: {asset_ticker_to_use} via {data_source}")
            progress_bar.progress((i + 1) / total_assets_count)

        # ... (affichage des résultats et erreurs) ...
        progress_bar_container.empty()
        status_placeholder.success("Scan terminé !")
        st.toast("Scan terminé !", icon="✅")
        if premium_signals:
            results_df = pd.DataFrame(premium_signals).sort_values(
                by=["_signal_type", "_raw_rating", "Actif"], ascending=[True, False, True]
            ).drop(columns=["_raw_rating", "_signal_type"])
            results_placeholder.dataframe(results_df, use_container_width=True, hide_index=True)
        else:
            results_placeholder.info("Aucun signal 5 étoiles ou plus détecté avec les paramètres actuels.")
        if error_logs:
            error_df = pd.DataFrame(error_logs)
            error_messages_expander = st.expander("Afficher les erreurs de calcul/API détaillées", expanded=True)
            with error_messages_expander:
                st.warning("Des erreurs sont survenues pendant le scan :")
                st.dataframe(error_df, use_container_width=True, hide_index=True)
else:
    results_placeholder.info("⚙️ Configurez les paramètres et cliquez sur 'Lancer le Scan' pour commencer.")

st.sidebar.markdown("---")
st.sidebar.info("Indicateur: Canadian Confluence. Données: Alpaca & yfinance.")
