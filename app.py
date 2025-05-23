import streamlit as st
import pandas as pd
import numpy as np # Sera importé selon la version de requirements.txt
import yfinance as yf
import pandas_ta as pd_ta # Sera importé selon la version de requirements.txt
from datetime import datetime, timedelta 

# --- Section 1: Fonction de Calcul de l'Indicateur Canadian Confluence ---
def calculate_canadian_confluence(df, hmaLength, adxThreshold, rsiLength, adxLength, ichimokuLength, len1_smooth_ha, len2_smooth_ha):
    min_rows_needed = max(hmaLength, adxLength, rsiLength, ichimokuLength, 26, 52, len1_smooth_ha, len2_smooth_ha) + 50
    if len(df) < min_rows_needed:
        current_price_fallback = 0
        if not df.empty and 'Close' in df.columns and not df['Close'].empty:
            current_price_fallback = df['Close'].iloc[-1]
        return "Erreur", f"Pas assez de données ({len(df)}/{min_rows_needed})", current_price_fallback

    df_calc = df.copy()
    hma_slope_signal, ha_signal, smoothed_ha_signal, rsi_signal, adx_has_momentum_signal, ichimoku_signal = 0,0,0,0,0,0
    current_price = 0 # Default
    if not df_calc.empty and 'Close' in df_calc.columns and not df_calc['Close'].empty:
        current_price = df_calc['Close'].iloc[-1]


    try:
        hma_series = df_calc.ta.hma(length=hmaLength, append=False)
        if hma_series is not None and not hma_series.isna().all() and len(hma_series) >= 2:
            if hma_series.iloc[-1] > hma_series.iloc[-2]: hma_slope_signal = 1
            elif hma_series.iloc[-1] < hma_series.iloc[-2]: hma_slope_signal = -1
        else: return "Erreur HMA", "Calcul HMA impossible (série vide ou <2 valeurs)", current_price
    except Exception as e: return "Erreur HMA", str(e), current_price

    try:
        ha_df = df_calc.ta.ha(append=False)
        if ha_df is not None and 'HA_close' in ha_df.columns and 'HA_open' in ha_df.columns and not ha_df['HA_close'].isna().all() and len(ha_df) > 0:
            if ha_df['HA_close'].iloc[-1] > ha_df['HA_open'].iloc[-1]: ha_signal = 1
            elif ha_df['HA_close'].iloc[-1] < ha_df['HA_open'].iloc[-1]: ha_signal = -1
        else: return "Erreur HA", "Calcul HA impossible (série vide ou colonnes manquantes)", current_price
    except Exception as e: return "Erreur HA", str(e), current_price

    try:
        ohlc_ema = pd.DataFrame(index=df_calc.index)
        ohlc_ema['Open'] = df_calc['Open'].ewm(span=len1_smooth_ha, adjust=False).mean()
        ohlc_ema['High'] = df_calc['High'].ewm(span=len1_smooth_ha, adjust=False).mean()
        ohlc_ema['Low'] = df_calc['Low'].ewm(span=len1_smooth_ha, adjust=False).mean()
        ohlc_ema['Close'] = df_calc['Close'].ewm(span=len1_smooth_ha, adjust=False).mean()

        ha_on_ema = pd.DataFrame(index=ohlc_ema.index)
        ha_on_ema['haclose_s1'] = (ohlc_ema['Open'] + ohlc_ema['High'] + ohlc_ema['Low'] + ohlc_ema['Close']) / 4
        ha_on_ema['haopen_s1'] = np.nan # Utilisez np.nan (minuscule) qui vient de la version de numpy installée
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
        else: return "Erreur HA Lissé", "Calcul EMA HA Lissé impossible", current_price
    except Exception as e: return "Erreur HA Lissé", str(e), current_price

    try:
        hlc4 = (df_calc['Open'] + df_calc['High'] + df_calc['Low'] + df_calc['Close']) / 4
        rsi_series = pd_ta.rsi(close=hlc4, length=rsiLength, append=False)
        if rsi_series is not None and not rsi_series.isna().all() and len(rsi_series) > 0:
            if rsi_series.iloc[-1] > 50: rsi_signal = 1
            elif rsi_series.iloc[-1] < 50: rsi_signal = -1
        else: return "Erreur RSI", "Calcul RSI impossible", current_price
    except Exception as e: return "Erreur RSI", str(e), current_price

    try:
        adx_df = df_calc.ta.adx(length=adxLength, append=False)
        adx_col_name = f'ADX_{adxLength}'
        if adx_df is not None and adx_col_name in adx_df.columns and not adx_df[adx_col_name].isna().all() and len(adx_df) > 0:
            adx_val = adx_df[adx_col_name].iloc[-1]
            if adx_val >= adxThreshold: adx_has_momentum_signal = 1
        else: return "Erreur ADX", f"Calcul ADX impossible (col {adx_col_name})", current_price
    except Exception as e: return "Erreur ADX", str(e), current_price

    try:
        tenkan_period, kijun_period, senkou_b_period = ichimokuLength, 26, 52
        tenkan_sen = (df_calc['High'].rolling(window=tenkan_period).max() + df_calc['Low'].rolling(window=tenkan_period).min()) / 2
        kijun_sen = (df_calc['High'].rolling(window=kijun_period).max() + df_calc['Low'].rolling(window=kijun_period).min()) / 2
        senkou_a_current = (tenkan_sen + kijun_sen) / 2
        senkou_b_current = (df_calc['High'].rolling(window=senkou_b_period).max() + df_calc['Low'].rolling(window=senkou_b_period).min()) / 2
        if tenkan_sen.empty or kijun_sen.empty or senkou_a_current.empty or senkou_b_current.empty or \
           tenkan_sen.isna().all() or kijun_sen.isna().all() or senkou_a_current.isna().all() or senkou_b_current.isna().all():
             return "Erreur Ichimoku", "Calcul lignes Ichimoku impossible", current_price
        cloud_top_current = pd.concat([senkou_a_current, senkou_b_current], axis=1).max(axis=1)
        cloud_bottom_current = pd.concat([senkou_a_current, senkou_b_current], axis=1).min(axis=1)
        if cloud_top_current.empty or cloud_bottom_current.empty or cloud_top_current.isna().all() or cloud_bottom_current.isna().all():
            return "Erreur Ichimoku", "Calcul cloud Ichimoku impossible", current_price
        current_close_val = df_calc['Close'].iloc[-1]
        current_cloud_top_val = cloud_top_current.iloc[-1]
        current_cloud_bottom_val = cloud_bottom_current.iloc[-1]
        if pd.isna(current_close_val) or pd.isna(current_cloud_top_val) or pd.isna(current_cloud_bottom_val):
             return "Erreur Ichimoku", "Données cloud Ichimoku manquantes (NaN)", current_price
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

# --- Section 2: Configuration de l'Application Streamlit ---
st.set_page_config(layout="wide")
st.title("🚀 Canadian Confluence Premium Scanner (yfinance)")
st.markdown("Analyse les actifs pour des signaux de 5 ou 6 étoiles basés sur l'indicateur Canadian Confluence.")

st.sidebar.header("⚙️ Paramètres de l'Indicateur")
hmaLength_input = st.sidebar.number_input("HMA Length (Pine: 20)", min_value=1, value=20, step=1, key="hma_len")
adxThreshold_input = st.sidebar.number_input("ADX Threshold (Pine: 20)", min_value=1, value=20, step=1, key="adx_thresh")
rsiLength_input = st.sidebar.number_input("RSI Length (Pine: 10)", min_value=1, value=10, step=1, key="rsi_len")
adxLength_input = st.sidebar.number_input("ADX Length (Pine: 14)", min_value=1, value=14, step=1, key="adx_len")
ichimokuLength_input = st.sidebar.number_input("Ichimoku: Tenkan Length (Pine: 9)", min_value=1, value=9, step=1, key="ichi_tenkan")
len1_input = st.sidebar.number_input("Smoothed HA Length 1 (EMA OHLC, Pine: 10)", min_value=1, value=10, step=1, key="sha_len1")
len2_input = st.sidebar.number_input("Smoothed HA Length 2 (EMA sur HA, Pine: 10)", min_value=1, value=10, step=1, key="sha_len2")


st.sidebar.header("🕒 Paramètres du Scan")
timeframe_options = {
    "1 minute": "1m", "5 minutes": "5m", "15 minutes": "15m", "30 minutes": "30m",
    "1 heure": "1h", "4 heures": "4h", "1 jour": "1d", "1 semaine": "1wk"
}
timeframe_display = st.sidebar.selectbox("Unité de Temps", list(timeframe_options.keys()), index=4, key="timeframe_select")
timeframe_yf = timeframe_options[timeframe_display]

assets_forex_input_default = "EURUSD=X,GBPUSD=X,USDJPY=X,AUDUSD=X,USDCAD=X,NZDUSD=X,USDCHF=X"
assets_commodities_input_default = "XAUUSD=X,SI=F,CL=F,NG=F" 
assets_indices_input_default = "^GSPC,^DJI,^IXIC,^FTSE,^GDAXI,^FCHI,^N225"

assets_forex_input = st.sidebar.text_area("Paires Forex", assets_forex_input_default, key="forex_assets")
assets_commodities_input = st.sidebar.text_area("Matières Premières", assets_commodities_input_default, key="commodities_assets")
assets_indices_input = st.sidebar.text_area("Indices", assets_indices_input_default, key="indices_assets")

results_placeholder = st.empty()
status_placeholder = st.empty()

if st.sidebar.button("🚀 Lancer le Scan", use_container_width=True, type="primary", key="scan_button_main"):
    all_asset_tickers_str = f"{assets_forex_input},{assets_commodities_input},{assets_indices_input}"
    all_assets = [ticker.strip().upper() for ticker in all_asset_tickers_str.split(',') if ticker.strip()]

    if not all_assets:
        status_placeholder.warning("Veuillez entrer au moins un ticker d'actif.")
        results_placeholder.empty()
    else:
        results_placeholder.empty()
        status_placeholder.info(f"Scan en cours pour {len(all_assets)} actifs sur {timeframe_display}...")
        st.toast(f"Scan lancé pour {len(all_assets)} actifs...", icon="⏳")
        premium_signals = []
        error_logs = []
        progress_bar_container = st.empty()
        progress_bar = progress_bar_container.progress(0)
        total_assets_count = len(all_assets)
        max_lookback_param = max(hmaLength_input, adxLength_input, rsiLength_input, ichimokuLength_input, 26, 52, len1_input, len2_input)
        candles_to_fetch = max_lookback_param + 150 

        asset_name_display_mapping = {
            "XAUUSD=X": "Or (XAU/USD)", "SI=F": "Argent (Futures)", "CL=F": "Pétrole (WTI Futures)", "NG=F": "Gaz Naturel (Futures)",
            "^GSPC": "S&P 500", "^DJI": "US30 (Dow Jones)", "^IXIC": "NAS100 (Nasdaq)",
            "^FTSE": "FTSE 100 (UK)", "^GDAXI": "DAX 40 (Allemagne)", "^FCHI": "CAC 40 (France)",
            "^N225": "Nikkei 225 (Japon)"
        }
        
        for i, asset_ticker in enumerate(all_assets):
            st.sidebar.info(f"Début: {asset_ticker} ({i+1}/{total_assets_count})") 
            print(f"DEBUG: Début traitement: {asset_ticker} ({i+1}/{total_assets_count})")

            current_asset_status = f"Traitement de {asset_ticker} ({i+1}/{total_assets_count})..."
            status_placeholder.text(current_asset_status)
            
            asset_name_display = asset_ticker 
            if asset_ticker in asset_name_display_mapping:
                asset_name_display = asset_name_display_mapping[asset_ticker]
            else: 
                name_temp = asset_ticker.replace("=X", "").replace(".SI", "").replace("=F", "").replace("^", "")
                asset_name_display = name_temp
            
            try:
                yf_period_to_use = "2y" 
                if timeframe_yf == "1m": yf_period_to_use = "7d"
                elif timeframe_yf in ["2m", "5m", "15m", "30m", "90m"]: yf_period_to_use = "60d"
                elif timeframe_yf == "1h":
                    if asset_ticker == "XAUUSD=X":
                        yf_period_to_use = "1y" 
                        print(f"DEBUG: XAUUSD=X 1h, using period: {yf_period_to_use}")
                    else:
                        yf_period_to_use = "730d" 
                elif timeframe_yf == "4h": yf_period_to_use = "2y" 
                elif timeframe_yf == "1d": yf_period_to_use = "5y"
                elif timeframe_yf == "1wk": yf_period_to_use = "10y"

                print(f"DEBUG: yfinance request for {asset_ticker} with P:{yf_period_to_use}, I:{timeframe_yf}")
                data = yf.download(asset_ticker, period=yf_period_to_use, interval=timeframe_yf, progress=False, auto_adjust=True, timeout=20)
                
                if isinstance(data.columns, pd.MultiIndex): data.columns = data.columns.get_level_values(0)
                
                if data.empty or len(data) < max_lookback_param + 5:
                    msg = f"Pas assez de données pour {asset_name_display} (obtenues: {len(data)}, requis min: {max_lookback_param+5}). Intervalle: {timeframe_yf}, Période: {yf_period_to_use}."
                    error_logs.append({"Actif": asset_name_display, "Erreur": msg, "Détail": "Vérifiez la disponibilité des données."})
                    continue
                
                data_for_indicator = data.iloc[-candles_to_fetch:]
                if len(data_for_indicator) < max_lookback_param + 5:
                    msg = f"Données insuffisantes après filtrage pour {asset_name_display} ({len(data_for_indicator)}/{max_lookback_param+5})."
                    error_logs.append({"Actif": asset_name_display, "Erreur": msg})
                    continue

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
            except Exception as e:
                error_logs.append({"Actif": asset_name_display, "Erreur Générale": str(e)})
            
            st.sidebar.info(f"Fin: {asset_ticker}") 
            print(f"DEBUG: Fin traitement: {asset_ticker}")
            progress_bar.progress((i + 1) / total_assets_count)

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
            error_messages_expander = st.expander("Afficher les erreurs de calcul détaillées", expanded=True)
            with error_messages_expander:
                st.warning("Des erreurs sont survenues pendant le scan :")
                st.dataframe(error_df, use_container_width=True, hide_index=True)
else:
    results_placeholder.info("⚙️ Configurez les paramètres et cliquez sur 'Lancer le Scan' pour commencer.")

st.sidebar.markdown("---")
st.sidebar.info("Indicateur: Canadian Confluence. Données: yfinance.")
