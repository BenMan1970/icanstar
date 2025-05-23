import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import pandas_ta as pd_ta

# --- Section 1: Fonction de Calcul de l'Indicateur Canadian Confluence ---
# La fonction reste la même, mais elle ne sera pas appelée dans cette version de test
def calculate_canadian_confluence(df, hmaLength, adxThreshold, rsiLength, adxLength, ichimokuLength, len1_smooth_ha, len2_smooth_ha):
    """
    Calcule les confluences haussières et baissières selon l'indicateur Canadian Confluence.
    Retourne (bull_confluences, bear_confluences, current_price_or_error_message).
    """
    # S'assurer que le DataFrame a assez de lignes pour les calculs
    min_rows_needed = max(hmaLength, adxLength, rsiLength, ichimokuLength, 26, 52, len1_smooth_ha, len2_smooth_ha) + 50
    if len(df) < min_rows_needed:
        return "Erreur", f"Pas assez de données ({len(df)}/{min_rows_needed})", 0

    df_calc = df.copy()
    hma_slope_signal, ha_signal, smoothed_ha_signal, rsi_signal, adx_has_momentum_signal, ichimoku_signal = 0,0,0,0,0,0

    try:
        hma_series = df_calc.ta.hma(length=hmaLength, append=False)
        if hma_series is not None and not hma_series.isna().all() and len(hma_series) >= 2:
            if hma_series.iloc[-1] > hma_series.iloc[-2]: hma_slope_signal = 1
            elif hma_series.iloc[-1] < hma_series.iloc[-2]: hma_slope_signal = -1
        else: return "Erreur HMA", "Calcul impossible", df_calc['Close'].iloc[-1]
    except Exception as e: return "Erreur HMA", str(e), df_calc['Close'].iloc[-1]

    try:
        ha_df = df_calc.ta.ha(append=False)
        if ha_df is not None and 'HA_close' in ha_df.columns and 'HA_open' in ha_df.columns and not ha_df['HA_close'].isna().all():
            if ha_df['HA_close'].iloc[-1] > ha_df['HA_open'].iloc[-1]: ha_signal = 1
            elif ha_df['HA_close'].iloc[-1] < ha_df['HA_open'].iloc[-1]: ha_signal = -1
        else: return "Erreur HA", "Calcul impossible", df_calc['Close'].iloc[-1]
    except Exception as e: return "Erreur HA", str(e), df_calc['Close'].iloc[-1]

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
        
        smooth_ha_open = ha_on_ema['haopen_s1'].ewm(span=len2_smooth_ha, adjust=False).mean()
        smooth_ha_close = ha_on_ema['haclose_s1'].ewm(span=len2_smooth_ha, adjust=False).mean()

        if not smooth_ha_open.isna().all() and not smooth_ha_close.isna().all():
            if smooth_ha_close.iloc[-1] > smooth_ha_open.iloc[-1]: smoothed_ha_signal = 1
            elif smooth_ha_close.iloc[-1] < smooth_ha_open.iloc[-1]: smoothed_ha_signal = -1
        else: return "Erreur HA Lissé", "Calcul EMA impossible", df_calc['Close'].iloc[-1]
    except Exception as e: return "Erreur HA Lissé", str(e), df_calc['Close'].iloc[-1]

    try:
        hlc4 = (df_calc['Open'] + df_calc['High'] + df_calc['Low'] + df_calc['Close']) / 4
        rsi_series = pd_ta.rsi(close=hlc4, length=rsiLength, append=False)
        if rsi_series is not None and not rsi_series.isna().all():
            if rsi_series.iloc[-1] > 50: rsi_signal = 1
            elif rsi_series.iloc[-1] < 50: rsi_signal = -1
        else: return "Erreur RSI", "Calcul impossible", df_calc['Close'].iloc[-1]
    except Exception as e: return "Erreur RSI", str(e), df_calc['Close'].iloc[-1]

    try:
        adx_df = df_calc.ta.adx(length=adxLength, append=False)
        if adx_df is not None and f'ADX_{adxLength}' in adx_df.columns and not adx_df[f'ADX_{adxLength}'].isna().all():
            adx_val = adx_df[f'ADX_{adxLength}'].iloc[-1]
            if adx_val >= adxThreshold: adx_has_momentum_signal = 1
        else: return "Erreur ADX", "Calcul impossible", df_calc['Close'].iloc[-1]
    except Exception as e: return "Erreur ADX", str(e), df_calc['Close'].iloc[-1]

    try:
        tenkan_period, kijun_period, senkou_b_period = ichimokuLength, 26, 52
        tenkan_sen = (df_calc['High'].rolling(window=tenkan_period).max() + df_calc['Low'].rolling(window=tenkan_period).min()) / 2
        kijun_sen = (df_calc['High'].rolling(window=kijun_period).max() + df_calc['Low'].rolling(window=kijun_period).min()) / 2
        senkou_a_current = (tenkan_sen + kijun_sen) / 2
        senkou_b_current = (df_calc['High'].rolling(window=senkou_b_period).max() + df_calc['Low'].rolling(window=senkou_b_period).min()) / 2
        cloud_top_current = pd.concat([senkou_a_current, senkou_b_current], axis=1).max(axis=1)
        cloud_bottom_current = pd.concat([senkou_a_current, senkou_b_current], axis=1).min(axis=1)
        current_close_val, current_cloud_top_val, current_cloud_bottom_val = df_calc['Close'].iloc[-1], cloud_top_current.iloc[-1], cloud_bottom_current.iloc[-1]

        if pd.isna(current_close_val) or pd.isna(current_cloud_top_val) or pd.isna(current_cloud_bottom_val):
             return "Erreur Ichimoku", "Données cloud manquantes", df_calc['Close'].iloc[-1]
        if current_close_val > current_cloud_top_val: ichimoku_signal = 1
        elif current_close_val < current_cloud_bottom_val: ichimoku_signal = -1
    except Exception as e: return "Erreur Ichimoku", str(e), df_calc['Close'].iloc[-1]
    
    bullConfluences, bearConfluences = 0, 0
    if hma_slope_signal == 1: bullConfluences += 1
    if ha_signal == 1: bullConfluences += 1
    if smoothed_ha_signal == 1: bullConfluences += 1
    if rsi_signal == 1: bullConfluences += 1
    if adx_has_momentum_signal == 1: bullConfluences += 1
    if ichimoku_signal == 1: bullConfluences += 1

    if hma_slope_signal == -1: bearConfluences += 1
    if ha_signal == -1: bearConfluences += 1
    if smoothed_ha_signal == -1: bearConfluences += 1
    if rsi_signal == -1: bearConfluences += 1
    if adx_has_momentum_signal == 1: bearConfluences += 1
    if ichimoku_signal == -1: bearConfluences += 1
    
    return bullConfluences, bearConfluences, df_calc['Close'].iloc[-1]


# --- Section 2: Configuration de l'Application Streamlit ---
st.set_page_config(layout="wide")
st.title("🚀 Canadian Confluence Premium Scanner")
st.markdown("Analyse les actifs pour des signaux de 5 ou 6 étoiles basés sur l'indicateur Canadian Confluence.")

# Paramètres dans la sidebar
st.sidebar.header("⚙️ Paramètres de l'Indicateur")
hmaLength_input = st.sidebar.number_input("HMA Length (Pine: 20)", min_value=1, value=20, step=1)
adxThreshold_input = st.sidebar.number_input("ADX Threshold (Pine: 20)", min_value=1, value=20, step=1)
rsiLength_input = st.sidebar.number_input("RSI Length (Pine: 10)", min_value=1, value=10, step=1)
adxLength_input = st.sidebar.number_input("ADX Length (Pine: 14)", min_value=1, value=14, step=1)
ichimokuLength_input = st.sidebar.number_input("Ichimoku: Tenkan Length (Pine: 9)", min_value=1, value=9, step=1)
len1_input = st.sidebar.number_input("Smoothed HA Length 1 (EMA OHLC, Pine: 10)", min_value=1, value=10, step=1)
len2_input = st.sidebar.number_input("Smoothed HA Length 2 (EMA sur HA, Pine: 10)", min_value=1, value=10, step=1)

st.sidebar.header("🕒 Paramètres du Scan")
timeframe_options = {
    "1 minute": "1m", "5 minutes": "5m", "15 minutes": "15m", "30 minutes": "30m",
    "1 heure": "1h", "4 heures": "4h", "1 jour": "1d", "1 semaine": "1wk"
}
timeframe_display = st.sidebar.selectbox("Unité de Temps", list(timeframe_options.keys()), index=4) # Default to 1 heure
timeframe_yf = timeframe_options[timeframe_display]

assets_forex_input_default = "EURUSD=X,GBPUSD=X,USDJPY=X,AUDUSD=X,USDCAD=X,NZDUSD=X,USDCHF=X"
assets_commodities_input_default = "GC=F,SI=F"
assets_indices_input_default = "^DJI,^IXIC,^GSPC"

assets_forex_input = st.sidebar.text_area("Paires Forex", assets_forex_input_default)
assets_commodities_input = st.sidebar.text_area("Matières Premières", assets_commodities_input_default)
assets_indices_input = st.sidebar.text_area("Indices", assets_indices_input_default)

# Placeholder pour les résultats et messages
results_placeholder = st.empty()
status_placeholder = st.empty()
error_messages_expander = st.expander("Afficher les erreurs de calcul détaillées (si applicable)", expanded=False)


# --- MODIFICATION POUR TESTER LE BOUTON ---
if st.sidebar.button(" Lancer le Scan ", use_container_width=True, type="primary", key="scan_button_test"):
    status_placeholder.success("Le bouton 'Lancer le Scan' a été cliqué avec succès !")
    st.toast("Bouton cliqué !", icon="🎉")
    
    # Tout le code de scan est temporairement désactivé ci-dessous
    # pour isoler le fonctionnement du bouton.
    # Si le message ci-dessus s'affiche, le bouton fonctionne.
    # Nous réactiverons le code de scan par la suite.

    # # DEBUT DU CODE DE SCAN ORIGINAL (COMMENTÉ)
    # all_asset_tickers_str = f"{assets_forex_input},{assets_commodities_input},{assets_indices_input}"
    # all_assets = [ticker.strip().upper() for ticker in all_asset_tickers_str.split(',') if ticker.strip()]

    # if not all_assets:
    #     status_placeholder.warning("Veuillez entrer au moins un ticker d'actif.")
    # else:
    #     status_placeholder.info(f"Scan en cours pour {len(all_assets)} actifs sur {timeframe_display}...")
    #     st.toast(f"Scan lancé pour {len(all_assets)} actifs...", icon="⏳")
        
    #     premium_signals = []
    #     error_logs = []

    #     progress_bar = results_placeholder.progress(0) # Mettre la progress bar dans results_placeholder
    #     total_assets_count = len(all_assets)

    #     max_lookback = max(hmaLength_input, adxLength_input, rsiLength_input, ichimokuLength_input, 26, 52, len1_input, len2_input)
    #     candles_to_fetch = max_lookback + 150 

    #     yf_period = "2y" 
    #     if timeframe_yf == "1m": yf_period = "7d"
    #     elif timeframe_yf in ["2m", "5m", "15m", "30m", "90m"]: yf_period = "59d"
    #     elif timeframe_yf == "1h": yf_period = "729d"
        
    #     for i, asset_ticker in enumerate(all_assets):
    #         current_asset_status = f"Traitement de {asset_ticker} ({i+1}/{total_assets_count})..."
    #         status_placeholder.text(current_asset_status) # Mettre à jour le statut ici
    #         try:
    #             data = yf.download(asset_ticker, period=yf_period, interval=timeframe_yf, progress=False, auto_adjust=True, timeout=20)
                
    #             if data.empty or len(data) < max_lookback + 5:
    #                 msg = f"Pas assez de données pour {asset_ticker} ({len(data)}/{max_lookback+5})."
    #                 error_logs.append({"Actif": asset_ticker, "Erreur": msg})
    #                 st.sidebar.warning(msg, icon="⚠️")
    #                 continue
                
    #             data_for_indicator = data.iloc[-candles_to_fetch:]
    #             if len(data_for_indicator) < max_lookback + 5:
    #                 msg = f"Données insuffisantes après filtrage pour {asset_ticker} ({len(data_for_indicator)}/{max_lookback+5})."
    #                 error_logs.append({"Actif": asset_ticker, "Erreur": msg})
    #                 st.sidebar.warning(msg, icon="⚠️")
    #                 continue

    #             bull_rating, bear_rating, current_price = calculate_canadian_confluence(
    #                 data_for_indicator, hmaLength_input, adxThreshold_input, rsiLength_input, 
    #                 adxLength_input, ichimokuLength_input, len1_input, len2_input
    #             )
    #             asset_name_display = asset_ticker.replace("=X", "").replace(".SI", "")

    #             if isinstance(bull_rating, str): 
    #                 error_logs.append({"Actif": asset_name_display, "Erreur": f"{bull_rating}: {bear_rating}"})
    #                 continue

    #             signal_text, final_rating = "NEUTRE", 0
    #             if bull_rating >= 5 and bull_rating >= bear_rating:
    #                 stars, signal_text, final_rating = "⭐" * bull_rating, f"{'⭐' * bull_rating} ACHAT", bull_rating
    #             elif bear_rating >= 5:
    #                 stars, signal_text, final_rating = "⭐" * bear_rating, f"{'⭐' * bear_rating} VENTE", bear_rating
                
    #             if final_rating >= 5:
    #                 price_format = "{:.4f}" if asset_ticker not in assets_indices_input.split(',') and asset_ticker not in ["GC=F", "SI=F"] else "{:.2f}"
    #                 premium_signals.append({
    #                     "Actif": asset_name_display, "Signal": signal_text, "Note": final_rating,
    #                     "Prix Actuel": price_format.format(current_price) if pd.notna(current_price) else "N/A"
    #                 })
    #         except Exception as e:
    #             error_logs.append({"Actif": asset_ticker, "Erreur Générale": str(e)})
            
    #         progress_bar.progress((i + 1) / total_assets_count)

    #     status_placeholder.success("Scan terminé!") # Mettre à jour le statut ici
    #     st.toast("Scan terminé!", icon="✅")

    #     if premium_signals:
    #         results_df = pd.DataFrame(premium_signals).sort_values(by=["Note", "Actif"], ascending=[False, True])
    #         results_placeholder.dataframe(results_df, use_container_width=True, hide_index=True) # Afficher les résultats ici
    #     else:
    #         results_placeholder.info("Aucun signal 5 ou 6 étoiles détecté.") # Message si aucun résultat
        
    #     if error_logs:
    #         error_df = pd.DataFrame(error_logs)
    #         error_messages_expander.warning("Des erreurs sont survenues :")
    #         error_messages_expander.dataframe(error_df, use_container_width=True, hide_index=True)
    # # FIN DU CODE DE SCAN ORIGINAL (COMMENTÉ)

else:
    results_placeholder.info("Cliquez sur 'Lancer le Scan' pour commencer.")


st.sidebar.markdown("---")
st.sidebar.markdown("Indicateur original : Canadian Confluence.")
st.sidebar.markdown("Application développée avec l'aide de Gemini.")
