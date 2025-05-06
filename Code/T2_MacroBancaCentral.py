import os
import numpy as np
import pandas as pd
import seaborn as sns
import yfinance as yf
import statsmodels.api as sm
import matplotlib.pyplot as plt

ruta_base = r"C:\Users\User\Desktop\BancaCentral\Taller2_BCentral\Outputs"
data_bases = r"C:\Users\User\Desktop\BancaCentral\Taller2_BCentral\Data"

# Configuración
sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)

# Periodo de análisis
start = "2010-01-01"
end = "2020-12-31"

# Descargar activos
tickers = {
    "COPUSD": "COP=X" # Tasa de cambio COL/USD
}

# Función para obtener datos
def get_data(ticker_symbol, name):
    ticker = yf.Ticker(ticker_symbol)
    data = ticker.history(start=start, end=end)
    return data["Close"].rename(name)

# Obtener series
wti = pd.read_csv(os.path.join(data_bases, "WTI.csv"))
wti_df = pd.DataFrame(wti).bfill()
wti_df = wti_df.rename(columns={
    'observation_date': 'Date',
    'DCOILWTICO': 'WTI'
    })
wti_df['Date'] = pd.to_datetime(wti_df['Date'], errors='coerce')
wti_df['Date'] = wti_df['Date'].dt.tz_localize(None)
wti_df = wti_df[wti_df['WTI'] >= 0] #Acotar a los valores realistas no negativos

brent = pd.read_csv(os.path.join(data_bases, "BRENT.csv"))
brent_df = pd.DataFrame(brent).bfill()
brent_df = brent_df.rename(columns={
    'observation_date': 'Date',
    'DCOILBRENTEU': 'Brent'
    })
brent_df['Date'] = pd.to_datetime(brent_df['Date'], errors='coerce')
brent_df['Date'] = brent_df['Date'].dt.tz_localize(None)


fx = get_data(tickers["COPUSD"], "COP/USD")
fx = fx[fx >= 1700]

fx_df = pd.DataFrame(fx)
fx_df['Date'] = pd.to_datetime(fx_df.index)
fx_df['Date'] = fx_df['Date'].dt.tz_localize(None)
fx_df.reset_index(drop=True, inplace=True)

#-----------Generacion del gráfico-------------
fig, axs = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

# WTI
axs[0].plot(wti_df['Date'], wti_df['WTI'], color='orange', label='WTI')
axs[0].set_ylabel('WTI (USD)')
axs[0].legend()
axs[0].grid(True)

# Brent
axs[1].plot(brent_df['Date'], brent_df['Brent'], color='green', label='Brent')
axs[1].set_ylabel('Brent (USD)')
axs[1].legend()
axs[1].grid(True)

# COP-USD
axs[2].plot(fx_df['Date'], fx_df['COP/USD'], color='purple', label='COP/USD')
axs[2].set_ylabel('COP/USD')
axs[2].legend()
axs[2].grid(True)

# Eje X común
axs[2].set_xlabel('Fecha')
fig.suptitle('Precios del Petróleo y Tipo de Cambio (2010 a 2020)', fontsize=14)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(os.path.join(ruta_base, "Graficos_WTI_Brent_&_COPUSD.png"), dpi=300)
plt.show()


wti_df['WTI-pct_change'] = wti_df['WTI'].pct_change() * 100
brent_df['Brent-pct_change'] = brent_df['Brent'].pct_change() * 100
fx_df['COP/USD-pct_change'] = fx_df['COP/USD'].pct_change()  * 100



wti_df.reset_index(drop=True, inplace=True)
brent_df.reset_index(drop=True, inplace=True)
fx_df.reset_index(drop=True, inplace=True)


fx_df['COP/USD'] = fx_df['COP/USD'].fillna(fx_df['COP/USD'].rolling(window=5, min_periods=1).mean())
wti_df['WTI'] = wti_df['WTI'].fillna(wti_df['WTI'].rolling(window=5, min_periods=1).mean())
brent_df['Brent'] = brent_df['Brent'].fillna(brent_df['Brent'].rolling(window=5, min_periods=1).mean())

wti_df['WTI-Normalized'] = 100 * (wti_df['WTI'] / wti_df['WTI'].iloc[0])
brent_df['Brent-Normalized'] = 100 * (brent_df['Brent'] / brent_df['Brent'].iloc[0])
fx_df['COP/USD-Normalized'] = 100 * (fx_df['COP/USD'] / fx_df['COP/USD'].iloc[0])

wti_df['WTI-percentage'] = wti_df['WTI-Normalized'] - 100
brent_df['Brent-percentage'] = brent_df['Brent-Normalized'] - 100
fx_df['COP/USD-percentage'] = fx_df['COP/USD-Normalized'] -100


#-----------Generacion del gráfico-------------
plt.figure(figsize=(10, 6))
plt.plot(wti_df['Date'], wti_df['WTI-percentage'], label='WTI Normalized', color='blue')
plt.plot(brent_df['Date'], brent_df['Brent-percentage'], label='Brent Normalized', color='purple')
plt.plot(fx_df['Date'], fx_df['COP/USD-percentage'], label='COP/USD Normalized', color='orange')
plt.xlabel('Date')
plt.ylabel('Normalized Price (percentage change)')
plt.title('Normalized Price of WTI and COP/USD (percentage change accumulate)')
plt.legend()
plt.xticks(rotation=45)
plt.grid(True)
plt.savefig(os.path.join(ruta_base, "percentage_change_assests.png"), dpi=300)  
plt.show()

#-----------Regresión para cambio del 1%-------------
# Aseguramos que las fechas coincidan entre los tres dataframes
# Asegúrate de que todas las fechas sean tipo datetime y sin zona horaria
wti_df['Date'] = pd.to_datetime(wti_df['Date']).dt.tz_localize(None)
brent_df['Date'] = pd.to_datetime(brent_df['Date']).dt.tz_localize(None)
fx_df['Date'] = pd.to_datetime(fx_df['Date']).dt.tz_localize(None)

merged = pd.merge(wti_df[['Date', 'WTI-pct_change']],
                  brent_df[['Date', 'Brent-pct_change']],
                  on='Date', how='inner')
merged = pd.merge(merged, fx_df[['Date', 'COP/USD-pct_change']], on='Date', how='inner')
merged = merged.fillna(0)


merged = merged.rename(columns={
    'WTI-pct_change': 'WTI_pct',
    'Brent-pct_change': 'Brent_pct',
    'COP/USD-pct_change': 'COP_USD_pct'
})

# Regresión 1: WTI vs COP/USD
X_wti = sm.add_constant(merged['WTI_pct'])
model_wti = sm.OLS(merged['COP_USD_pct'], X_wti).fit()
model_wti.summary()
# Regresión 2: Brent vs COP/USD
X_brent = sm.add_constant(merged['Brent_pct'])
model_brent = sm.OLS(merged['COP_USD_pct'], X_brent).fit()
model_brent.summary()

# Crear DataFrame con resultados
summary_df = pd.DataFrame({
    'WTI_coef': model_wti.params,
    'WTI_pval': model_wti.pvalues,
    'WTI_R2': model_wti.rsquared,
    'Brent_coef': model_brent.params,
    'Brent_pval': model_brent.pvalues,
    'Brent_R2': model_brent.rsquared
})

# Exportar a Excel
summary_df.to_excel(os.path.join(ruta_base,'regresiones_petroleo_fx.xlsx'), index_label='Variable')

#-----------Comparison with short term interest rate yields-------------
#load data
col3myield = pd.read_csv(os.path.join(data_bases, "COL_3M_Yield.csv"))
col3myield_df = pd.DataFrame(col3myield).bfill()
col3myield_df = col3myield_df.rename(columns={
    'observation_date': 'Date',
    'COLIR3TIB01STM': 'col3myield'
    })
col3myield_df['Date'] = pd.to_datetime(col3myield_df['Date'], errors='coerce')
col3myield_df['Date'] = col3myield_df['Date'].dt.tz_localize(None)

us3myield = pd.read_csv(os.path.join(data_bases, "US_3M_Yield.csv"))
us3myield_df = pd.DataFrame(us3myield).bfill()
us3myield_df = us3myield_df.rename(columns={
    'observation_date': 'Date',
    'DTB3': 'us3myield'
    })
us3myield_df['Date'] = pd.to_datetime(us3myield_df['Date'], errors='coerce')
us3myield_df['Date'] = us3myield_df['Date'].dt.tz_localize(None)

# Asegúrate de que todas las fechas sean tipo datetime y sin zona horaria
col3myield_df['Date'] = pd.to_datetime(col3myield_df['Date']).dt.tz_localize(None)
us3myield_df['Date'] = pd.to_datetime(us3myield_df['Date']).dt.tz_localize(None)

# Calculamos los cambios diarios de precios
wti_df['WTI_diff'] = wti_df['WTI'].diff()
brent_df['Brent_diff'] = brent_df['Brent'].diff()
# Suponemos una producción de 1 millón de barriles por día
produccion_diaria = 1_000_000

# Calcula las ganancias/pérdidas diarias en USD
wti_df['USD_WTI'] = wti_df['WTI_diff'] * produccion_diaria
brent_df['USD_Brent'] = brent_df['Brent_diff'] * produccion_diaria

dollar_flow = pd.merge(wti_df[['Date', 'USD_WTI']],
                  brent_df[['Date', 'USD_Brent']],
                  on='Date', how='inner')

dollar_flow = pd.merge(dollar_flow,
                  col3myield_df[['Date', 'col3myield']],
                  on='Date', how='outer')
dollar_flow = pd.merge(dollar_flow,
                  us3myield_df[['Date', 'us3myield']],
                  on='Date', how='outer')
dollar_flow = pd.merge(dollar_flow,
                  fx_df[['Date', 'COP/USD']],
                  on='Date', how='outer')
dollar_flow = dollar_flow.bfill().ffill()

dollar_flow = dollar_flow.dropna()

dollar_flow['COP_WTI'] = dollar_flow['USD_WTI']*dollar_flow['COP/USD']
dollar_flow['COP_Brent'] = dollar_flow['USD_Brent']*dollar_flow['COP/USD']

plt.figure(figsize=(14, 10))

# Subplot 1: USD
plt.subplot(2, 1, 1)
plt.plot(dollar_flow['Date'], dollar_flow['USD_WTI'], label='WTI en USD', color='orange')
plt.plot(dollar_flow['Date'], dollar_flow['USD_Brent'], label='Brent en USD', color='green')
plt.title('Variación Diaria de Ingresos por Petróleo (USD)')
plt.xlabel('Fecha')
plt.ylabel('Dólares (USD)')
plt.legend()
plt.grid(True)

# Subplot 2: COP
plt.subplot(2, 1, 2)
plt.plot(dollar_flow['Date'], dollar_flow['COP_WTI'], label='WTI en COP', color='purple')
plt.plot(dollar_flow['Date'], dollar_flow['COP_Brent'], label='Brent en COP', color='blue')
plt.title('Variación Diaria de Ingresos por Petróleo (COP)')
plt.xlabel('Fecha')
plt.ylabel('Pesos Colombianos (COP)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(ruta_base, "oil_currency_variance.png"), dpi=300)  
plt.show()

#10.000.000 == 10e7

# Correlaciones simples
print("Correlación entre Tasa COL y COP_WTI:", dollar_flow['col3myield'].corr(dollar_flow['COP_WTI']))
print("Correlación entre Tasa USA y COP_WTI:", dollar_flow['us3myield'].corr(dollar_flow['COP_WTI']))

print("Correlación entre Tasa COL y COP_Brent:", dollar_flow['col3myield'].corr(dollar_flow['COP_Brent']))
print("Correlación entre Tasa USA y COP_Brent:", dollar_flow['us3myield'].corr(dollar_flow['COP_Brent']))


#-------------------------------------------------------------------------------------------------
list_load_dollar_estimation = ["Remesas_COL","Direct_Investment_foreign_QoQ_Col","M1_COL",
                               "Reservas_sin_FLAR","Cuenta_Corriente","Balanza_Pagos"]


# Cargamos las bases de datos pendientes
for list_load in list_load_dollar_estimation:
    globals()[list_load] = pd.read_csv(os.path.join(data_bases, ".".join([list_load,"csv"])),sep=";")

# Suponiendo que ya tienes los CSV descargados
remesas = pd.read_csv(os.path.join(data_bases, "Remesas_COL.csv"),sep=";")
ied = pd.read_csv(os.path.join(data_bases, "Direct_Investment_foreign_QoQ_Col.csv"),sep=";")
pesos = pd.read_csv(os.path.join(data_bases, "M1_COL.csv"),sep=";")
reservas = pd.read_csv(os.path.join(data_bases, "Reservas_sin_FLAR.csv"),sep=";")
current_account = pd.read_csv(os.path.join(data_bases, "Cuenta_Corriente.csv"),sep=";")
payment_balance = pd.read_csv(os.path.join(data_bases, "Balanza_Pagos.csv"),sep=";")

remesas = pd.DataFrame(remesas)
remesas.iloc[:,1]

# Unificar las fechas
dollar_flow_estimation = Remesas_COL.merge(Direct_Investment_foreign_QoQ_Col, on='Fecha') \
                .merge(M1_COL, on='Fecha') \
                .merge(Reservas_sin_FLAR, on='Fecha') \
                .merge(Cuenta_Corriente, on='Fecha')\
                .merge(Balanza_Pagos, on='Fecha')\

# Renombrar columnas para claridad
dollar_flow_estimation.columns = ['Date', 'Remesas_USD', 'IED_USD', 'BaseMonetaria_COP', 'Reservas_USD', 'Cuenta_Corriente_USD','Balanza_Pagos']

variables_dollar_estimation = ['Remesas_USD', 'IED_USD', 'BaseMonetaria_COP', 'Reservas_USD', 'Cuenta_Corriente_USD','Balanza_Pagos']
for variable_dollar in variables_dollar_estimation:
    # Eliminar comas, espacios y símbolos extraños antes de convertir
    dollar_flow_estimation[variable_dollar] = (
        dollar_flow_estimation[variable_dollar]
        .astype(str)
        .str.replace(',', '.', regex=False)
    )
    dollar_flow_estimation[variable_dollar] = pd.to_numeric(dollar_flow_estimation[variable_dollar], errors='coerce')
    dollar_flow_estimation[variable_dollar] = dollar_flow_estimation[variable_dollar].round(2)
    print(variable_dollar,'modificada con exito')
    

# Estimar dólares en Colombia: suma de reservas + entradas netas de remesas + IED convertidas
dollar_flow_estimation['Dolares_estimados'] = dollar_flow_estimation['Reservas_USD'] + dollar_flow_estimation['Remesas_USD'] + dollar_flow_estimation['IED_USD']

# --- GRÁFICOS ---

# i. Remesas y IED
plt.figure(figsize=(14, 6))
plt.plot(dollar_flow_estimation['Date'], dollar_flow_estimation['Remesas_USD'], label='Remesas (USD)', color='green')
plt.plot(dollar_flow_estimation['Date'], dollar_flow_estimation['IED_USD'], label='Inversión Extranjera Directa (USD)', color='blue')
plt.title('Remesas y Inversión Extranjera Directa (IED) en Colombia')
plt.ylabel('Millones de USD')
plt.xlabel('Fecha')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(ruta_base, "Remesas_&_IED_olombia.png"), dpi=300)  
plt.show()


# ii. Pesos en circulación
plt.figure(figsize=(14, 6))
plt.plot(dollar_flow_estimation['Date'], dollar_flow_estimation['BaseMonetaria_COP'], label='Base Monetaria (COP)', color='purple')
plt.title('Cantidad de Pesos en Circulación en Colombia')
plt.ylabel('Millones de COP')
plt.xlabel('Fecha')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(ruta_base, "Pesos_Circulación_Colombia.png"), dpi=300)  
plt.show()

# iii. Estimación de dólares
plt.figure(figsize=(14, 6))
plt.plot(dollar_flow_estimation['Date'], dollar_flow_estimation['Dolares_estimados'], label='Estimación de Dólares en Colombia', color='orange')
plt.title('Estimación de USD en Colombia')
plt.ylabel('Millones de USD')
plt.xlabel('Fecha')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(ruta_base, "Estimación_USD_en_Colombia.png"), dpi=300)  
plt.show()

#https://totoro.banrep.gov.co/estadisticas-economicas/faces/pages/charts/line.xhtml?facesRedirect=true ----Reservas internacionales brutas (sin FLAR)

