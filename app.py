import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from etfs_info import ETFs_Data
import os
from fpdf import FPDF  # Añadimos la librería FPDF para generar PDF

# Aplicar el estilo de Seaborn a los gráficos
sns.set(style="whitegrid")
sns.set_palette("muted")

# Cargar el logo de Allianz desde un archivo local
if os.path.exists("allianz.svg"):
    st.image("allianz.svg", width=200)
else:
    st.write("Logotipo no disponible")

# Título principal estilizado
st.markdown("<h1 style='text-align: center; color: #003366;'>Simulador de Inversión Allianz</h1>", unsafe_allow_html=True)

# Instrucción amigable
st.markdown("<h3 style='color: #336699;'>Selecciona uno, dos o tres ETFs para comparar su rendimiento y simular la inversión:</h3>", unsafe_allow_html=True)

# Selector de ETF (ahora para seleccionar hasta tres ETFs)
etf_nombres = [etf['nombre'] for etf in ETFs_Data]
seleccion_etfs = st.multiselect('Selecciona uno, dos o tres ETFs para comparar', etf_nombres, default=[etf_nombres[0]])

# Verificar que se hayan seleccionado entre uno y tres ETFs
if 1 <= len(seleccion_etfs) <= 3:
    etf_info_list = []

    # Descargar los datos del S&P 500 para calcular la Beta y Alpha
    sp500 = yf.download('^GSPC', period='10y')

    # Verificar si existe la columna 'Adj Close', si no, usar 'Close'
    if 'Adj Close' in sp500.columns:
        sp500 = sp500['Adj Close']
    elif 'Close' in sp500.columns:
        sp500 = sp500['Close']
    else:
        st.error("No se encontraron datos válidos para el S&P 500. Verifica la conexión o el símbolo.")
        sp500 = None
        
    # Obtener la información de cada ETF seleccionado
    for etf_nombre in seleccion_etfs:
        etf_info = next((etf for etf in ETFs_Data if etf['nombre'] == etf_nombre), None)
        etf_info_list.append(etf_info)
    
    # Mostrar las descripciones de los ETFs seleccionados
    for etf_info in etf_info_list:
        st.markdown(f"<h4 style='color: #003366;'>Descripción de {etf_info['nombre']}:</h4>", unsafe_allow_html=True)
        st.write(f"{etf_info['descripcion']}")

    # Selector de periodos de tiempo
    periodos = ['1mo', '3mo', '6mo', '1y', 'ytd', '5y', '10y']
    seleccion_periodo = st.selectbox('Selecciona el periodo de tiempo', periodos)

    # Entrada de monto inicial de inversión
    monto_inicial = st.number_input("Introduce el monto inicial de inversión ($)", min_value=100.0, value=1000.0)

    # Usar una tasa libre de riesgo predeterminada (por ejemplo, 2%)
    tasa_libre_riesgo = 2.0

    # Barra de progreso para indicar la descarga de datos
    if st.button('Simular inversión y comparar ETFs'):
        st.write("Descargando datos...")
        progress_bar = st.progress(0)

        # Descargar los datos de cada ETF seleccionado
        datos_list = []
        for idx, etf_info in enumerate(etf_info_list):
            datos = yf.download(etf_info['simbolo'], period=seleccion_periodo)

            # Limpiar los nombres de las columnas (quitar espacios en blanco)
            if isinstance(datos.columns, pd.MultiIndex):
                # Aplanar MultiIndex si es necesario
                datos.columns = ['_'.join(col).strip() for col in datos.columns.values]
            else:
                if datos.columns.dtype == 'object':  # Solo aplicar si las columnas son strings
                    datos.columns = datos.columns.str.strip()

            # Verificar si los datos contienen la columna 'Adj Close' o 'Close'
            columna_cierre = 'Adj Close' if 'Adj Close' in datos.columns else 'Close'

            if columna_cierre not in datos.columns:
                st.error(f"No se encontró la columna 'Adj Close' o 'Close' para el ETF {etf_info['nombre']}. Revisa los datos o el periodo seleccionado.")
                continue  # Pasamos al siguiente ETF sin agregar datos
            else:
                datos_list.append((etf_info, datos))
            
            # Actualizar la barra de progreso
            progress_bar.progress((idx + 1) / len(etf_info_list))

        # Si no hay datos válidos, no se continúa con los cálculos
        if not datos_list:
            st.error("No se encontraron datos para los ETFs seleccionados.")
        else:
            # Función para calcular rendimiento, volatilidad, beta, máximo drawdown, y alpha
            def calcular_rendimiento_riesgo(datos, tasa_libre_riesgo, sp500, columna_cierre):
                if len(datos) > 1:  # Verificar si hay suficientes datos
                    # Calcular rendimiento
                    rendimiento = (datos[columna_cierre].iloc[-1] - datos[columna_cierre].iloc[0]) / datos[columna_cierre].iloc[0] * 100
                    
                    # Calcular volatilidad (desviación estándar de los rendimientos diarios)
                    rendimientos_diarios = datos[columna_cierre].pct_change().dropna()
                    volatilidad = rendimientos_diarios.std() * np.sqrt(252)  # Anualizada

                    # Calcular la Beta (comparando con S&P 500)
                    retornos_mercado = sp500.pct_change().dropna()

                    # Alinear los rendimientos diarios del ETF y del mercado
                    datos_alineados = pd.concat([rendimientos_diarios, retornos_mercado], axis=1, join="inner").dropna()

                    # Ahora calcular la covarianza entre los datos alineados
                    beta = np.cov(datos_alineados.iloc[:, 0], datos_alineados.iloc[:, 1])[0, 1] / np.var(datos_alineados.iloc[:, 1])

                    # Calcular el máximo drawdown
                    max_value = datos[columna_cierre].cummax()
                    drawdown = (datos[columna_cierre] - max_value) / max_value
                    max_drawdown = drawdown.min() * 100  # Expresado en %

                    # Calcular el Alpha
                    rendimiento_mercado = (sp500[-1] - sp500[0]) / sp500[0] * 100
                    alpha = rendimiento - (tasa_libre_riesgo + beta * (rendimiento_mercado - tasa_libre_riesgo))

                    return rendimiento, volatilidad, beta, max_drawdown, alpha
                else:
                    st.error(f"Los datos para el ETF {etf_info['nombre']} no son suficientes para calcular el rendimiento.")
                    return None, None, None, None, None
            
            # Calcular resultados para cada ETF
            resultados = []
            for etf_info, datos in datos_list:
                rendimiento, volatilidad, beta, max_drawdown, alpha = calcular_rendimiento_riesgo(datos, tasa_libre_riesgo, sp500, columna_cierre)
                if rendimiento is not None:
                    resultados.append((etf_info['nombre'], rendimiento, volatilidad, beta, max_drawdown, alpha))

            # Mostrar resultados de cada ETF
            for nombre, rendimiento, volatilidad, beta, max_drawdown, alpha in resultados:
                st.markdown(f"<h4 style='color: #336699;'>Resultados para {nombre}:</h4>", unsafe_allow_html=True)
                st.write(f"*Rendimiento*: {rendimiento:.2f}%")
                st.write(f"*Volatilidad*: {volatilidad:.2f}")
                st.write(f"*Beta*: {beta:.2f}")
                st.write(f"*Máximo drawdown*: {max_drawdown:.2f}%")
                st.write(f"*Alpha*: {alpha:.2f}")

            # Simulación de inversión para cada ETF
            for nombre, rendimiento, _, _, _, _ in resultados:
                valor_final = monto_inicial * (1 + rendimiento / 100)
                periodo_texto = seleccion_periodo.replace("mo", "meses").replace("y", "años")
                st.markdown(f"<h4>Si hubieras invertido <strong>${monto_inicial:,.2f}</strong> hace <strong>{periodo_texto}</strong> en <strong>{nombre}</strong>, ahora tendrías <strong>${valor_final:,.2f}</strong>.</h4>", unsafe_allow_html=True)

            # Gráfico comparativo de precios ajustados (Plotly para hacerlo interactivo)
            st.write("*Gráfico comparativo de precios ajustados*")
            fig = px.line()
            for idx, (_, datos) in enumerate(datos_list):
                fig.add_scatter(x=datos.index, y=datos[columna_cierre], mode='lines', name=f'{etf_info_list[idx]["nombre"]}')
            
            fig.update_layout(
                title='Comparación de precios ajustados entre los ETFs seleccionados',
                xaxis_title='Fecha',
                yaxis_title='Precio',
                template="plotly_white"
            )
            st.plotly_chart(fig)

            # Generar PDF con los resultados
            def generar_pdf(resultados):
                pdf = FPDF()
                pdf.add_page()
                pdf.set_font("Arial", size=12)

                pdf.cell(200, 10, txt="Informe de simulación de inversión", ln=True, align="C")

                for nombre, rendimiento, volatilidad, beta, max_drawdown, alpha in resultados:
                    pdf.cell(200, 10, txt=f"ETF: {nombre}", ln=True)
                    pdf.cell(200, 10, txt=f"Rendimiento: {rendimiento:.2f}%", ln=True)
                    pdf.cell(200, 10, txt=f"Volatilidad: {volatilidad:.2f}", ln=True)
                    pdf.cell(200, 10, txt=f"Beta: {beta:.2f}", ln=True)
                    pdf.cell(200, 10, txt=f"Max Drawdown: {max_drawdown:.2f}%", ln=True)
                    pdf.cell(200, 10, txt=f"Alpha: {alpha:.2f}", ln=True)
                    pdf.ln(10)

                return pdf.output(dest="S").encode("latin1")  # Retorna el PDF generado

            # Botón para descargar el PDF
            if st.button('Descargar informe en PDF'):
                pdf_data = generar_pdf(resultados)
                st.download_button(
                    label="Descargar PDF",
                    data=pdf_data,
                    file_name="informe_inversion.pdf",
                    mime="application/pdf"
                )

else:
    st.error("Por favor selecciona entre uno y tres ETFs para realizar la simulación.")



