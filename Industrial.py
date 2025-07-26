import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Cargar dataset procesado
@st.cache_data
def cargar_datos():
    return pd.read_csv("Dataset_Talento_Procesado.csv")

data = cargar_datos()

st.title("🤖 Chatbot Industrial - Ingelean")

# Capturar pregunta del usuario
pregunta = st.chat_input("Haz tu pregunta sobre fallos, eficiencia, energía...")

if pregunta:
    pregunta_lower = pregunta.lower()

    # --- Consulta: Temperatura promedio en fallos
    if "temperatura" in pregunta_lower and "fallo" in pregunta_lower:
        resultado = data.loc[data['fallo_detectado'] == 'Sí', 'temperatura'].mean().round(2)
        st.write(f"🌡️ El promedio de temperatura en registros con fallos fue de {resultado} °C.")

    # --- Consumo energético por máquina
    elif "consumo" in pregunta_lower and "energ" in pregunta_lower:
        energia = data.groupby('maquina_id').agg({
            'consumo_energia': 'sum',
            'cantidad_producida': 'sum'
        }).reset_index()
        energia['energia_por_unidad'] = (
            energia['consumo_energia'] / energia['cantidad_producida']
        ).round(4)
        energia = energia.sort_values('energia_por_unidad', ascending=False)
        st.write("⚡ Consumo energético por unidad producida:")
        st.dataframe(energia)

    # --- Operador con más fallos
    elif "operador" in pregunta_lower and "fallo" in pregunta_lower:
        operador_fallos = data[data['fallo_detectado'] == 'Sí']['operador_id'].value_counts().reset_index()
        operador_fallos.columns = ['operador_id', 'total_fallos']
        top_operador = operador_fallos.iloc[0]
        st.write(f"👷 El operador con más fallos fue {top_operador['operador_id']} con {top_operador['total_fallos']} fallos.")

    # --- Máquinas que requieren más calibración
    elif "calibracion" in pregunta_lower or "calibración" in pregunta_lower:
        calibraciones = data[data['observaciones'].str.upper().str.contains('REVISAR CALIBRACIÓN', na=False)]
        top_maquinas = calibraciones['maquina_id'].value_counts().head(3).reset_index()
        top_maquinas.columns = ['maquina_id', 'calibraciones_requeridas']
        st.write("🔧 Las 3 máquinas con más solicitudes de calibración son:")
        st.dataframe(top_maquinas)

    # --- Eficiencia promedio por turno
    elif "eficiencia" in pregunta_lower and "turno" in pregunta_lower:
        eficiencia_turno = (
            data.groupby('turno')['eficiencia_porcentual']
            .mean().round(2)
            .sort_values(ascending=False)
            .reset_index()
            .rename(columns={'eficiencia_porcentual': 'eficiencia_promedio'})
        )
        st.write("📊 Promedio de eficiencia por jornada:")
        st.dataframe(eficiencia_turno)

    # --- Comparación visual de clústeres (Radar)
    elif "clú" in pregunta_lower and "comparar" in pregunta_lower:
        st.write("🧐 Comparación visual de clústeres en variables clave:")

        variables = [
            'eficiencia_porcentual', 'unidades_defectuosas',
            'cantidad_producida', 'paradas_programadas', 'tiempo_ciclo'
        ]

        if all(var in data.columns for var in variables):
            df_cluster = data[variables].dropna()
            X_scaled = StandardScaler().fit_transform(df_cluster)
            kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(X_scaled)
            data.loc[df_cluster.index, 'cluster'] = clusters

            resumen = data.groupby('cluster')[variables].mean()
            resumen_norm = (resumen - resumen.min()) / (resumen.max() - resumen.min())

            labels = resumen_norm.columns.tolist()
            num_vars = len(labels)
            angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
            angles += angles[:1]

            fig, ax = plt.subplots(figsize=(6,6), subplot_kw=dict(polar=True))
            colores = ['orchid', 'skyblue', 'mediumseagreen']

            for i, row in resumen_norm.iterrows():
                values = row.tolist() + [row.tolist()[0]]
                ax.plot(angles, values, label=f'Clúster {int(i)}', color=colores[int(i)])
                ax.fill(angles, values, alpha=0.25, color=colores[int(i)])

            ax.set_title("Radar de clústeres", size=14)
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(labels)
            ax.set_yticklabels([])
            ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1))
            st.pyplot(fig)

            st.markdown("""
            **Resumen:**
            - 🟣 **Clúster 0**: Alta eficiencia y producción, pero también más defectos.
            - 🔵 **Clúster 1**: Menos errores y pocas paradas, pero tiempos más largos.
            - 🟢 **Clúster 2**: Muchas paradas programadas, baja producción.
            """)
        else:
            st.warning("❗️ El dataset no tiene todas las variables necesarias para generar el radar.")

    # --- Perfil de clústeres (tabla)
    elif "clú" in pregunta_lower or "cluster" in pregunta_lower:
        if 'cluster' not in data.columns:
            st.warning("🔁 Ejecuta primero la sección de clustering en tu código principal para habilitar esta función.")
        else:
            resumen_clusters = data.groupby('cluster')[[
                'fallo_binario', 'temperatura', 'vibración', 'humedad', 'tiempo_ciclo',
                'cantidad_producida', 'unidades_defectuosas', 'eficiencia_porcentual',
                'consumo_energia', 'paradas_programadas', 'paradas_imprevistas'
            ]].mean().round(2)
            st.write("🔍 Perfil promedio por clúster:")
            st.dataframe(resumen_clusters)

    else:
        st.info("❓ No entendí tu pregunta. Intenta con palabras como 'fallo', 'calibración', 'eficiencia', 'turno', 'clúster'...")
