import streamlit as st
import pandas as pd
import os
import plotly.express as px
import seaborn as sns
import numpy as np

# --- Loading data ---
@st.cache_data
def load_data(folder_path: str) -> pd.DataFrame:
    all_results = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file == "effets_causaux.csv":
                path = os.path.join(root, file)
                combinaison = os.path.basename(root)
                df = pd.read_csv(path)
                df["Combinaison"] = combinaison
                df["Cible_Combo"] = df["Cible"] + " [" + combinaison + "]"
                all_results.append(df)

    if all_results:
        return pd.concat(all_results, ignore_index=True)
    else:
        raise Exception("No data found in the specified folder.")



def render_first_page(folder_results: str):

    df_all = load_data(folder_results)

    # --- Sélections dynamiques ---

    # 1. Combinaisons
    combinaisons = sorted(df_all["Combinaison"].unique())
    combinaisons_sel = st.multiselect(
        "🧩 Choose the combinations of targets to be displayed:",
        options=combinaisons,
        default=combinaisons,
    )

    # 2. Traitements - auto-désélection de ceux à 0
    df_temp = df_all[df_all["Combinaison"].isin(combinaisons_sel)]
    traitement_stats = df_temp.groupby("Traitement")["Effet Causal Moyen (CATE)"].apply(lambda x: (x != 0).any())
    traitements_valides = sorted(traitement_stats[traitement_stats].index.tolist())
    traitements = sorted(df_all["Traitement"].unique())

    traitements_sel = st.multiselect(
        "🔧 Filter by treatments (those with 0 are deselected) :",
        options=traitements,
        default=traitements_valides,
    )

    # Filtrage selon combinaison + traitement
    df_filtre = df_all[
        (df_all["Combinaison"].isin(combinaisons_sel)) &
        (df_all["Traitement"].isin(traitements_sel))
    ]

    # 3. Cibles
    with st.expander("🎯 Sort by target", expanded=False):
        cibles_combo = sorted(df_filtre["Cible_Combo"].unique())
        cibles_sel = st.multiselect(
            "🎯 Sort by target:",
            options=cibles_combo,
            default=cibles_combo,
            label_visibility="collapsed"
        )

    df_filtre = df_filtre[df_filtre["Cible_Combo"].isin(cibles_sel)]
    df_filtre["Effet Causal Moyen (CATE)"] = np.sign(df_filtre["Effet Causal Moyen (CATE)"]) * np.log10(1 + np.abs(df_filtre["Effet Causal Moyen (CATE)"]))
    if df_filtre.empty:
        st.warning("No data to display.")
        return

    # --- Tabs d'affichage ---
    tab1, tab2, tab3 = st.tabs(["📊 Heatmap", "📉 Barplot", "🧩 Clustermap"])

    # --- HEATMAP ---
    with tab1:

        pivot = df_filtre.pivot(index="Cible_Combo", columns="Traitement", values="Effet Causal Moyen (CATE)")
        fig = px.imshow(
            pivot,
            text_auto=".2f",
            color_continuous_scale="RdBu",
            zmin=-pivot.abs().max().max(),
            zmax=pivot.abs().max().max(),
            aspect="auto",
            labels=dict(color="CATE")
        )
        fig.update_layout(title="Conditional Average Treatment Effect (CATE)", xaxis_title="Treatment", yaxis_title="Target + Combinaison")
        st.plotly_chart(fig, use_container_width=True)

    # --- BARPLOT ---
    with tab2:
        fig = px.bar(
            df_filtre,
            x="Traitement",
            y="Effet Causal Moyen (CATE)",
            color="Cible_Combo",
            barmode="group",
            title="CATE by treatment and target",
            labels={"Effet Causal Moyen (CATE)": "Conditional Average Treatment Effect"},
        )
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)

    # --- CLUSTERMAP ---
    with tab3:
        pivot = df_filtre.pivot(index="Cible_Combo", columns="Traitement", values="Effet Causal Moyen (CATE)").fillna(0)
        fig = sns.clustermap(pivot, cmap="coolwarm", center=0, figsize=(14, 12), annot=True, fmt=".2f")
        st.pyplot(fig.figure)

