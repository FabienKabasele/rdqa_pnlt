import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')
import os

# Configuration de la page
st.set_page_config(
    page_title="Tableau de Bord RDQA - PNLT RDC",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# CSS personnalisé
st.markdown("""
<style>
    .main-header {
        font-size: 2.8rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
        padding: 1.5rem;
        background: linear-gradient(135deg, #1E3A8A 0%, #3B82F6 100%);
        border-radius: 15px;
        color: white;
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
    }
    .section-title {
        color: #1E3A8A;
        border-bottom: 4px solid #3B82F6;
        padding-bottom: 0.5rem;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        font-size: 1.8rem;
        font-weight: bold;
    }
    .subsection-title {
        color: #2563EB;
        border-left: 5px solid #3B82F6;
        padding-left: 1rem;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        font-size: 1.5rem;
    }
    .metric-box {
        background: white;
        padding: 1.2rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.08);
        border: 1px solid #E5E7EB;
        text-align: center;
        transition: transform 0.2s;
    }
    .metric-box:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.12);
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: bold;
        color: #1E3A8A;
        margin: 0.5rem 0;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #6B7280;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .highlight-box {
        background: linear-gradient(135deg, #F0F9FF 0%, #E0F2FE 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 6px solid #0EA5E9;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(14, 165, 233, 0.1);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
        background-color: #F3F4F6;
        padding: 0.5rem;
        border-radius: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: white;
        border-radius: 8px;
        color: #6B7280;
        padding: 10px 24px;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background-color: #3B82F6 !important;
        color: white !important;
    }
    .data-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #E5E7EB;
        margin-bottom: 1rem;
    }
    .metric-value-dark {
        font-size: 2.2rem;
        font-weight: bold;
        color: #111827;
        margin: 0.5rem 0;
    }
    .metric-label-dark {
        font-size: 0.9rem;
        color: #4b5563;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        font-weight: 600;
    }
    .cdt-graph-title {
        font-size: 1.2rem;
        font-weight: 600;
        color: #1f2937;
        margin-bottom: 0.5rem;
    }
    .cdt-graph-subtitle {
        font-size: 0.9rem;
        color: #6b7280;
        margin-bottom: 1rem;
    }
    .graph-container {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        margin-bottom: 1.5rem;
        border: 1px solid #e5e7eb;
    }
    .section-divider {
        border-top: 3px solid #3b82f6;
        margin: 2rem 0;
        padding-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# FONCTIONS BCZS
# ============================================================================

def calculate_quality_score(df):
    """Calcule les scores de qualité pour BCZS"""
    if df.empty or len(df) == 0:
        return pd.Series(dtype=float)
    
    score_components = {}
    
    structure_mapping = {
        'q7': 'Personnel rapport BCZS',
        'q8': 'Indicateurs affichés',
        'q9': 'Plan saisie DHIS2',
        'q10': 'Vérification données',
        'q11': 'Validation réunion',
        'q13': 'Rétroinformation supervision'
    }
    
    structure_vars = [col for col in structure_mapping.keys() if col in df.columns]
    if structure_vars:
        def normalize_score(x):
            if pd.isna(x):
                return 0
            elif x == 1:
                return 100
            elif x == 2:
                return 0
            elif x == 3:
                return 50
            else:
                return 0
        
        structure_scores = df[structure_vars].applymap(normalize_score)
        score_components['Structure S&E'] = structure_scores.mean().mean()
    
    indicator_vars = [f'q16_{i}' for i in range(1, 11)]
    existing_indicators = [col for col in indicator_vars if col in df.columns]
    
    if existing_indicators:
        def normalize_knowledge(x):
            if pd.isna(x):
                return 0
            elif x == 1:
                return 100
            elif x == 2:
                return 50
            elif x == 3:
                return 0
            else:
                return 0
        
        knowledge_scores = df[existing_indicators].applymap(normalize_knowledge)
        score_components['Connaissance indicateurs'] = knowledge_scores.mean().mean()
    
    supervision_mapping = {
        'q23': 'Monitorage qualité',
        'q24': 'Vérification données',
        'q26': 'Vérification registres',
        'q27': 'Rapport feedback',
        'q28': 'Politique qualité',
        'q29': 'Monitorage CDT',
        'q30': 'Validation trimestrielle',
        'q31': 'Correction données',
        'q32': 'Tableau de bord',
        'q33': 'Supervision qualité',
        'q34': 'Application recommandations'
    }
    
    supervision_vars = [col for col in supervision_mapping.keys() if col in df.columns]
    if supervision_vars:
        supervision_scores = df[supervision_vars].applymap(lambda x: 100 if x == 1 else (50 if x == 2 else (25 if x == 3 else 0)))
        score_components['Supervision qualité'] = supervision_scores.mean().mean()
    
    archive_mapping = {
        'q35': 'Espace archivage',
        'q36': 'Procédure admin DB',
        'q37': 'Système protégé',
        'q38': 'Canal unique report'
    }
    
    archive_vars = [col for col in archive_mapping.keys() if col in df.columns]
    if archive_vars:
        archive_scores = df[archive_vars].applymap(lambda x: 100 if x == 1 else (75 if x == 2 else (50 if x == 3 else (25 if x == 4 else 0))))
        score_components['Archivage'] = archive_scores.mean().mean()
    
    if score_components:
        score_components['Score Global'] = np.mean(list(score_components.values()))
    
    return pd.Series(score_components)

def create_radar_chart(scores):
    """Crée un radar chart avec Plotly"""
    if scores.empty:
        return None
    
    categories = list(scores.index)
    if 'Score Global' in categories:
        categories.remove('Score Global')
    
    if not categories:
        return None
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=scores[categories].values,
        theta=categories,
        fill='toself',
        name='Performance',
        line=dict(color='rgb(59, 130, 246)', width=3),
        fillcolor='rgba(59, 130, 246, 0.3)'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                tickfont=dict(size=11),
                gridcolor='lightgray',
                linecolor='gray'
            ),
            angularaxis=dict(
                tickfont=dict(size=12),
                gridcolor='lightgray',
                linecolor='gray'
            ),
            bgcolor='rgba(240, 249, 255, 0.3)'
        ),
        showlegend=False,
        height=400,
        margin=dict(l=50, r=50, t=50, b=50),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

def create_knowledge_heatmap(df):
    """Crée une heatmap de connaissance des indicateurs"""
    indicator_labels = {
        'q16_1': 'Cas TB notifiés',
        'q16_2': 'TB pharmaco-sensible',
        'q16_3': 'TB-PS mis en traitement',
        'q16_4': 'TB-PS issue connue',
        'q16_5': 'Cas TB-PR',
        'q16_6': 'TB-PR mis en traitement',
        'q16_7': 'TB-PR issue connue',
        'q16_8': 'Statut VIH documenté',
        'q16_9': 'Enfants <5 sous TPT',
        'q16_10': 'TB orienté communauté'
    }
    
    existing_indicators = {k: v for k, v in indicator_labels.items() if k in df.columns}
    
    if not existing_indicators:
        return None, None
    
    knowledge_data = []
    for var, label in existing_indicators.items():
        if var in df.columns:
            total = df[var].count()
            if total > 0:
                connait_complet = (df[var] == 1).sum()
                connait_partiel = (df[var] == 2).sum()
                
                taux_total = ((connait_complet * 100) + (connait_partiel * 50)) / total
                
                knowledge_data.append({
                    'Indicateur': label,
                    'Connaît complètement': connait_complet,
                    'Connaît partiellement': connait_partiel,
                    'Ne connaît pas': total - connait_complet - connait_partiel,
                    'Score total': taux_total,
                    'Nombre réponses': total
                })
    
    if knowledge_data:
        knowledge_df = pd.DataFrame(knowledge_data)
        knowledge_df = knowledge_df.sort_values('Score total')
        
        fig = px.bar(
            knowledge_df,
            x='Score total',
            y='Indicateur',
            orientation='h',
            color='Score total',
            color_continuous_scale='RdYlGn',
            range_color=[0, 100],
            title='Connaissance des définitions d\'indicateurs TB',
            hover_data=['Connaît complètement', 'Connaît partiellement', 'Ne connaît pas', 'Nombre réponses']
        )
        
        fig.update_layout(
            height=400,
            yaxis_title='',
            xaxis_title='Score de connaissance (%)',
            coloraxis_showscale=False,
            hovermode='y unified'
        )
        
        return fig, knowledge_df
    return None, None

def create_province_analysis(df, province_col='q010b'):
    """Analyse comparative par province"""
    if province_col not in df.columns or df[province_col].nunique() < 2:
        return None, None
    
    province_scores = []
    for province in df[province_col].dropna().unique():
        province_df = df[df[province_col] == province]
        scores = calculate_quality_score(province_df)
        
        if not scores.empty and 'Score Global' in scores:
            province_scores.append({
                'Province': str(province),
                'Score Global': scores['Score Global'],
                'Nombre ZS': len(province_df),
                'Structure S&E': scores.get('Structure S&E', np.nan),
                'Connaissance': scores.get('Connaissance indicateurs', np.nan),
                'Supervision': scores.get('Supervision qualité', np.nan),
                'Archivage': scores.get('Archivage', np.nan)
            })
    
    if province_scores:
        comparison_df = pd.DataFrame(province_scores)
        comparison_df = comparison_df.sort_values('Score Global')
        
        fig = go.Figure()
        
        colors = px.colors.sequential.Viridis
        
        for i, row in comparison_df.iterrows():
            fig.add_trace(go.Bar(
                y=[row['Province']],
                x=[row['Score Global']],
                name=row['Province'],
                orientation='h',
                marker=dict(
                    color=colors[i % len(colors)],
                    line=dict(color='white', width=1)
                ),
                hovertemplate=(
                    f"<b>{row['Province']}</b><br>"
                    f"Score Global: {row['Score Global']:.1f}%<br>"
                    f"ZS: {row['Nombre ZS']}<br>"
                    f"Structure: {row.get('Structure S&E', 'N/A'):.1f}%<br>"
                    f"Connaissance: {row.get('Connaissance', 'N/A'):.1f}%<br>"
                    f"Supervision: {row.get('Supervision', 'N/A'):.1f}%<br>"
                    f"Archivage: {row.get('Archivage', 'N/A'):.1f}%"
                )
            ))
        
        fig.update_layout(
            title='Score Global par Province',
            height=max(300, len(comparison_df) * 30),
            xaxis_title='Score Global (%)',
            yaxis_title='',
            showlegend=False,
            bargap=0.2
        )
        
        return fig, comparison_df
    return None, None

def create_training_analysis(df):
    """Analyse des formations du personnel"""
    training_mapping = {
        'q12_1': 'PATI 6',
        'q12_2': 'DHIS2',
        'q12_3': 'Suivi & Évaluation',
        'q12_4': 'Gestion des données'
    }
    
    existing_training = {k: v for k, v in training_mapping.items() if k in df.columns}
    
    if not existing_training:
        return None, None
    
    training_data = []
    for var, label in existing_training.items():
        total = df[var].count()
        if total > 0:
            forme = (df[var] == 1).sum()
            taux_formation = (forme / total * 100)
            
            training_data.append({
                'Formation': label,
                'Formés': forme,
                'Non formés': total - forme,
                'Total': total,
                'Taux formation (%)': taux_formation
            })
    
    if training_data:
        training_df = pd.DataFrame(training_data)
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Personnel formé par domaine', 'Taux de formation'),
            specs=[[{"type": "bar"}, {"type": "bar"}]],
            horizontal_spacing=0.15
        )
        
        fig.add_trace(
            go.Bar(
                name='Formés',
                x=training_df['Formation'],
                y=training_df['Formés'],
                marker_color='#10B981',
                text=training_df['Formés'],
                textposition='auto'
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Bar(
                name='Non formés',
                x=training_df['Formation'],
                y=training_df['Non formés'],
                marker_color='#EF4444',
                text=training_df['Non formés'],
                textposition='auto'
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Bar(
                x=training_df['Formation'],
                y=training_df['Taux formation (%)'],
                marker=dict(
                    color=training_df['Taux formation (%)'],
                    colorscale='RdYlGn',
                    cmin=0,
                    cmax=100
                ),
                text=training_df['Taux formation (%)'].round(1).astype(str) + '%',
                textposition='auto'
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            height=350,
            showlegend=True,
            barmode='stack',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        fig.update_xaxes(tickangle=45, row=1, col=1)
        fig.update_xaxes(tickangle=45, row=1, col=2)
        
        return fig, training_df
    return None, None

# ============================================================================
# FONCTIONS CDT
# ============================================================================

def calculate_cdt_quality_score(df_cdt):
    """Calcule les scores de qualité pour CDT"""
    if df_cdt.empty or len(df_cdt) == 0:
        return pd.Series(dtype=float)
    
    score_components = {}
    
    structure_mapping = {
        'q100': 'Responsabilité enregistrement',
        'q101': 'Personnel formé',
        'q102': 'Superviseur désigné'
    }
    
    structure_vars = [col for col in structure_mapping.keys() if col in df_cdt.columns]
    if structure_vars:
        def normalize_structure(x):
            if pd.isna(x):
                return 0
            elif x in [1, 2]:
                return 100
            elif x == 0:
                return 0
            else:
                return 50
        
        structure_scores = df_cdt[structure_vars].applymap(normalize_structure)
        score_components['Structure S&E CDT'] = structure_scores.mean().mean()
    
    doc_vars = ['q103', 'q104']
    existing_docs = [col for col in doc_vars if col in df_cdt.columns]
    if existing_docs:
        doc_scores = df_cdt[existing_docs].applymap(lambda x: 100 if x == 1 else (50 if x == 2 else 0))
        score_components['Documentation disponible'] = doc_scores.mean().mean()
    
    indicator_vars = [f'q105_{i}' for i in range(1, 11)]
    existing_indicators = [col for col in indicator_vars if col in df_cdt.columns]
    if existing_indicators:
        indicator_scores = df_cdt[existing_indicators].applymap(
            lambda x: 100 if x == 1 else (50 if x == 2 else 0)
        )
        score_components['Connaissance indicateurs CDT'] = indicator_scores.mean().mean()
    
    directive_vars = [f'q106_{i}' for i in range(1, 5)]
    existing_directives = [col for col in directive_vars if col in df_cdt.columns]
    if existing_directives:
        directive_scores = df_cdt[existing_directives].applymap(
            lambda x: 100 if x == 1 else (67 if x == 2 else (33 if x == 3 else 0))
        )
        score_components['Directives rapportage'] = directive_scores.mean().mean()
    
    if 'q107' in df_cdt.columns:
        instruction_score = df_cdt['q107'].apply(
            lambda x: 100 if x == 1 else (67 if x == 2 else (33 if x == 3 else 0))
        ).mean()
        score_components['Instructions claires'] = instruction_score
    
    doc_source_vars = [
        'q108_1', 'q108_2', 'q108_4', 'q108_5', 
        'q108_6', 'q108_7', 'q108_8', 'q108_10', 'q108_13'
    ]
    existing_sources = [col for col in doc_source_vars if col in df_cdt.columns]
    if existing_sources:
        available_docs = df_cdt[existing_sources].applymap(lambda x: 1 if x in [1, 2] else 0).sum(axis=1)
        total_docs_expected = df_cdt[existing_sources].applymap(lambda x: 0 if x == 4 else 1).sum(axis=1)
        
        mask = total_docs_expected > 0
        if mask.any():
            doc_availability = (available_docs[mask] / total_docs_expected[mask] * 100).mean()
            score_components['Documents sources disponibles'] = doc_availability
    
    quality_vars = ['q112', 'q113', 'q114', 'q115', 'q116', 'q117', 'q118', 'q119']
    existing_quality = [col for col in quality_vars if col in df_cdt.columns]
    if existing_quality:
        def normalize_quality(x):
            if pd.isna(x):
                return 0
            elif x == 1:
                return 100
            elif x == 2:
                return 50
            elif x in [0, 3, 4]:
                return 0
            else:
                return 0
        
        quality_scores = df_cdt[existing_quality].applymap(normalize_quality)
        score_components['Supervision qualité CDT'] = quality_scores.mean().mean()
    
    archive_vars = ['q120', 'q121', 'q122', 'q123']
    existing_archive = [col for col in archive_vars if col in df_cdt.columns]
    if existing_archive:
        archive_scores = df_cdt[existing_archive].applymap(lambda x: 100 if x == 1 else 0)
        score_components['Archivage CDT'] = archive_scores.mean().mean()
    
    if 'q109' in df_cdt.columns:
        reporting_rate = (df_cdt['q109'] == 1).mean() * 100
        score_components['Rapportage régulier'] = reporting_rate
    
    if score_components:
        score_components['Score Global CDT'] = np.mean(list(score_components.values()))
    
    return pd.Series(score_components)

def create_cdt_document_analysis(df_cdt):
    """Analyse de la disponibilité des documents sources"""
    document_labels = {
        'q108_1': 'Registre tuberculose',
        'q108_2': 'Registre TB pharmacorésistante',
        'q108_4': 'Registre laboratoire',
        'q108_5': 'Registre expéditions échantillons',
        'q108_6': 'Registre Xpert TB',
        'q108_7': 'Registre cas contacts',
        'q108_8': 'Registre traitement préventif',
        'q108_10': 'Fiches traitement patient',
        'q108_13': 'Rapport mensuel'
    }
    
    existing_docs = {k: v for k, v in document_labels.items() if k in df_cdt.columns}
    
    if not existing_docs:
        return None, None
    
    doc_data = []
    for var, label in existing_docs.items():
        total = df_cdt[var].count()
        if total > 0:
            disponible = ((df_cdt[var] == 1) | (df_cdt[var] == 2)).sum()
            non_applicable = (df_cdt[var] == 4).sum()
            non_disponible = total - disponible - non_applicable
            
            taux_disponible = (disponible / (total - non_applicable) * 100) if (total - non_applicable) > 0 else 0
            
            doc_data.append({
                'Document': label,
                'Disponible': disponible,
                'Non disponible': non_disponible,
                'Non applicable': non_applicable,
                'Total': total,
                'Taux disponibilité (%)': taux_disponible
            })
    
    if doc_data:
        doc_df = pd.DataFrame(doc_data)
        doc_df = doc_df.sort_values('Taux disponibilité (%)')
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            y=doc_df['Document'],
            x=doc_df['Disponible'],
            name='Disponible',
            orientation='h',
            marker_color='#10B981',
            text=doc_df['Disponible'],
            textposition='auto'
        ))
        
        fig.add_trace(go.Bar(
            y=doc_df['Document'],
            x=doc_df['Non disponible'],
            name='Non disponible',
            orientation='h',
            marker_color='#EF4444',
            text=doc_df['Non disponible'],
            textposition='auto'
        ))
        
        fig.add_trace(go.Bar(
            y=doc_df['Document'],
            x=doc_df['Non applicable'],
            name='Non applicable',
            orientation='h',
            marker_color='#6B7280',
            text=doc_df['Non applicable'],
            textposition='auto'
        ))
        
        fig.update_layout(
            title='Disponibilité des documents sources dans les CDT',
            height=max(400, len(doc_df) * 30),
            xaxis_title='Nombre de CDT',
            yaxis_title='',
            barmode='stack',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        return fig, doc_df
    return None, None

def create_cdt_geographical_analysis(df_cdt, province_col='q010b', healthzone_col='q011b'):
    """Analyse des CDT par province et zone de santé"""
    if province_col not in df_cdt.columns:
        return None, None, None
    
    province_stats = []
    for province in df_cdt[province_col].dropna().unique():
        province_df = df_cdt[df_cdt[province_col] == province]
        scores = calculate_cdt_quality_score(province_df)
        
        if not scores.empty and 'Score Global CDT' in scores:
            province_stats.append({
                'Province': str(province),
                'Score Global': scores['Score Global CDT'],
                'Nombre CDT': len(province_df),
                'Documents disponibles': scores.get('Documents sources disponibles', 0),
                'Connaissance indicateurs': scores.get('Connaissance indicateurs CDT', 0),
                'Rapportage régulier': scores.get('Rapportage régulier', 0)
            })
    
    province_df = pd.DataFrame(province_stats) if province_stats else pd.DataFrame()
    
    zone_df = pd.DataFrame()
    if healthzone_col in df_cdt.columns:
        zone_stats = []
        for zone in df_cdt[healthzone_col].dropna().unique():
            zone_data = df_cdt[df_cdt[healthzone_col] == zone]
            zone_stats.append({
                'Zone de Santé': str(zone),
                'Nombre CDT': len(zone_data),
                'Province': zone_data[province_col].iloc[0] if province_col in zone_data.columns else 'Inconnu'
            })
        
        zone_df = pd.DataFrame(zone_stats) if zone_stats else pd.DataFrame()
    
    fig = None
    if not province_df.empty:
        province_df = province_df.sort_values('Score Global')
        
        fig = go.Figure()
        
        colors = px.colors.sequential.Plasma
        
        for i, row in province_df.iterrows():
            fig.add_trace(go.Bar(
                x=[row['Score Global']],
                y=[row['Province']],
                orientation='h',
                name=row['Province'],
                marker=dict(
                    color=colors[i % len(colors)],
                    line=dict(color='white', width=1)
                ),
                hovertemplate=(
                    f"<b>{row['Province']}</b><br>"
                    f"Score: {row['Score Global']:.1f}%<br>"
                    f"CDT: {row['Nombre CDT']}<br>"
                    f"Documents: {row.get('Documents disponibles', 0):.1f}%<br>"
                    f"Connaissance: {row.get('Connaissance indicateurs', 0):.1f}%<br>"
                    f"Rapportage: {row.get('Rapportage régulier', 0):.1f}%"
                )
            ))
        
        fig.update_layout(
            title='Performance des CDT par Province',
            height=max(300, len(province_df) * 30),
            xaxis_title='Score Global (%)',
            yaxis_title='',
            showlegend=False
        )
    
    return fig, province_df, zone_df

def create_cdt_training_analysis(df_cdt):
    """Analyse de la formation du personnel CDT"""
    if 'q101' not in df_cdt.columns:
        return None, None
    
    training_data = []
    
    total = df_cdt['q101'].count()
    if total > 0:
        bien_forme = (df_cdt['q101'] == 1).sum()
        anciennement_forme = (df_cdt['q101'] == 2).sum()
        partiellement_forme = (df_cdt['q101'] == 3).sum()
        pas_forme = (df_cdt['q101'] == 0).sum()
        
        taux_bien_forme = (bien_forme / total * 100)
        taux_forme_total = ((bien_forme + anciennement_forme + 0.5 * partiellement_forme) / total * 100)
        
        training_data.append({
            'Catégorie': 'Bien formé',
            'Nombre': bien_forme,
            'Pourcentage': taux_bien_forme
        })
        
        training_data.append({
            'Catégorie': 'Anciennement formé',
            'Nombre': anciennement_forme,
            'Pourcentage': (anciennement_forme / total * 100)
        })
        
        training_data.append({
            'Catégorie': 'Partiellement formé',
            'Nombre': partiellement_forme,
            'Pourcentage': (partiellement_forme / total * 100)
        })
        
        training_data.append({
            'Catégorie': 'Pas formé',
            'Nombre': pas_forme,
            'Pourcentage': (pas_forme / total * 100)
        })
    
    if training_data:
        training_df = pd.DataFrame(training_data)
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Formation du personnel CDT', 'Taux de formation global'),
            specs=[[{"type": "pie"}, {"type": "indicator"}]],
            column_widths=[0.7, 0.3]
        )
        
        fig.add_trace(
            go.Pie(
                labels=training_df['Catégorie'],
                values=training_df['Nombre'],
                hole=0.4,
                marker_colors=['#10B981', '#3B82F6', '#F59E0B', '#EF4444'],
                textinfo='label+percent',
                hoverinfo='label+value+percent'
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=taux_forme_total,
                title={'text': "Taux formation"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#3B82F6"},
                    'steps': [
                        {'range': [0, 50], 'color': "#EF4444"},
                        {'range': [50, 75], 'color': "#F59E0B"},
                        {'range': [75, 100], 'color': "#10B981"}
                    ]
                }
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            height=400,
            showlegend=False
        )
        
        return fig, training_df
    return None, None

def create_cdt_reporting_analysis(df_cdt):
    """Analyse du rapportage des CDT"""
    if 'q109' not in df_cdt.columns:
        return None, None
    
    reporting_data = []
    
    total = df_cdt['q109'].count()
    if total > 0:
        rapporte = (df_cdt['q109'] == 1).sum()
        ne_rapporte_pas = total - rapporte
        
        reporting_data.append({
            'Statut': 'Rapporte régulièrement',
            'Nombre': rapporte,
            'Pourcentage': (rapporte / total * 100)
        })
        
        reporting_data.append({
            'Statut': 'Ne rapporte pas régulièrement',
            'Nombre': ne_rapporte_pas,
            'Pourcentage': (ne_rapporte_pas / total * 100)
        })
    
    means_data = []
    if 'q110' in df_cdt.columns:
        for means in [1, 2, 3]:
            count = (df_cdt['q110'] == means).sum()
            if count > 0:
                label = {1: 'Papier', 2: 'Électronique', 3: 'Ne sait pas'}[means]
                means_data.append({
                    'Moyen': label,
                    'Nombre': count,
                    'Pourcentage': (count / total * 100)
                })
    
    if reporting_data:
        reporting_df = pd.DataFrame(reporting_data)
        means_df = pd.DataFrame(means_data) if means_data else pd.DataFrame()
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Fréquence de rapportage', 'Moyens de rapportage'),
            specs=[[{"type": "bar"}, {"type": "bar"}]]
        )
        
        fig.add_trace(
            go.Bar(
                x=reporting_df['Statut'],
                y=reporting_df['Pourcentage'],
                marker_color=['#10B981', '#EF4444'],
                text=reporting_df['Pourcentage'].round(1).astype(str) + '%',
                textposition='auto'
            ),
            row=1, col=1
        )
        
        if not means_df.empty:
            fig.add_trace(
                go.Bar(
                    x=means_df['Moyen'],
                    y=means_df['Pourcentage'],
                    marker_color=px.colors.qualitative.Set3,
                    text=means_df['Pourcentage'].round(1).astype(str) + '%',
                    textposition='auto'
                ),
                row=1, col=2
            )
        
        fig.update_layout(
            height=400,
            showlegend=False
        )
        
        fig.update_yaxes(range=[0, 100], row=1, col=1)
        fig.update_yaxes(range=[0, 100], row=1, col=2)
        
        return fig, reporting_df, means_df
    return None, None, None

# ============================================================================
# NOUVELLES FONCTIONS POUR GRAPHIQUES GROUPÉS
# ============================================================================

def create_grouped_bar_chart(df_cdt, questions_dict, title, color_scheme='blues'):
    """Crée un graphique groupé avec toutes les questions d'une catégorie"""
    data = []
    
    for q_code, q_label in questions_dict.items():
        if q_code in df_cdt.columns:
            total = df_cdt[q_code].count()
            if total > 0:
                # Calculer le pourcentage de "Oui" (réponse 1)
                oui_count = (df_cdt[q_code] == 1).sum()
                oui_pct = (oui_count / total * 100) if total > 0 else 0
                
                # Calculer le pourcentage de "Partiellement" (réponse 2)
                partiel_count = (df_cdt[q_code] == 2).sum()
                partiel_pct = (partiel_count / total * 100) if total > 0 else 0
                
                # Score composite (100% pour Oui, 50% pour Partiellement)
                composite_score = oui_pct + (partiel_pct * 0.5)
                
                data.append({
                    'Question': q_label,
                    'Question Code': q_code,
                    'Oui (%)': oui_pct,
                    'Partiellement (%)': partiel_pct,
                    'Non (%)': 100 - oui_pct - partiel_pct,
                    'Score composite': composite_score,
                    'Total répondants': total,
                    'Oui (n)': oui_count,
                    'Partiellement (n)': partiel_count,
                    'Non (n)': total - oui_count - partiel_count
                })
    
    if not data:
        return None
    
    df = pd.DataFrame(data)
    df = df.sort_values('Score composite')
    
    # Choisir la palette de couleurs
    if color_scheme == 'greens':
        colors = ['#10B981', '#34D399', '#059669']
    elif color_scheme == 'reds':
        colors = ['#EF4444', '#F87171', '#DC2626']
    elif color_scheme == 'oranges':
        colors = ['#F59E0B', '#FBBF24', '#D97706']
    else:  # blues par défaut
        colors = ['#3B82F6', '#60A5FA', '#2563EB']
    
    # Créer le graphique
    fig = go.Figure()
    
    # Barre pour Oui
    fig.add_trace(go.Bar(
        name='Oui',
        y=df['Question'],
        x=df['Oui (%)'],
        orientation='h',
        marker_color=colors[0],
        text=[f"{row['Oui (%)']:.1f}% (n={row['Oui (n)']})" for _, row in df.iterrows()],
        textposition='inside',
        textfont=dict(color='white', size=10),
        hovertemplate="<b>%{y}</b><br>Oui: %{x:.1f}%<br>Nombre: %{customdata[0]}<extra></extra>",
        customdata=df[['Oui (n)']].values
    ))
    
    # Barre pour Partiellement
    fig.add_trace(go.Bar(
        name='Partiellement',
        y=df['Question'],
        x=df['Partiellement (%)'],
        orientation='h',
        marker_color=colors[1],
        text=[f"{row['Partiellement (%)']:.1f}% (n={row['Partiellement (n)']})" for _, row in df.iterrows()],
        textposition='inside',
        textfont=dict(color='white', size=10),
        hovertemplate="<b>%{y}</b><br>Partiellement: %{x:.1f}%<br>Nombre: %{customdata[0]}<extra></extra>",
        customdata=df[['Partiellement (n)']].values
    ))
    
    # Barre pour Non
    fig.add_trace(go.Bar(
        name='Non',
        y=df['Question'],
        x=df['Non (%)'],
        orientation='h',
        marker_color=colors[2],
        text=[f"{row['Non (%)']:.1f}% (n={row['Non (n)']})" for _, row in df.iterrows()],
        textposition='inside',
        textfont=dict(color='white', size=10),
        hovertemplate="<b>%{y}</b><br>Non: %{x:.1f}%<br>Nombre: %{customdata[0]}<extra></extra>",
        customdata=df[['Non (n)']].values
    ))
    
    # Ajouter le score composite comme annotation à droite
    annotations = []
    for i, row in df.iterrows():
        annotations.append(dict(
            x=105,
            y=row['Question'],
            text=f"{row['Score composite']:.1f}%",
            showarrow=False,
            xanchor='left',
            font=dict(size=11, color='#1f2937')
        ))
    
    fig.update_layout(
        title=dict(
            text=f"<b>{title}</b><br><span style='font-size:0.8em;color:#6B7280'>Total CDT: {len(df_cdt)}</span>",
            x=0.5,
            xanchor='center'
        ),
        height=max(400, len(df) * 40),
        barmode='stack',
        xaxis_title="Pourcentage des répondants (%)",
        yaxis_title="",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(t=100, l=150, r=150, b=50),
        annotations=annotations,
        xaxis=dict(range=[0, 110])
    )
    
    return fig

def create_grouped_bar_chart(df_cdt, questions_dict, title, color_scheme='blues'):
    """Crée un graphique groupé avec toutes les questions d'une catégorie"""
    # ... le code complet de la fonction ...

def create_cdt_structure_chart(df_cdt):
    """Graphique unique pour Structure et fonction du S&E"""
    structure_questions = {
        'q100': 'Q100: Responsabilité clairement attribuée',
        'q101': 'Q101: Personnel formé approprié',
        'q102': 'Q102: Superviseur désigné'
    }
    return create_grouped_bar_chart(df_cdt, structure_questions, "Structure et fonction du Suivi & Évaluation", 'blues')

def create_cdt_knowledge_chart(df_cdt):
    """Graphique unique pour Connaissance des indicateurs"""
    knowledge_questions = {
        'q105_1': 'Q105.1: Cas TB notifiés',
        'q105_2': 'Q105.2: TB pharmacosensible',
        'q105_3': 'Q105.3: TB-PS en traitement',
        'q105_4': 'Q105.4: Cas TB-PR',
        'q105_5': 'Q105.5: TB-PR en traitement',
        'q105_6': 'Q105.6: TB-PS issue connue',
        'q105_7': 'Q105.7: TB-PR issue connue',
        'q105_8': 'Q105.8: Statut VIH documenté',
        'q105_9': 'Q105.9: Enfants <5 sous TPT',
        'q105_10': 'Q105.10: TB orienté communauté'
    }
    return create_grouped_bar_chart(df_cdt, knowledge_questions, "Connaissance des définitions d'indicateurs", 'greens')

def create_cdt_quality_chart(df_cdt):
    """Graphique unique pour Qualité des données"""
    quality_questions = {
        'q112': 'Q112: Processus vérification qualité',
        'q113': 'Q113: Contrôles d\'exactitude réguliers',
        'q114': 'Q114: Contrôles de cohérence',
        'q115': 'Q115: Vérification complétude registres',
        'q116': 'Q116: Documentation résultats contrôles',
        'q117': 'Q117: Politique écrite qualité données',
        'q118': 'Q118: Visites supervision régulières',
        'q119': 'Q119: Visite supervision (6 derniers mois)'
    }
    return create_grouped_bar_chart(df_cdt, quality_questions, "Qualité des données et supervision", 'oranges')

def create_cdt_archive_chart(df_cdt):
    """Graphique unique pour Archivage et confidentialité"""
    archive_questions = {
        'q120': 'Q120: Registres organisés pour recherche',
        'q121': 'Q121: Espace adéquat stockage',
        'q122': 'Q122: Accès limité registres',
        'q123': 'Q123: Données personnelles sécurisées'
    }
    return create_grouped_bar_chart(df_cdt, archive_questions, "Archivage des données et confidentialité", 'reds')
    
    return create_grouped_bar_chart(df_cdt, document_questions, "Disponibilité des documents sources", 'purples')

# ============================================================================


    # TAB 4: GRAPHIQUES DÉTAILLÉS CDT (4 GRAPHIQUES SEULEMENT)

    with tab4:
        st.markdown('<h2 class="section-title">📊 ANALYSE DÉTAILLÉE PAR CATÉGORIE - CDT</h2>', unsafe_allow_html=True)
        
        if not df_cdt.empty:
            # Résumé statistique
            st.markdown("#### 📈 RÉSUMÉ STATISTIQUE")
            col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
            
            with col_stat1:
                total_cdt = len(df_cdt)
                st.metric("Total CDT analysés", total_cdt)
            
            with col_stat2:
                response_rates = {}
                for col in df_cdt.columns:
                    if col.startswith('q'):
                        total = len(df_cdt)
                        answered = df_cdt[col].count()
                        response_rates[col] = answered / total * 100 if total > 0 else 0
                
                avg_response_rate = np.mean(list(response_rates.values())) if response_rates else 0
                st.metric("Taux réponse moyen", f"{avg_response_rate:.1f}%")
            
            with col_stat3:
                question_cols = [col for col in df_cdt.columns if col.startswith('q')]
                st.metric("Questions disponibles", len(question_cols))
            
            with col_stat4:
                cdt_scores = calculate_cdt_quality_score(df_cdt)
                if 'Score Global CDT' in cdt_scores:
                    st.metric("Score global CDT", f"{cdt_scores['Score Global CDT']:.1f}%")
            
                       # Graphique 1: Structure et fonction du S&E
            st.markdown("### 📋 STRUCTURE ET FONCTION DU SUIVI & ÉVALUATION")
            structure_fig = create_cdt_structure_chart(df_cdt)
            if structure_fig:
                st.plotly_chart(structure_fig, use_container_width=True)
                st.markdown("*Note: Score composite = 100% pour 'Oui' + 50% pour 'Partiellement'*")
            else:
                st.info("Aucune donnée disponible pour la structure S&E")
            
            # Graphique 2: Connaissance des indicateurs
            st.markdown("---")
            st.markdown("### 🎯 CONNAISSANCE DES DÉFINITIONS D'INDICATEURS")
            knowledge_fig = create_cdt_knowledge_chart(df_cdt)
            if knowledge_fig:
                st.plotly_chart(knowledge_fig, use_container_width=True)
                st.markdown("*Note: Score composite = 100% pour 'Oui' + 50% pour 'Partiellement'*")
            else:
                st.info("Aucune donnée disponible pour la connaissance des indicateurs")
            
            # Graphique 3: Qualité des données
            st.markdown("---")
            st.markdown("### ✅ QUALITÉ DES DONNÉES ET SUPERVISION")
            quality_fig = create_cdt_quality_chart(df_cdt)
            if quality_fig:
                st.plotly_chart(quality_fig, use_container_width=True)
                st.markdown("*Note: Score composite = 100% pour 'Oui' + 50% pour 'Partiellement'*")
            else:
                st.info("Aucune donnée disponible pour la qualité des données")
            
            # Graphique 4: Archivage et confidentialité
            st.markdown("---")
            st.markdown("### 🔐 ARCHIVAGE ET CONFIDENTIALITÉ")
            archive_fig = create_cdt_archive_chart(df_cdt)
            if archive_fig:
                st.plotly_chart(archive_fig, use_container_width=True)
                st.markdown("*Note: Score composite = 100% pour 'Oui' + 50% pour 'Partiellement'*")
            else:
                st.info("Aucune donnée disponible pour l'archivage")
            # Téléchargement des données
            st.markdown("---")
            st.markdown("### 💾 EXPORT DES DONNÉES")
            
            col_exp1, col_exp2 = st.columns(2)
            
            with col_exp1:
                if not df_cdt.empty:
                    csv = df_cdt.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Télécharger données brutes CDT",
                        data=csv,
                        file_name="cdt_donnees_brutes.csv",
                        mime="text/csv"
                    )
            
            with col_exp2:
                st.markdown("**📊 Statistiques rapides:**")
                st.markdown(f"- **CDT analysés:** {len(df_cdt)}")
                st.markdown(f"- **Questions disponibles:** {len([col for col in df_cdt.columns if col.startswith('q')])}")
                st.markdown(f"- **Données manquantes:** {df_cdt.isnull().sum().sum()} cellules")
        
        else:
            st.markdown("""
            <div class="highlight-box">
                <h4>📊 CHARGEMENT DES DONNÉES REQUIS</h4>
                <p>Pour afficher les graphiques détaillés par catégorie :</p>
                <ol>
                    <li>Chargez les données CDT dans l'onglet <strong>CDT - Synthèse</strong></li>
                    <li>Assurez-vous que les colonnes Q100 à Q123 sont présentes</li>
                    <li>Les données doivent être au format numérique (1, 2, 3, etc.)</li>
                </ol>
                <p><strong>Format attendu:</strong> CSV avec colonnes q100, q101, q102, etc.</p>
            </div>
            """, unsafe_allow_html=True)


# ============================================================================
# FONCTIONS DE CHARGEMENT
# ============================================================================

def load_data():
    """Charge les données BCZS et CDT"""
    # Charger les données BCZS
    try:
        bczs_path = "F:\\PNUD 2025\\Analyses\\drc_pnlt_rdqa_bczs_WIDE.csv"
        if os.path.exists(bczs_path):
            df_bczs = pd.read_csv(bczs_path, encoding='utf-8')
            st.session_state.df_bczs = df_bczs
        else:
            df_bczs = pd.DataFrame()
    except Exception as e:
        st.error(f"❌ Erreur de chargement BCZS: {str(e)}")
        df_bczs = pd.DataFrame()
    
    # Charger les données CDT
    try:
        cdt_paths = [
            "F:\\PNUD 2025\\Analyses\\drc_pnlt_rdqa_cdt_WIDE.csv",
            "drc_pnlt_rdqa_cdt_WIDE.csv",
            "./drc_pnlt_rdqa_cdt_WIDE.csv"
        ]
        
        df_cdt = pd.DataFrame()
        
        for path in cdt_paths:
            if os.path.exists(path):
                try:
                    df_cdt = pd.read_csv(path, encoding='utf-8')
                    break
                except:
                    continue
        
        if df_cdt.empty:
            df_cdt = pd.DataFrame()
        else:
            st.session_state.df_cdt = df_cdt
            
    except Exception as e:
        df_cdt = pd.DataFrame()
    
    return df_bczs, df_cdt

# ============================================================================
# INTERFACE PRINCIPALE
# ============================================================================

def main():
    # En-tête principal
    col_logo, col_title = st.columns([1, 4])
    
    with col_logo:
        st.image("https://via.placeholder.com/100x100/1E3A8A/FFFFFF?text=PNLT", width=100)
    
    with col_title:
        st.markdown('<h1 class="main-header">📊 TABLEAU DE BORD RDQA - PNLT RDC</h1>', unsafe_allow_html=True)
    
    # Onglets principaux
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🏥 BCZS - Synthèse", 
        "📈 BCZS - Analyse détaillée", 
        "🏘️ CDT - Synthèse",
        "📊 CDT - Graphiques détaillés",  
        "🔗 Comparaison multi-niveaux"
    ])
    
    # Charger les données
    if 'df_bczs' not in st.session_state:
        with st.spinner('Chargement des données...'):
            df_bczs, df_cdt = load_data()
    else:
        df_bczs = st.session_state.df_bczs
        df_cdt = st.session_state.get('df_cdt', pd.DataFrame())
    
    # TAB 1: SYNTHÈSE BCZS
    with tab1:
        if not df_bczs.empty:
            st.markdown('<h2 class="section-title">📈 SYNTHÈSE GLOBALE BCZS</h2>', unsafe_allow_html=True)
            
            scores = calculate_quality_score(df_bczs)
            
            if not scores.empty:
                st.markdown("#### 🎯 INDICATEURS CLÉS DE PERFORMANCE")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    if 'Score Global' in scores:
                        st.markdown(f"""
                        <div class="metric-box">
                            <div class="metric-label">Score Global</div>
                            <div class="metric-value">{scores['Score Global']:.1f}%</div>
                            <div style="background: #E5E7EB; height: 8px; border-radius: 4px; margin-top: 10px;">
                                <div style="background: #3B82F6; width: {scores['Score Global']}%; height: 100%; border-radius: 4px;"></div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                
                with col2:
                    if 'Connaissance indicateurs' in scores:
                        st.markdown(f"""
                        <div class="metric-box">
                            <div class="metric-label">Connaissance indicateurs</div>
                            <div class="metric-value">{scores['Connaissance indicateurs']:.1f}%</div>
                        </div>
                        """, unsafe_allow_html=True)
                
                with col3:
                    if 'Structure S&E' in scores:
                        st.markdown(f"""
                        <div class="metric-box">
                            <div class="metric-label">Structure S&E</div>
                            <div class="metric-value">{scores['Structure S&E']:.1f}%</div>
                        </div>
                        """, unsafe_allow_html=True)
                
                with col4:
                    if 'Supervision qualité' in scores:
                        st.markdown(f"""
                        <div class="metric-box">
                            <div class="metric-label">Supervision qualité</div>
                            <div class="metric-value">{scores['Supervision qualité']:.1f}%</div>
                        </div>
                        """, unsafe_allow_html=True)
                
                st.markdown("#### 📊 PROFIL DE PERFORMANCE")
                radar_col1, radar_col2 = st.columns([2, 1])
                
                with radar_col1:
                    radar_fig = create_radar_chart(scores)
                    if radar_fig:
                        st.plotly_chart(radar_fig, use_container_width=True)
                
                with radar_col2:
                    st.markdown("#### 📋 DÉTAIL DES SCORES")
                    for component, score in scores.items():
                        if component != 'Score Global':
                            st.markdown(f"""
                            <div class="data-card">
                                <strong>{component}</strong><br>
                                <span style="font-size: 1.5rem; color: #1E3A8A;">{score:.1f}%</span>
                            </div>
                            """, unsafe_allow_html=True)
                
                st.markdown("#### 🗺️ PERFORMANCE PAR PROVINCE")
                province_fig, province_df = create_province_analysis(df_bczs)
                
                if province_fig:
                    col_map, col_table = st.columns([2, 1])
                    
                    with col_map:
                        st.plotly_chart(province_fig, use_container_width=True)
                    
                    with col_table:
                        if province_df is not None:
                            top_provinces = province_df.nlargest(5, 'Score Global')
                            st.markdown("**🏆 Top 5 provinces**")
                            for i, (_, row) in enumerate(top_provinces.iterrows(), 1):
                                st.markdown(f"{i}. **{row['Province']}**: {row['Score Global']:.1f}%")
                
                st.markdown("#### 💡 RECOMMANDATIONS PRIORITAIRES")
                
                if 'Connaissance indicateurs' in scores and scores['Connaissance indicateurs'] < 50:
                    st.markdown("""
                    <div class="highlight-box">
                        <h4>🚨 PRIORITÉ 1 - RENFORCEMENT DES CAPACITÉS</h4>
                        <p><strong>Problème:</strong> Faible connaissance des définitions d'indicateurs TB</p>
                        <p><strong>Action recommandée:</strong> Sessions de formation intensive sur les 10 indicateurs clés</p>
                        <p><strong>Cible:</strong> 100% des chargés de données formés d'ici 3 mois</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                if 'Structure S&E' in scores and scores['Structure S&E'] < 60:
                    st.markdown("""
                    <div class="highlight-box">
                        <h4>⚠️ PRIORITÉ 2 - AMÉLIORATION DES PROCESSUS</h4>
                        <p><strong>Problème:</strong> Structures de S&E insuffisamment développées</p>
                        <p><strong>Action recommandée:</strong> Mise en place de plans de saisie et validation standardisés</p>
                        <p><strong>Cible:</strong> 80% des ZS avec processus formalisés d'ici 6 mois</p>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.warning("⏳ Aucune donnée BCZS disponible")
    
    # TAB 2: ANALYSE DÉTAILLÉE BCZS
    with tab2:
        if not df_bczs.empty:
            st.markdown('<h2 class="section-title">🔍 ANALYSE DÉTAILLÉE BCZS</h2>', unsafe_allow_html=True)
            
            st.markdown("#### 🎯 CONNAISSANCE DES INDICATEURS TB")
            heatmap_fig, knowledge_df = create_knowledge_heatmap(df_bczs)
            
            if heatmap_fig:
                col_chart, col_stats = st.columns([3, 1])
                
                with col_chart:
                    st.plotly_chart(heatmap_fig, use_container_width=True)
                
                with col_stats:
                    if knowledge_df is not None:
                        worst_indicators = knowledge_df.nsmallest(3, 'Score total')
                        st.markdown("**📉 Indicateurs les moins connus**")
                        for _, row in worst_indicators.iterrows():
                            st.markdown(f"• {row['Indicateur']}: {row['Score total']:.1f}%")
            
            st.markdown("#### 🎓 FORMATION DU PERSONNEL")
            training_fig, training_df = create_training_analysis(df_bczs)
            
            if training_fig:
                st.plotly_chart(training_fig, use_container_width=True)
                
                if training_df is not None:
                    col_train1, col_train2, col_train3 = st.columns(3)
                    
                    with col_train1:
                        avg_training = training_df['Taux formation (%)'].mean()
                        st.metric("Taux formation moyen", f"{avg_training:.1f}%")
                    
                    with col_train2:
                        best_training = training_df.loc[training_df['Taux formation (%)'].idxmax()]
                        st.metric("Meilleure formation", best_training['Formation'])
                    
                    with col_train3:
                        worst_training = training_df.loc[training_df['Taux formation (%)'].idxmin()]
                        st.metric("Formation à renforcer", worst_training['Formation'])
            
            st.markdown("#### 📋 DONNÉES BRUTES")
            
            with st.expander("Afficher les données BCZS"):
                st.dataframe(df_bczs, use_container_width=True)
                
                st.markdown("##### 📊 Statistiques descriptives")
                numeric_cols = df_bczs.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    st.dataframe(df_bczs[numeric_cols].describe(), use_container_width=True)
        
        else:
            st.warning("⏳ Aucune donnée BCZS disponible")
    
    # TAB 3: SYNTHÈSE CDT
    with tab3:
        st.markdown('<h2 class="section-title">🏘️ CDT - SYNTHÈSE ET INDICATEURS CLÉS</h2>', unsafe_allow_html=True)
        
        if not df_cdt.empty:
            cdt_scores = calculate_cdt_quality_score(df_cdt)
            
            if not cdt_scores.empty:
                st.markdown("#### 🎯 INDICATEURS CLÉS CDT")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    if 'Score Global CDT' in cdt_scores:
                        st.markdown(f"""
                        <div class="metric-box">
                            <div class="metric-label-dark">Score Global CDT</div>
                            <div class="metric-value-dark">{cdt_scores['Score Global CDT']:.1f}%</div>
                            <div style="background: #e5e7eb; height: 8px; border-radius: 4px; margin-top: 10px;">
                                <div style="background: #059669; width: {cdt_scores['Score Global CDT']}%; height: 100%; border-radius: 4px;"></div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                
                with col2:
                    if 'Documents sources disponibles' in cdt_scores:
                        st.markdown(f"""
                        <div class="metric-box">
                            <div class="metric-label-dark">Documents disponibles</div>
                            <div class="metric-value-dark">{cdt_scores['Documents sources disponibles']:.1f}%</div>
                            <div style="background: #e5e7eb; height: 8px; border-radius: 4px; margin-top: 10px;">
                                <div style="background: #059669; width: {cdt_scores['Documents sources disponibles']}%; height: 100%; border-radius: 4px;"></div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                
                with col3:
                    if 'Connaissance indicateurs CDT' in cdt_scores:
                        st.markdown(f"""
                        <div class="metric-box">
                            <div class="metric-label-dark">Connaissance indicateurs</div>
                            <div class="metric-value-dark">{cdt_scores['Connaissance indicateurs CDT']:.1f}%</div>
                            <div style="background: #e5e7eb; height: 8px; border-radius: 4px; margin-top: 10px;">
                                <div style="background: #059669; width: {cdt_scores['Connaissance indicateurs CDT']}%; height: 100%; border-radius: 4px;"></div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                
                with col4:
                    if 'Rapportage régulier' in cdt_scores:
                        st.markdown(f"""
                        <div class="metric-box">
                            <div class="metric-label-dark">Rapportage régulier</div>
                            <div class="metric-value-dark">{cdt_scores['Rapportage régulier']:.1f}%</div>
                            <div style="background: #e5e7eb; height: 8px; border-radius: 4px; margin-top: 10px;">
                                <div style="background: #059669; width: {cdt_scores['Rapportage régulier']}%; height: 100%; border-radius: 4px;"></div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                
                cdt_tab1, cdt_tab2, cdt_tab3, cdt_tab4 = st.tabs([
                    "📊 Analyse géographique", 
                    "📁 Documents sources", 
                    "🎓 Formation & Rapportage",
                    "📋 Données complètes"
                ])
                
                with cdt_tab1:
                    st.markdown("#### 🗺️ RÉPARTITION GÉOGRAPHIQUE")
                    geo_fig, province_df, zone_df = create_cdt_geographical_analysis(df_cdt)
                    
                    if geo_fig:
                        st.plotly_chart(geo_fig, use_container_width=True)
                        
                        col_geo1, col_geo2 = st.columns(2)
                        
                        with col_geo1:
                            if not province_df.empty:
                                st.markdown("**📈 Top provinces**")
                                top_provinces = province_df.nlargest(5, 'Score Global')
                                for i, (_, row) in enumerate(top_provinces.iterrows(), 1):
                                    st.markdown(f"{i}. **{row['Province']}**: {row['Score Global']:.1f}% ({row['Nombre CDT']} CDT)")
                        
                        with col_geo2:
                            if not zone_df.empty:
                                st.markdown("**🏥 Distribution par zone**")
                                st.dataframe(zone_df, use_container_width=True)
                
                with cdt_tab2:
                    st.markdown("#### 📁 DISPONIBILITÉ DES DOCUMENTS SOURCES")
                    doc_fig, doc_df = create_cdt_document_analysis(df_cdt)
                    
                    if doc_fig:
                        st.plotly_chart(doc_fig, use_container_width=True)
                        
                        if doc_df is not None:
                            col_doc1, col_doc2 = st.columns(2)
                            
                            with col_doc1:
                                worst_docs = doc_df.nsmallest(3, 'Taux disponibilité (%)')
                                st.markdown("**📉 Documents les moins disponibles**")
                                for _, row in worst_docs.iterrows():
                                    st.markdown(f"• {row['Document']}: {row['Taux disponibilité (%)']:.1f}%")
                            
                            with col_doc2:
                                avg_availability = doc_df['Taux disponibilité (%)'].mean()
                                st.metric("Disponibilité moyenne", f"{avg_availability:.1f}%")
                
                with cdt_tab3:
                    st.markdown("#### 🎓 FORMATION DU PERSONNEL")
                    training_fig, training_df = create_cdt_training_analysis(df_cdt)
                    
                    if training_fig:
                        st.plotly_chart(training_fig, use_container_width=True)
                    
                    st.markdown("#### 📤 RAPPORTAGE DES DONNÉES")
                    reporting_fig, reporting_df, means_df = create_cdt_reporting_analysis(df_cdt)
                    
                    if reporting_fig:
                        st.plotly_chart(reporting_fig, use_container_width=True)
                
                with cdt_tab4:
                    st.markdown("#### 📋 DONNÉES BRUTES CDT")
                    
                    with st.expander("Afficher toutes les données CDT"):
                        st.dataframe(df_cdt, use_container_width=True)
                        
                        st.markdown("##### 📊 Statistiques descriptives")
                        numeric_cols = df_cdt.select_dtypes(include=[np.number]).columns
                        if len(numeric_cols) > 0:
                            st.dataframe(df_cdt[numeric_cols].describe(), use_container_width=True)
                    
                    csv = df_cdt.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Télécharger les données CDT (CSV)",
                        data=csv,
                        file_name="drc_pnlt_rdqa_cdt_analyse.csv",
                        mime="text/csv"
                    )
                
                st.markdown("#### 💡 RECOMMANDATIONS CDT")
                
                if 'Documents sources disponibles' in cdt_scores and cdt_scores['Documents sources disponibles'] < 70:
                    st.markdown("""
                    <div class="highlight-box">
                        <h4>📁 PRIORITÉ 1 - DOCUMENTS SOURCES</h4>
                        <p><strong>Problème:</strong> Documents sources insuffisamment disponibles</p>
                        <p><strong>Action recommandée:</strong> Distribution et formation sur l'utilisation des registres standard</p>
                        <p><strong>Cible:</strong> 90% de disponibilité dans tous les CDT d'ici 2 mois</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                if 'Connaissance indicateurs CDT' in cdt_scores and cdt_scores['Connaissance indicateurs CDT'] < 60:
                    st.markdown("""
                    <div class="highlight-box">
                        <h4>🎯 PRIORITÉ 2 - CONNAISSANCE INDICATEURS</h4>
                        <p><strong>Problème:</strong> Personnel ne maîtrise pas les définitions d'indicateurs</p>
                        <p><strong>Action recommandée:</strong> Sessions de rappel et fiches mémo dans chaque CDT</p>
                        <p><strong>Cible:</strong> 80% du personnel connaît les 10 indicateurs clés</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                if 'Rapportage régulier' in cdt_scores and cdt_scores['Rapportage régulier'] < 80:
                    st.markdown("""
                    <div class="highlight-box">
                        <h4>📤 PRIORITÉ 3 - RAPPORTAGE RÉGULIER</h4>
                        <p><strong>Problème:</strong> Taux de rapportage insuffisant</p>
                        <p><strong>Action recommandée:</strong> Système de suivi et rappels automatiques</p>
                        <p><strong>Cible:</strong> 95% des CDT rapportent mensuellement</p>
                    </div>
                    """, unsafe_allow_html=True)
            
        else:
            st.markdown("""
            <div class="highlight-box">
                <h4>📁 CHARGEMENT DES DONNÉES CDT</h4>
                <p>Pour analyser les données des Centres de Diagnostic et Traitement :</p>
                <ol>
                    <li>Le fichier doit être au format CSV avec les colonnes du XLSForm CDT</li>
                    <li>Les colonnes doivent commencer par <code>q001a</code>, <code>q010a</code>, etc.</li>
                    <li>Assurez-vous que les données sont au format WIDE (une ligne par CDT)</li>
                </ol>
            </div>
            """, unsafe_allow_html=True)
            
            uploaded_file = st.file_uploader(
                "Téléchargez votre fichier CDT:",
                type=['csv', 'xlsx'],
                key="cdt_upload",
                help="Format attendu: CSV ou Excel avec colonnes q100, q101, q102, etc."
            )
            
            if uploaded_file is not None:
                try:
                    if uploaded_file.name.endswith('.csv'):
                        df_cdt_uploaded = pd.read_csv(uploaded_file, encoding='utf-8')
                    else:
                        df_cdt_uploaded = pd.read_excel(uploaded_file)
                    
                    required_cols = ['q100', 'q101', 'q102', 'q103', 'q104']
                    missing_cols = [col for col in required_cols if col not in df_cdt_uploaded.columns]
                    
                    if missing_cols:
                        st.warning(f"⚠️ Colonnes manquantes: {', '.join(missing_cols)}")
                        st.info("Le fichier devrait contenir les questions du questionnaire CDT")
                    else:
                        st.session_state.df_cdt = df_cdt_uploaded
                        st.success(f"✅ Données CDT chargées: {len(df_cdt_uploaded)} centres")
                        st.rerun()
                        
                except Exception as e:
                    st.error(f"❌ Erreur de chargement: {str(e)}")
    
    # TAB 4: GRAPHIQUES DÉTAILLÉS CDT
    with tab4:
        st.markdown('<h2 class="section-title">📊 ANALYSE DÉTAILLÉE PAR QUESTION - CDT</h2>', unsafe_allow_html=True)
        
        if not df_cdt.empty:
            # Résumé statistique
            st.markdown("#### 📈 RÉSUMÉ STATISTIQUE")
            col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
            
            with col_stat1:
                total_cdt = len(df_cdt)
                st.metric("Total CDT analysés", total_cdt)
            
            with col_stat2:
                response_rates = {}
                for col in df_cdt.columns:
                    if col.startswith('q'):
                        total = len(df_cdt)
                        answered = df_cdt[col].count()
                        response_rates[col] = answered / total * 100 if total > 0 else 0
                
                avg_response_rate = np.mean(list(response_rates.values())) if response_rates else 0
                st.metric("Taux réponse moyen", f"{avg_response_rate:.1f}%")
            
            with col_stat3:
                question_cols = [col for col in df_cdt.columns if col.startswith('q')]
                st.metric("Questions disponibles", len(question_cols))
            
            with col_stat4:
                cdt_scores = calculate_cdt_quality_score(df_cdt)
                if 'Score Global CDT' in cdt_scores:
                    st.metric("Score global CDT", f"{cdt_scores['Score Global CDT']:.1f}%")
            
            # Structure et fonction du S&E
            st.markdown("### 📋 STRUCTURE ET FONCTION DU SUIVI & ÉVALUATION")
            st.markdown("#### Q100-Q102: Organisation et responsabilités")
            
            structure_fig = create_cdt_structure_chart(df_cdt)
            
                        
            # Connaissance des indicateurs
            st.markdown("---")
            st.markdown("### 🎯 CONNAISSANCE DES DÉFINITIONS D'INDICATEURS")
            st.markdown("#### Q105_1 à Q105_10: Maîtrise des indicateurs TB")
            
            
    
    # TAB 5: COMPARAISON MULTI-NIVEAUX
    with tab5:
        st.markdown('<h2 class="section-title">🔗 COMPARAISON MULTI-NIVEAUX</h2>', unsafe_allow_html=True)
        
        if not df_bczs.empty and not df_cdt.empty:
            bczs_scores = calculate_quality_score(df_bczs)
            cdt_scores = calculate_cdt_quality_score(df_cdt)
            
            comparison_data = []
            
            comparable_indicators = [
                ('Connaissance indicateurs', 'Connaissance indicateurs CDT'),
                ('Supervision qualité', 'Supervision qualité CDT'),
                ('Archivage', 'Archivage CDT')
            ]
            
            for bczs_key, cdt_key in comparable_indicators:
                if bczs_key in bczs_scores and cdt_key in cdt_scores:
                    comparison_data.append({
                        'Indicateur': bczs_key.replace(' CDT', ''),
                        'BCZS': bczs_scores[bczs_key],
                        'CDT': cdt_scores[cdt_key],
                        'Écart': cdt_scores[cdt_key] - bczs_scores[bczs_key]
                    })
            
            if comparison_data:
                comp_df = pd.DataFrame(comparison_data)
                
                fig = go.Figure()
                
                fig.add_trace(go.Bar(
                    name='BCZS',
                    x=comp_df['Indicateur'],
                    y=comp_df['BCZS'],
                    marker_color='#3B82F6',
                    text=comp_df['BCZS'].round(1).astype(str) + '%',
                    textposition='auto'
                ))
                
                fig.add_trace(go.Bar(
                    name='CDT',
                    x=comp_df['Indicateur'],
                    y=comp_df['CDT'],
                    marker_color='#10B981',
                    text=comp_df['CDT'].round(1).astype(str) + '%',
                    textposition='auto'
                ))
                
                fig.update_layout(
                    title='Comparaison des performances BCZS vs CDT',
                    height=400,
                    barmode='group',
                    xaxis_title='',
                    yaxis_title='Score (%)',
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("#### 📊 INSIGHTS COMPARATIFS")
                
                max_gap = comp_df.loc[comp_df['Écart'].abs().idxmax()]
                if max_gap['Écart'] > 10:
                    st.info(f"**Plus grande différence:** {max_gap['Indicateur']} - CDT score {max_gap['Écart']:.1f} points de plus")
                elif max_gap['Écart'] < -10:
                    st.warning(f"**Plus grande différence:** {max_gap['Indicateur']} - CDT score {-max_gap['Écart']:.1f} points de moins")
                
                st.markdown("#### 💡 RECOMMANDATIONS INTÉGRÉES")
                
                avg_bczs = bczs_scores.get('Score Global', 0)
                avg_cdt = cdt_scores.get('Score Global CDT', 0)
                
                if avg_cdt > avg_bczs + 10:
                    st.markdown("""
                    <div class="highlight-box">
                        <h4>✅ POINT FORT - PERFORMANCE CDT</h4>
                        <p>Les CDT performent mieux que les BCZS. Capitalisez sur cette force en :</p>
                        <ul>
                            <li>Documentant les bonnes pratiques des CDT performants</li>
                            <li>Organisant des échanges BCZS-CDT</li>
                            <li>Utilisant les CDT comme sites démonstratifs</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
                elif avg_bczs > avg_cdt + 10:
                    st.markdown("""
                    <div class="highlight-box">
                        <h4>⚠️ DÉFI - RENFORCEMENT CDT</h4>
                        <p>Les CDT sont en retard par rapport aux BCZS. Priorités :</p>
                        <ul>
                            <li>Renforcer l'appui technique des BCZS aux CDT</li>
                            <li>Simplifier les procédures pour les CDT</li>
                            <li>Augmenter les visites de supervision</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("📋 Pour la comparaison, chargez à la fois les données BCZS et CDT")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.9rem; padding: 1rem;">
        <p>📋 <strong>Programme National de Lutte contre la Tuberculose (PNLT)</strong> - République Démocratique du Congo</p>
        <p>📞 Contact technique: Fabien Kabasele, +243974061912 | 📧 fabiennkabasele@gmail.com</p>
        <p>🔄 Données actualisées le: {date} | 📊 BCZS analysés: {n_bczs} | 🏘️ CDT analysés: {n_cdt}</p>
        <p>📈 Score moyen BCZS: {score_bczs} | 🏥 Score moyen CDT: {score_cdt}</p>
    </div>
    """.format(
        date=pd.Timestamp.now().strftime('%d/%m/%Y %H:%M'),
        n_bczs=len(df_bczs) if not df_bczs.empty else 0,
        n_cdt=len(df_cdt) if not df_cdt.empty else 0,
        score_bczs=f"{calculate_quality_score(df_bczs).get('Score Global', 0):.1f}%" if not df_bczs.empty else "N/A",
        score_cdt=f"{calculate_cdt_quality_score(df_cdt).get('Score Global CDT', 0):.1f}%" if not df_cdt.empty else "N/A"
    ), unsafe_allow_html=True)

# ============================================================================
# EXÉCUTION
# ============================================================================

if __name__ == "__main__":
    main()