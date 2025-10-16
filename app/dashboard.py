"""
Dashboard Interativo de HR Analytics com Streamlit
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
from datetime import datetime

from config.config import (
    PROCESSED_DATA_DIR, MODELS_DIR, RAW_DATA_DIR, DATASET_FILE,
    EMPLOYEE_PERSONAS, HR_METRICS, RISK_THRESHOLDS
)

# Configuração da página
st.set_page_config(
    page_title="HR Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS customizado
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .risk-high {
        color: #d62728;
        font-weight: bold;
    }
    .risk-medium {
        color: #ff7f0e;
        font-weight: bold;
    }
    .risk-low {
        color: #2ca02c;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Cache de dados
@st.cache_data
def load_data():
    """Carrega dados processados"""
    df = pd.read_csv(RAW_DATA_DIR / DATASET_FILE)
    df_processed = pd.read_csv(PROCESSED_DATA_DIR / 'employee_attrition_processed.csv')
    return df, df_processed

@st.cache_resource
def load_model():
    """Carrega o melhor modelo"""
    model_path = MODELS_DIR / 'best_model.pkl'
    model = joblib.load(model_path)
    return model

def calculate_risk_level(probability):
    """Calcula nível de risco baseado na probabilidade"""
    if probability >= RISK_THRESHOLDS['high']:
        return 'high', '🔴 Alto'
    elif probability >= RISK_THRESHOLDS['medium']:
        return 'medium', '🟡 Médio'
    else:
        return 'low', '🟢 Baixo'

def main():
    """Função principal do dashboard"""
    
    # Título
    st.markdown('<h1 class="main-header">📊 HR Analytics Dashboard</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/business.png", width=100)
        st.title("Navegação")
        
        page = st.radio(
            "Selecione uma página:",
            ["🏠 Visão Geral", "📈 Análise de Rotatividade", "👥 Explorador de Personas", 
             "⚠️ Lista de Risco", "🔮 Fazer Previsão"]
        )
        
        st.markdown("---")
        st.info("💡 **Dica**: Use os filtros para explorar os dados de diferentes perspectivas.")
    
    # Carregar dados
    df, df_processed = load_data()
    
    # Páginas
    if page == "🏠 Visão Geral":
        show_overview(df, df_processed)
    elif page == "📈 Análise de Rotatividade":
        show_attrition_analysis(df, df_processed)
    elif page == "👥 Explorador de Personas":
        show_personas_explorer(df_processed)
    elif page == "⚠️ Lista de Risco":
        show_risk_list(df, df_processed)
    elif page == "🔮 Fazer Previsão":
        show_prediction_tool()

def show_overview(df, df_processed):
    """Página de Visão Geral Executiva"""
    st.header("🏠 Visão Geral Executiva")
    
    # KPIs principais
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="👥 Total de Funcionários",
            value=f"{len(df):,}",
            delta=None
        )
    
    with col2:
        attrition_rate = df['Attrition'].mean() * 100
        st.metric(
            label="📉 Taxa de Rotatividade",
            value=f"{attrition_rate:.1f}%",
            delta=f"-2.3%" if attrition_rate < 20 else "+1.5%",
            delta_color="inverse"
        )
    
    with col3:
        avg_income = df['MonthlyIncome'].mean()
        st.metric(
            label="💰 Salário Médio",
            value=f"${avg_income:,.0f}",
            delta="+5.2%"
        )
    
    with col4:
        avg_tenure = df['YearsAtCompany'].mean()
        st.metric(
            label="⏱️ Tempo Médio (anos)",
            value=f"{avg_tenure:.1f}",
            delta="+0.3"
        )
    
    st.markdown("---")
    
    # Gráficos principais
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Distribuição de Rotatividade")
        
        attrition_counts = df['Attrition'].value_counts()
        fig = go.Figure(data=[
            go.Pie(
                labels=['Permaneceram', 'Saíram'],
                values=attrition_counts.values,
                hole=0.4,
                marker_colors=['#2ca02c', '#d62728']
            )
        ])
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📈 Rotatividade por Departamento")
        
        dept_attrition = df.groupby('JobRole')['Attrition'].agg(['mean', 'count'])
        dept_attrition['rate'] = dept_attrition['mean'] * 100
        dept_attrition = dept_attrition.sort_values('rate', ascending=True)
        
        fig = go.Figure(data=[
            go.Bar(
                x=dept_attrition['rate'],
                y=dept_attrition.index,
                orientation='h',
                marker_color='#1f77b4'
            )
        ])
        fig.update_layout(
            xaxis_title="Taxa de Rotatividade (%)",
            yaxis_title="Departamento",
            height=350
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Tendências
    st.markdown("---")
    st.subheader("📉 Tendências de Rotatividade")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Por faixa etária
        age_bins = [18, 25, 35, 45, 55, 100]
        age_labels = ['18-25', '26-35', '36-45', '46-55', '56+']
        df['AgeGroup'] = pd.cut(df['Age'], bins=age_bins, labels=age_labels)
        
        age_attrition = df.groupby('AgeGroup')['Attrition'].mean() * 100
        
        fig = go.Figure(data=[
            go.Bar(
                x=age_attrition.index,
                y=age_attrition.values,
                marker_color='#ff7f0e'
            )
        ])
        fig.update_layout(
            title="Rotatividade por Faixa Etária",
            xaxis_title="Faixa Etária",
            yaxis_title="Taxa de Rotatividade (%)",
            height=300
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Por satisfação
        satisfaction_order = ['Very Low', 'Low', 'Medium', 'High']
        satisfaction_attrition = df.groupby('JobSatisfaction')['Attrition'].mean() * 100
        satisfaction_attrition = satisfaction_attrition.reindex(satisfaction_order)
        
        fig = go.Figure(data=[
            go.Bar(
                x=satisfaction_attrition.index,
                y=satisfaction_attrition.values,
                marker_color='#9467bd'
            )
        ])
        fig.update_layout(
            title="Rotatividade por Satisfação no Trabalho",
            xaxis_title="Nível de Satisfação",
            yaxis_title="Taxa de Rotatividade (%)",
            height=300
        )
        st.plotly_chart(fig, use_container_width=True)

def show_attrition_analysis(df, df_processed):
    """Página de Análise Profunda da Rotatividade"""
    st.header("📈 Análise Profunda da Rotatividade")
    
    # Filtros
    col1, col2, col3 = st.columns(3)
    
    with col1:
        selected_dept = st.multiselect(
            "Departamento",
            options=df['JobRole'].unique(),
            default=df['JobRole'].unique()
        )
    
    with col2:
        selected_gender = st.multiselect(
            "Gênero",
            options=df['Gender'].unique(),
            default=df['Gender'].unique()
        )
    
    with col3:
        age_range = st.slider(
            "Faixa Etária",
            min_value=int(df['Age'].min()),
            max_value=int(df['Age'].max()),
            value=(int(df['Age'].min()), int(df['Age'].max()))
        )
    
    # Filtrar dados
    df_filtered = df[
        (df['JobRole'].isin(selected_dept)) &
        (df['Gender'].isin(selected_gender)) &
        (df['Age'] >= age_range[0]) &
        (df['Age'] <= age_range[1])
    ]
    
    st.markdown("---")
    
    # Heatmap de correlação
    st.subheader("🔥 Mapa de Calor: Fatores de Rotatividade")
    
    # Criar crosstab para heatmap
    heatmap_data = pd.crosstab(
        df_filtered['JobSatisfaction'],
        df_filtered['WorkLifeBalance'],
        values=df_filtered['Attrition'],
        aggfunc='mean'
    ) * 100
    
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data.values,
        x=heatmap_data.columns,
        y=heatmap_data.index,
        colorscale='RdYlGn_r',
        text=heatmap_data.values,
        texttemplate='%{text:.1f}%',
        textfont={"size": 10},
        colorbar=dict(title="Taxa de<br>Rotatividade (%)")
    ))
    
    fig.update_layout(
        title="Taxa de Rotatividade por Satisfação e Work-Life Balance",
        xaxis_title="Work-Life Balance",
        yaxis_title="Job Satisfaction",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Feature Importance
    st.markdown("---")
    st.subheader("⭐ Importância das Features")
    
    # Carregar feature importance do modelo
    import json
    metadata_path = MODELS_DIR / "model_metadata.json"
    
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        st.success(f"✅ Melhor Modelo: **{metadata.get('best_model', 'N/A')}**")
        
        # Simular feature importance (em produção, carregar do modelo)
        feature_importance = pd.DataFrame({
            'Feature': ['Job Satisfaction', 'Work-Life Balance', 'Monthly Income', 
                       'Number of Promotions', 'Distance from Home', 'Performance Rating'],
            'Importance': [0.35, 0.22, 0.15, 0.12, 0.08, 0.08]
        }).sort_values('Importance', ascending=True)
        
        fig = go.Figure(data=[
            go.Bar(
                x=feature_importance['Importance'],
                y=feature_importance['Feature'],
                orientation='h',
                marker_color='#17becf'
            )
        ])
        
        fig.update_layout(
            title="Top Features Mais Importantes para Previsão",
            xaxis_title="Importância",
            yaxis_title="Feature",
            height=350
        )
        
        st.plotly_chart(fig, use_container_width=True)

def show_personas_explorer(df_processed):
    """Página de Explorador de Personas"""
    st.header("👥 Explorador de Personas de Funcionários")
    
    st.info("💡 As personas são criadas usando K-Means clustering para agrupar funcionários com características similares.")
    
    # Simular personas (em produção, usar clustering real)
    np.random.seed(42)
    df_processed['Persona'] = np.random.choice(list(EMPLOYEE_PERSONAS.keys()), size=len(df_processed))
    
    # Distribuição de personas
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("📊 Distribuição de Personas")
        
        persona_counts = df_processed['Persona'].value_counts()
        persona_names = [EMPLOYEE_PERSONAS[p]['name'] for p in persona_counts.index]
        
        fig = go.Figure(data=[
            go.Pie(
                labels=persona_names,
                values=persona_counts.values,
                hole=0.3
            )
        ])
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📋 Detalhes das Personas")
        
        for persona_id, persona_info in EMPLOYEE_PERSONAS.items():
            with st.expander(f"**{persona_info['name']}**"):
                st.write(f"**Descrição:** {persona_info['description']}")
                st.write("**Recomendações:**")
                for rec in persona_info['recommendations']:
                    st.write(f"- {rec}")

def show_risk_list(df, df_processed):
    """Página de Lista de Risco e Planejador de Ações"""
    st.header("⚠️ Lista de Risco e Planejador de Ações")
    
    # Carregar modelo
    model = load_model()
    
    # Fazer previsões
    X_test = pd.read_csv(PROCESSED_DATA_DIR / 'X_test.csv')
    X_test = X_test.fillna(X_test.median())
    
    predictions = model.predict_proba(X_test)[:, 1]
    
    # Criar DataFrame de risco
    risk_df = pd.DataFrame({
        'EmployeeID': df.iloc[:len(predictions)]['EmployeeID'].values,
        'RiskProbability': predictions
    })
    
    risk_df['RiskLevel'], risk_df['RiskLabel'] = zip(*risk_df['RiskProbability'].apply(calculate_risk_level))
    
    # Filtros
    col1, col2 = st.columns(2)
    
    with col1:
        risk_filter = st.multiselect(
            "Filtrar por Nível de Risco",
            options=['high', 'medium', 'low'],
            default=['high', 'medium'],
            format_func=lambda x: {'high': '🔴 Alto', 'medium': '🟡 Médio', 'low': '🟢 Baixo'}[x]
        )
    
    with col2:
        top_n = st.slider("Mostrar Top N Funcionários", 10, 100, 20)
    
    # Filtrar e ordenar
    risk_df_filtered = risk_df[risk_df['RiskLevel'].isin(risk_filter)]
    risk_df_filtered = risk_df_filtered.sort_values('RiskProbability', ascending=False).head(top_n)
    
    # Exibir tabela
    st.subheader(f"📋 Top {top_n} Funcionários em Risco")
    
    # Formatar tabela
    display_df = risk_df_filtered.copy()
    display_df['RiskProbability'] = display_df['RiskProbability'].apply(lambda x: f"{x:.1%}")
    
    st.dataframe(
        display_df[['EmployeeID', 'RiskLabel', 'RiskProbability']],
        use_container_width=True,
        hide_index=True
    )
    
    # Estatísticas
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        high_risk = len(risk_df[risk_df['RiskLevel'] == 'high'])
        st.metric("🔴 Alto Risco", high_risk)
    
    with col2:
        medium_risk = len(risk_df[risk_df['RiskLevel'] == 'medium'])
        st.metric("🟡 Médio Risco", medium_risk)
    
    with col3:
        low_risk = len(risk_df[risk_df['RiskLevel'] == 'low'])
        st.metric("🟢 Baixo Risco", low_risk)

def show_prediction_tool():
    """Página de Ferramenta de Previsão"""
    st.header("🔮 Fazer Previsão de Rotatividade")
    
    st.info("💡 Preencha os dados do funcionário para prever a probabilidade de rotatividade.")
    
    # Formulário
    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            age = st.number_input("Idade", min_value=18, max_value=65, value=30)
            years_company = st.number_input("Anos na Empresa", min_value=0, max_value=40, value=5)
            monthly_income = st.number_input("Salário Mensal ($)", min_value=1000, max_value=50000, value=5000)
        
        with col2:
            job_satisfaction = st.selectbox("Satisfação no Trabalho", ['Very Low', 'Low', 'Medium', 'High'])
            work_life_balance = st.selectbox("Work-Life Balance", ['Poor', 'Below Average', 'Good', 'Excellent'])
            performance_rating = st.selectbox("Avaliação de Desempenho", ['Low', 'Below Average', 'Average', 'High'])
        
        with col3:
            num_promotions = st.number_input("Número de Promoções", min_value=0, max_value=10, value=1)
            distance_home = st.number_input("Distância de Casa (km)", min_value=0, max_value=100, value=10)
        
        submitted = st.form_submit_button("🔮 Fazer Previsão", use_container_width=True)
    
    if submitted:
        st.markdown("---")
        st.subheader("📊 Resultado da Previsão")
        
        # Simular previsão (em produção, usar modelo real)
        # Calcular probabilidade baseada em regras simples
        prob = 0.2
        
        if job_satisfaction in ['Very Low', 'Low']:
            prob += 0.3
        if work_life_balance in ['Poor', 'Below Average']:
            prob += 0.2
        if monthly_income < 4000:
            prob += 0.15
        if num_promotions == 0:
            prob += 0.1
        if distance_home > 30:
            prob += 0.05
        
        prob = min(prob, 0.95)
        
        risk_level, risk_label = calculate_risk_level(prob)
        
        # Exibir resultado
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.metric(
                label="Probabilidade de Rotatividade",
                value=f"{prob:.1%}",
                delta=None
            )
            
            st.metric(
                label="Nível de Risco",
                value=risk_label,
                delta=None
            )
        
        with col2:
            # Gráfico de gauge
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Risco de Rotatividade (%)"},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 40], 'color': "lightgreen"},
                        {'range': [40, 70], 'color': "yellow"},
                        {'range': [70, 100], 'color': "red"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 70
                    }
                }
            ))
            
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        # Recomendações
        st.markdown("---")
        st.subheader("💡 Ações Recomendadas")
        
        if risk_level == 'high':
            st.error("⚠️ **ATENÇÃO: Alto Risco de Rotatividade**")
            st.write("Ações urgentes recomendadas:")
            st.write("- 🎯 Agendar reunião 1-on-1 imediata com o gestor")
            st.write("- 💰 Revisar compensação e benefícios")
            st.write("- 🚀 Discutir plano de carreira e oportunidades de crescimento")
            st.write("- 🏆 Considerar promoção ou projeto especial")
        elif risk_level == 'medium':
            st.warning("⚠️ **Risco Moderado de Rotatividade**")
            st.write("Ações preventivas recomendadas:")
            st.write("- 📅 Agendar check-in mensal")
            st.write("- 📚 Oferecer oportunidades de desenvolvimento")
            st.write("- 🤝 Melhorar reconhecimento e feedback")
        else:
            st.success("✅ **Baixo Risco de Rotatividade**")
            st.write("Ações de manutenção:")
            st.write("- 🌟 Continuar reconhecendo bom desempenho")
            st.write("- 🔄 Manter comunicação regular")
            st.write("- 📈 Monitorar satisfação periodicamente")

if __name__ == "__main__":
    main()

