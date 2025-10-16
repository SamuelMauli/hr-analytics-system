"""
==============================================================================
DASHBOARD DE OTIMIZAÇÃO - BRANCH AND BOUND
==============================================================================

Disciplina: Pesquisa Operacional
Interface: Streamlit
Objetivo: Visualizar e interagir com o algoritmo Branch and Bound

PÁGINAS DO DASHBOARD:
---------------------
1. 📊 Análise de Dados: EDA do dataset de rotatividade
2. 🎯 Configuração: Parâmetros do problema de otimização
3. 🚀 Execução: Rodar Branch and Bound e visualizar resultados
4. 📈 Análise de Sensibilidade: Impacto de variação de parâmetros
5. 🔬 Comparação: Branch and Bound vs Heurística Gulosa

AUTOR: Manus AI
DATA: 16 de outubro de 2025
==============================================================================
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import json

from config.config import PROCESSED_DATA_DIR
from src.optimization.branch_and_bound import (
    Project, BranchAndBound, greedy_heuristic
)


# ==============================================================================
# CONFIGURAÇÃO DA PÁGINA
# ==============================================================================

st.set_page_config(
    page_title="HR Analytics - Branch and Bound",
    page_icon="🎯",
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
        border-left: 4px solid #1f77b4;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 0.25rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        border-radius: 0.25rem;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# FUNÇÕES AUXILIARES
# ==============================================================================

@st.cache_data
def load_projects():
    """
    Carrega projetos de retenção do CSV.
    
    DEFESA: Cache para evitar recarregamento desnecessário
    """
    projects_file = PROCESSED_DATA_DIR / 'retention_projects.csv'
    
    if not projects_file.exists():
        st.error(f"❌ Arquivo de projetos não encontrado: {projects_file}")
        st.info("Execute primeiro: python src/optimization/prepare_optimization_data.py")
        st.stop()
    
    df = pd.read_csv(projects_file)
    
    projects = [
        Project(
            id=row['id'],
            name=row['name'],
            cost=row['cost'],
            impact=row['impact'],
            category=row['category']
        )
        for _, row in df.iterrows()
    ]
    
    return projects, df


@st.cache_data
def load_justifications():
    """Carrega justificativas dos projetos."""
    justif_file = PROCESSED_DATA_DIR / 'projects_justifications.json'
    
    if justif_file.exists():
        with open(justif_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []


def run_sensitivity_analysis(projects, base_budget, budget_range):
    """
    Executa análise de sensibilidade variando o orçamento.
    
    DEFESA: Análise essencial para entender robustez da solução
    """
    results = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, budget in enumerate(budget_range):
        status_text.text(f"Analisando orçamento: R$ {budget:.0f}k...")
        
        bb = BranchAndBound(projects, budget)
        result = bb.solve(verbose=False)
        
        if result["status"] == "optimal":
            results.append({
                'budget': budget,
                'total_impact': result['solution']['total_impact'],
                'total_cost': result['solution']['total_cost'],
                'n_projects': result['solution']['n_projects_selected'],
                'budget_used_pct': result['solution']['budget_used_pct'],
                'nodes_expanded': result['metrics']['nodes_expanded'],
                'execution_time': result['metrics']['execution_time_seconds']
            })
        
        progress_bar.progress((i + 1) / len(budget_range))
    
    status_text.empty()
    progress_bar.empty()
    
    return pd.DataFrame(results)


# ==============================================================================
# SIDEBAR - NAVEGAÇÃO
# ==============================================================================

st.sidebar.title("🎯 HR Analytics")
st.sidebar.markdown("### Branch and Bound")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navegação",
    ["📊 Análise de Dados", 
     "🎯 Configuração", 
     "🚀 Execução", 
     "📈 Sensibilidade",
     "🔬 Comparação"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Sobre")
st.sidebar.info(
    "Sistema de otimização para seleção de projetos de retenção de "
    "funcionários usando Branch and Bound.\n\n"
    "**Disciplina:** Pesquisa Operacional\n\n"
    "**Problema:** Mochila 0-1"
)


# ==============================================================================
# PÁGINA 1: ANÁLISE DE DADOS
# ==============================================================================

if page == "📊 Análise de Dados":
    st.markdown('<div class="main-header">📊 Análise de Dados</div>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    ### Dataset: Employee Attrition
    
    **Fonte:** [Kaggle - Employee Attrition Dataset](https://www.kaggle.com/datasets/stealthtechnologies/employee-attrition-dataset)
    
    Este dataset contém informações sobre 10.000 funcionários e fatores relacionados 
    à rotatividade (attrition). A partir desta análise, criamos um portfólio de 
    projetos de retenção para otimização.
    """)
    
    # Carregar projetos
    projects, projects_df = load_projects()
    justifications = load_justifications()
    
    # Estatísticas gerais
    st.markdown("### 📈 Estatísticas do Portfólio")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total de Projetos", len(projects))
    
    with col2:
        total_cost = projects_df['cost'].sum()
        st.metric("Custo Total", f"R$ {total_cost:.0f}k")
    
    with col3:
        total_impact = projects_df['impact'].sum()
        st.metric("Impacto Total", f"{total_impact:.1f}%")
    
    with col4:
        avg_efficiency = projects_df['efficiency'].mean()
        st.metric("Eficiência Média", f"{avg_efficiency:.3f}")
    
    st.markdown("---")
    
    # Visualizações
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Distribuição de Custos")
        fig = px.histogram(projects_df, x='cost', nbins=15,
                          labels={'cost': 'Custo (R$ mil)', 'count': 'Frequência'},
                          color_discrete_sequence=['#1f77b4'])
        fig.update_layout(showlegend=False, height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### Distribuição de Impactos")
        fig = px.histogram(projects_df, x='impact', nbins=15,
                          labels={'impact': 'Impacto (%)', 'count': 'Frequência'},
                          color_discrete_sequence=['#ff7f0e'])
        fig.update_layout(showlegend=False, height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    # Scatter plot: Custo vs Impacto
    st.markdown("#### Custo vs Impacto (Eficiência)")
    fig = px.scatter(projects_df, x='cost', y='impact', 
                     size='efficiency', color='category',
                     hover_data=['name'],
                     labels={'cost': 'Custo (R$ mil)', 
                            'impact': 'Impacto (%)',
                            'category': 'Categoria'},
                     title="Cada ponto representa um projeto (tamanho = eficiência)")
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    # Tabela de projetos
    st.markdown("#### 📋 Portfólio Completo de Projetos")
    
    # Adicionar filtros
    col1, col2 = st.columns(2)
    
    with col1:
        categories = ['Todos'] + sorted(projects_df['category'].unique().tolist())
        selected_category = st.selectbox("Filtrar por Categoria", categories)
    
    with col2:
        sort_by = st.selectbox("Ordenar por", 
                               ['Eficiência (maior)', 'Custo (menor)', 
                                'Impacto (maior)', 'Nome'])
    
    # Aplicar filtros
    filtered_df = projects_df.copy()
    if selected_category != 'Todos':
        filtered_df = filtered_df[filtered_df['category'] == selected_category]
    
    # Aplicar ordenação
    if sort_by == 'Eficiência (maior)':
        filtered_df = filtered_df.sort_values('efficiency', ascending=False)
    elif sort_by == 'Custo (menor)':
        filtered_df = filtered_df.sort_values('cost')
    elif sort_by == 'Impacto (maior)':
        filtered_df = filtered_df.sort_values('impact', ascending=False)
    else:
        filtered_df = filtered_df.sort_values('name')
    
    # Exibir tabela
    st.dataframe(
        filtered_df[['name', 'category', 'cost', 'impact', 'efficiency']].style.format({
            'cost': 'R$ {:.2f}k',
            'impact': '{:.2f}%',
            'efficiency': '{:.3f}'
        }),
        use_container_width=True,
        height=400
    )
    
    # Detalhes de um projeto selecionado
    st.markdown("#### 🔍 Detalhes do Projeto")
    selected_project_name = st.selectbox("Selecione um projeto", 
                                         filtered_df['name'].tolist())
    
    if selected_project_name:
        project_data = filtered_df[filtered_df['name'] == selected_project_name].iloc[0]
        justif_data = next((j for j in justifications if j['name'] == selected_project_name), None)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Custo", f"R$ {project_data['cost']:.2f}k")
        
        with col2:
            st.metric("Impacto", f"{project_data['impact']:.2f}%")
        
        with col3:
            st.metric("Eficiência", f"{project_data['efficiency']:.3f}")
        
        if justif_data:
            st.markdown(f"**Categoria:** {justif_data['category']}")
            st.markdown(f"**Justificativa:** {justif_data['justification']}")


# ==============================================================================
# PÁGINA 2: CONFIGURAÇÃO
# ==============================================================================

elif page == "🎯 Configuração":
    st.markdown('<div class="main-header">🎯 Configuração do Problema</div>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    ### Problema da Mochila 0-1 (Knapsack Problem)
    
    **Objetivo:** Selecionar projetos de retenção que maximizem o impacto total, 
    respeitando o orçamento disponível.
    """)
    
    # Carregar projetos
    projects, projects_df = load_projects()
    
    # Configurações
    st.markdown("### ⚙️ Parâmetros")
    
    col1, col2 = st.columns(2)
    
    with col1:
        budget = st.slider(
            "Orçamento Disponível (R$ mil)",
            min_value=50.0,
            max_value=float(projects_df['cost'].sum()),
            value=300.0,
            step=10.0,
            help="Orçamento total disponível para investir em projetos"
        )
    
    with col2:
        st.metric("Custo Total de Todos os Projetos", 
                 f"R$ {projects_df['cost'].sum():.2f}k")
        st.metric("Impacto Total Potencial", 
                 f"{projects_df['impact'].sum():.2f}%")
    
    # Salvar configurações no session_state
    st.session_state['budget'] = budget
    st.session_state['projects'] = projects
    
    # Modelagem matemática
    st.markdown("---")
    st.markdown("### 📐 Modelagem Matemática")
    
    st.latex(r"""
    \begin{aligned}
    & \text{Maximizar:} && Z = \sum_{i=1}^{n} impacto_i \cdot x_i \\
    & \text{Sujeito a:} && \sum_{i=1}^{n} custo_i \cdot x_i \leq Orçamento \\
    & && x_i \in \{0, 1\} \quad \forall i
    \end{aligned}
    """)
    
    st.markdown("""
    Onde:
    - **xᵢ**: Variável de decisão (1 se projeto i é selecionado, 0 caso contrário)
    - **impactoᵢ**: Impacto esperado do projeto i na redução de rotatividade (%)
    - **custoᵢ**: Custo de implementação do projeto i (R$ mil)
    - **Orçamento**: Orçamento total disponível (R$ mil)
    """)
    
    # Estratégia de Branch and Bound
    st.markdown("---")
    st.markdown("### 🌳 Estratégia de Branch and Bound")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **1. Bound (Limite Superior)**
        - Relaxação Linear Fracionária
        - Permite selecionar frações de projetos
        - Ordena por eficiência (impacto/custo)
        - Fornece upper bound válido
        """)
        
        st.markdown("""
        **2. Branching (Ramificação)**
        - Para cada nó, cria dois ramos:
          - Incluir projeto (xᵢ = 1)
          - Excluir projeto (xᵢ = 0)
        """)
    
    with col2:
        st.markdown("""
        **3. Pruning (Poda)**
        - Poda por inviabilidade: custo > orçamento
        - Poda por otimalidade: bound ≤ melhor solução
        - Poda por completude: todos decididos
        """)
        
        st.markdown("""
        **4. Busca**
        - Estratégia: Best-First Search
        - Estrutura: Fila de prioridade (heap)
        - Prioridade: Maior bound primeiro
        """)
    
    # Botão para próxima página
    st.markdown("---")
    if st.button("▶️ Prosseguir para Execução", type="primary"):
        st.switch_page("pages/execution.py")


# ==============================================================================
# PÁGINA 3: EXECUÇÃO
# ==============================================================================

elif page == "🚀 Execução":
    st.markdown('<div class="main-header">🚀 Execução do Branch and Bound</div>', 
                unsafe_allow_html=True)
    
    # Verificar se configurações existem
    if 'budget' not in st.session_state or 'projects' not in st.session_state:
        st.warning("⚠️ Configure os parâmetros primeiro na página 'Configuração'")
        projects, _ = load_projects()
        budget = 300.0
        st.session_state['budget'] = budget
        st.session_state['projects'] = projects
    else:
        budget = st.session_state['budget']
        projects = st.session_state['projects']
    
    st.info(f"💰 Orçamento configurado: R$ {budget:.2f}k")
    
    # Botão para executar
    if st.button("🚀 Executar Branch and Bound", type="primary"):
        with st.spinner("Executando algoritmo..."):
            # Executar Branch and Bound
            start_time = time.time()
            bb = BranchAndBound(projects, budget)
            result = bb.solve(verbose=False)
            execution_time = time.time() - start_time
            
            # Salvar resultado no session_state
            st.session_state['bb_result'] = result
            st.session_state['bb_execution_time'] = execution_time
        
        st.success("✅ Execução concluída!")
    
    # Exibir resultados se existirem
    if 'bb_result' in st.session_state:
        result = st.session_state['bb_result']
        
        if result['status'] == 'optimal':
            sol = result['solution']
            metrics = result['metrics']
            
            # Métricas principais
            st.markdown("### 📊 Solução Ótima Encontrada")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Impacto Total", f"{sol['total_impact']:.2f}%")
            
            with col2:
                st.metric("Custo Total", f"R$ {sol['total_cost']:.2f}k")
            
            with col3:
                st.metric("Orçamento Utilizado", f"{sol['budget_used_pct']:.1f}%")
            
            with col4:
                st.metric("Projetos Selecionados", 
                         f"{sol['n_projects_selected']}/{len(projects)}")
            
            # Projetos selecionados
            st.markdown("---")
            st.markdown("### 📋 Projetos Selecionados")
            
            selected_df = pd.DataFrame([
                {
                    'Nome': p.name,
                    'Categoria': p.category,
                    'Custo': p.cost,
                    'Impacto': p.impact,
                    'Eficiência': p.efficiency
                }
                for p in sol['selected_projects']
            ])
            
            st.dataframe(
                selected_df.style.format({
                    'Custo': 'R$ {:.2f}k',
                    'Impacto': '{:.2f}%',
                    'Eficiência': '{:.3f}'
                }),
                use_container_width=True
            )
            
            # Visualização: Custo vs Impacto
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Distribuição de Custos")
                fig = px.pie(selected_df, values='Custo', names='Nome',
                            title="Proporção do Orçamento por Projeto")
                fig.update_layout(height=350)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("#### Distribuição de Impactos")
                fig = px.pie(selected_df, values='Impacto', names='Nome',
                            title="Contribuição para Impacto Total")
                fig.update_layout(height=350)
                st.plotly_chart(fig, use_container_width=True)
            
            # Métricas de execução
            st.markdown("---")
            st.markdown("### 📈 Métricas de Execução")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Nós Expandidos", metrics['nodes_expanded'])
                st.metric("Nós Podados (Total)", metrics['nodes_pruned_total'])
            
            with col2:
                st.metric("Podados por Inviabilidade", 
                         metrics['nodes_pruned_infeasible'])
                st.metric("Podados por Bound", metrics['nodes_pruned_bound'])
            
            with col3:
                st.metric("Profundidade Máxima", metrics['max_depth'])
                st.metric("Tempo de Execução", 
                         f"{metrics['execution_time_seconds']:.3f}s")
            
            # Eficiência de poda
            st.markdown("#### Eficiência de Poda")
            pruning_eff = metrics['pruning_efficiency_pct']
            
            fig = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = pruning_eff,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Eficiência de Poda (%)"},
                delta = {'reference': 50},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 30], 'color': "lightgray"},
                        {'range': [30, 60], 'color': "gray"},
                        {'range': [60, 100], 'color': "lightgreen"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ))
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown(f"""
            **Interpretação:** {pruning_eff:.1f}% dos nós foram podados, evitando 
            exploração desnecessária da árvore de busca. Quanto maior, mais eficiente 
            foi o algoritmo.
            """)
        
        else:
            st.error("❌ Nenhuma solução viável encontrada com o orçamento configurado.")


# ==============================================================================
# PÁGINA 4: ANÁLISE DE SENSIBILIDADE
# ==============================================================================

elif page == "📈 Sensibilidade":
    st.markdown('<div class="main-header">📈 Análise de Sensibilidade</div>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    ### Impacto da Variação do Orçamento
    
    Esta análise mostra como a solução ótima varia conforme o orçamento disponível.
    """)
    
    # Carregar projetos
    projects, projects_df = load_projects()
    
    # Configurar range de orçamento
    col1, col2, col3 = st.columns(3)
    
    with col1:
        min_budget = st.number_input("Orçamento Mínimo (R$ mil)", 
                                     value=50.0, step=10.0)
    
    with col2:
        max_budget = st.number_input("Orçamento Máximo (R$ mil)", 
                                     value=600.0, step=10.0)
    
    with col3:
        step = st.number_input("Passo (R$ mil)", value=25.0, step=5.0)
    
    if st.button("🔬 Executar Análise de Sensibilidade", type="primary"):
        budget_range = list(range(int(min_budget), int(max_budget) + 1, int(step)))
        
        with st.spinner(f"Executando {len(budget_range)} simulações..."):
            sensitivity_df = run_sensitivity_analysis(projects, 300.0, budget_range)
            st.session_state['sensitivity_df'] = sensitivity_df
        
        st.success(f"✅ Análise concluída! {len(sensitivity_df)} pontos analisados.")
    
    # Exibir resultados
    if 'sensitivity_df' in st.session_state:
        df = st.session_state['sensitivity_df']
        
        # Gráfico 1: Impacto vs Orçamento
        st.markdown("### 📊 Impacto vs Orçamento")
        
        fig = px.line(df, x='budget', y='total_impact',
                     labels={'budget': 'Orçamento (R$ mil)', 
                            'total_impact': 'Impacto Total (%)'},
                     title="Como o impacto varia com o orçamento")
        fig.add_scatter(x=df['budget'], y=df['total_impact'], 
                       mode='markers', name='Pontos')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Gráfico 2: Número de Projetos vs Orçamento
        st.markdown("### 📈 Número de Projetos vs Orçamento")
        
        fig = px.line(df, x='budget', y='n_projects',
                     labels={'budget': 'Orçamento (R$ mil)', 
                            'n_projects': 'Número de Projetos Selecionados'},
                     title="Como o número de projetos varia com o orçamento")
        fig.add_scatter(x=df['budget'], y=df['n_projects'], 
                       mode='markers', name='Pontos')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Gráfico 3: Eficiência de Uso do Orçamento
        st.markdown("### 💰 Eficiência de Uso do Orçamento")
        
        fig = px.line(df, x='budget', y='budget_used_pct',
                     labels={'budget': 'Orçamento (R$ mil)', 
                            'budget_used_pct': 'Orçamento Utilizado (%)'},
                     title="Percentual do orçamento efetivamente utilizado")
        fig.add_scatter(x=df['budget'], y=df['budget_used_pct'], 
                       mode='markers', name='Pontos')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Estatísticas
        st.markdown("### 📊 Estatísticas da Análise")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Impacto Mínimo", f"{df['total_impact'].min():.2f}%")
            st.metric("Impacto Máximo", f"{df['total_impact'].max():.2f}%")
        
        with col2:
            st.metric("Projetos Mínimo", int(df['n_projects'].min()))
            st.metric("Projetos Máximo", int(df['n_projects'].max()))
        
        with col3:
            st.metric("Tempo Médio", f"{df['execution_time'].mean():.3f}s")
            st.metric("Nós Médios", f"{df['nodes_expanded'].mean():.0f}")


# ==============================================================================
# PÁGINA 5: COMPARAÇÃO
# ==============================================================================

elif page == "🔬 Comparação":
    st.markdown('<div class="main-header">🔬 Comparação: B&B vs Heurística</div>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    ### Branch and Bound vs Heurística Gulosa
    
    Comparação entre a solução ótima (Branch and Bound) e uma heurística simples 
    (Greedy - seleção por eficiência).
    """)
    
    # Carregar projetos
    projects, projects_df = load_projects()
    
    # Configurar orçamento
    budget = st.slider("Orçamento (R$ mil)", 50.0, 600.0, 300.0, 10.0)
    
    if st.button("⚡ Executar Comparação", type="primary"):
        with st.spinner("Executando ambos os algoritmos..."):
            # Branch and Bound
            bb_start = time.time()
            bb = BranchAndBound(projects, budget)
            bb_result = bb.solve(verbose=False)
            bb_time = time.time() - bb_start
            
            # Heurística Gulosa
            greedy_start = time.time()
            greedy_result = greedy_heuristic(projects, budget)
            greedy_time = time.time() - greedy_start
            
            # Salvar resultados
            st.session_state['comparison'] = {
                'bb': bb_result,
                'bb_time': bb_time,
                'greedy': greedy_result,
                'greedy_time': greedy_time
            }
        
        st.success("✅ Comparação concluída!")
    
    # Exibir resultados
    if 'comparison' in st.session_state:
        comp = st.session_state['comparison']
        bb_sol = comp['bb']['solution']
        greedy_sol = comp['greedy']['solution']
        
        # Tabela comparativa
        st.markdown("### 📊 Comparação de Resultados")
        
        comparison_df = pd.DataFrame({
            'Métrica': ['Impacto Total (%)', 'Custo Total (R$ mil)', 
                       'Projetos Selecionados', 'Orçamento Utilizado (%)',
                       'Tempo de Execução (s)'],
            'Branch and Bound': [
                f"{bb_sol['total_impact']:.2f}",
                f"{bb_sol['total_cost']:.2f}",
                bb_sol['n_projects_selected'],
                f"{bb_sol['budget_used_pct']:.1f}",
                f"{comp['bb_time']:.4f}"
            ],
            'Heurística Gulosa': [
                f"{greedy_sol['total_impact']:.2f}",
                f"{greedy_sol['total_cost']:.2f}",
                greedy_sol['n_projects_selected'],
                f"{greedy_sol['budget_used_pct']:.1f}",
                f"{comp['greedy_time']:.4f}"
            ]
        })
        
        st.dataframe(comparison_df, use_container_width=True)
        
        # Diferença de impacto
        impact_diff = bb_sol['total_impact'] - greedy_sol['total_impact']
        impact_diff_pct = (impact_diff / greedy_sol['total_impact'] * 100) if greedy_sol['total_impact'] > 0 else 0
        
        if impact_diff > 0:
            st.success(f"""
            ✅ **Branch and Bound encontrou solução {impact_diff:.2f}% melhor!**
            
            Isso representa uma melhoria de {impact_diff_pct:.1f}% em relação à heurística gulosa.
            """)
        elif impact_diff == 0:
            st.info("ℹ️ Ambos os algoritmos encontraram a mesma solução ótima.")
        else:
            st.warning("⚠️ Resultado inesperado - verifique a implementação.")
        
        # Gráfico de barras
        st.markdown("### 📊 Comparação Visual")
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Impacto Total (%)", "Tempo de Execução (s)")
        )
        
        fig.add_trace(
            go.Bar(x=['B&B', 'Heurística'], 
                   y=[bb_sol['total_impact'], greedy_sol['total_impact']],
                   name='Impacto',
                   marker_color=['#1f77b4', '#ff7f0e']),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Bar(x=['B&B', 'Heurística'], 
                   y=[comp['bb_time'], comp['greedy_time']],
                   name='Tempo',
                   marker_color=['#1f77b4', '#ff7f0e']),
            row=1, col=2
        )
        
        fig.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Projetos diferentes
        bb_ids = set(p.id for p in bb_sol['selected_projects'])
        greedy_ids = set(p.id for p in greedy_sol['selected_projects'])
        
        only_bb = bb_ids - greedy_ids
        only_greedy = greedy_ids - bb_ids
        
        if only_bb or only_greedy:
            st.markdown("### 🔍 Diferenças nas Seleções")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Apenas em B&B:**")
                if only_bb:
                    for proj in bb_sol['selected_projects']:
                        if proj.id in only_bb:
                            st.write(f"- {proj.name}")
                else:
                    st.write("(nenhum)")
            
            with col2:
                st.markdown("**Apenas em Heurística:**")
                if only_greedy:
                    for proj in greedy_sol['selected_projects']:
                        if proj.id in only_greedy:
                            st.write(f"- {proj.name}")
                else:
                    st.write("(nenhum)")


# ==============================================================================
# RODAPÉ
# ==============================================================================

st.sidebar.markdown("---")
st.sidebar.markdown("**Desenvolvido por:** Manus AI")
st.sidebar.markdown("**Disciplina:** Pesquisa Operacional")
st.sidebar.markdown("**Data:** 16/10/2025")

