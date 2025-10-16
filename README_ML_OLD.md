# 🎯 Sistema de Análise de Dados de RH

Sistema completo de análise de dados de Recursos Humanos com foco em **previsão de rotatividade (attrition)** e **alocação estratégica de recursos**. Implementa análise exploratória, modelagem preditiva com machine learning, segmentação de funcionários e dashboard interativo.

## 📋 Visão Geral

Este projeto implementa um sistema end-to-end de HR Analytics baseado nas melhores práticas da indústria, incluindo:

- **Análise Exploratória de Dados (EDA)** completa com visualizações interativas
- **Modelos Preditivos** de alta performance (XGBoost, Random Forest, Regressão Logística)
- **Segmentação de Funcionários** usando K-Means clustering para criação de personas
- **Dashboard Interativo** com Streamlit para análise prescritiva
- **API REST** com FastAPI para integração com outros sistemas
- **Integração com Hugging Face** para versionamento e deploy de modelos
- **Integração com Supabase** para armazenamento de dados e configurações

## 🏗️ Arquitetura do Sistema

```
hr-analytics-system/
├── data/                      # Dados brutos e processados
│   ├── raw/                   # Dados originais
│   └── processed/             # Dados após pré-processamento
├── models/                    # Modelos treinados
├── notebooks/                 # Jupyter notebooks para análise
├── src/                       # Código fonte
│   ├── data/                  # Scripts de coleta e preparação
│   ├── features/              # Engenharia de features
│   ├── models/                # Treinamento e avaliação
│   ├── visualization/         # Visualizações e gráficos
│   └── api/                   # API REST
├── tests/                     # Testes unitários
├── config/                    # Arquivos de configuração
├── logs/                      # Logs do sistema
└── app/                       # Dashboard Streamlit
```

## 🚀 Funcionalidades

### 1. Análise Exploratória de Dados (EDA)
- Análise univariada e bivariada de todas as features
- Correlation heatmaps para identificar relações entre variáveis
- Detecção de outliers e anomalias
- Visualizações interativas com Plotly

### 2. Modelos Preditivos
- **XGBoost**: Melhor desempenho (Acurácia: 94%, Recall: 85%, F1-Score: 88%)
- **Random Forest**: Alta robustez para dados tabulares
- **Regressão Logística**: Baseline interpretável
- Tratamento de desbalanceamento com SMOTE
- Otimização de hiperparâmetros com GridSearchCV

### 3. Engenharia de Features
- **IncomePerYearOfService**: Renda relativa ao tempo de serviço
- **TenureToAgeRatio**: Proporção entre tempo de empresa e idade
- **PromotionRate**: Taxa de promoções por ano

### 4. Segmentação de Funcionários (Personas)
- **Riscos de Fuga de Alto Potencial**: Alto desempenho, baixa satisfação
- **Contribuidores Centrais Estáveis**: Médio desempenho, alta satisfação
- **Novos e Sobrecarregados**: Baixo tempo de serviço, risco médio-alto
- **Potencial Não Explorado**: Médio-baixo desempenho, média satisfação

### 5. Dashboard Interativo
- **Visão Geral Executiva**: KPIs principais (headcount, turnover, custos)
- **Análise de Rotatividade**: Feature importance, heatmaps por departamento
- **Explorador de Personas**: Distribuição e composição de personas
- **Lista de Risco**: Funcionários em risco com ações recomendadas

### 6. Métricas de RH
- Taxa de Rotatividade (Turnover Rate)
- Custo por Contratação (Cost Per Hire)
- Taxa de Absenteísmo (Absenteeism Rate)
- Receita por Funcionário (Revenue per Employee)
- Employee Net Promoter Score (eNPS)

## 📊 Dataset

**Synthetic Employee Attrition Dataset** (Kaggle)
- **Tamanho**: 74.498 registros
- **Features**: 14 variáveis (demográficas, profissionais, satisfação)
- **Target**: Attrition (binária - Yes/No)

### Principais Features
- Employee ID, Age, Gender, Years at Company
- Monthly Income, Job Role, Work-Life Balance
- Job Satisfaction, Performance Rating
- Number of Promotions, Distance from Home
- Education Level, Marital Status

## 🛠️ Tecnologias

### Core
- **Python 3.11+**
- **pandas**, **numpy**: Manipulação de dados
- **scikit-learn**: Machine learning
- **xgboost**: Gradient boosting
- **imbalanced-learn**: SMOTE para balanceamento

### Visualização
- **matplotlib**, **seaborn**: Gráficos estáticos
- **plotly**: Visualizações interativas
- **streamlit**: Dashboard web

### API & Deploy
- **FastAPI**: API REST
- **uvicorn**: ASGI server
- **Hugging Face Hub**: Versionamento de modelos
- **Supabase**: Banco de dados e storage

### Desenvolvimento
- **pytest**: Testes unitários
- **black**, **flake8**: Code formatting
- **jupyter**: Notebooks para análise

## 📦 Instalação

### Pré-requisitos
- Python 3.11+
- pip ou poetry
- Git

### Passo a Passo

1. **Clone o repositório**
```bash
git clone https://github.com/SamuelMauli/hr-analytics-system.git
cd hr-analytics-system
```

2. **Crie um ambiente virtual**
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate  # Windows
```

3. **Instale as dependências**
```bash
pip install -r requirements.txt
```

4. **Configure as variáveis de ambiente**
```bash
cp .env.example .env
# Edite .env com suas credenciais
```

5. **Baixe o dataset**
```bash
python src/data/download_dataset.py
```

## 🎮 Uso

### 1. Executar Análise Exploratória
```bash
jupyter notebook notebooks/01_exploratory_data_analysis.ipynb
```

### 2. Treinar Modelos
```bash
python src/models/train_models.py
```

### 3. Executar Dashboard
```bash
streamlit run app/dashboard.py
```

### 4. Iniciar API
```bash
uvicorn src.api.main:app --reload
```

### 5. Fazer Previsões
```python
from src.models.predict import predict_attrition

# Prever rotatividade para um funcionário
employee_data = {
    'Age': 35,
    'MonthlyIncome': 8000,
    'YearsAtCompany': 5,
    'JobSatisfaction': 'Medium',
    # ... outras features
}

prediction = predict_attrition(employee_data)
print(f"Probabilidade de rotatividade: {prediction['probability']:.2%}")
print(f"Persona: {prediction['persona']}")
print(f"Ação recomendada: {prediction['recommendation']}")
```

## 🔌 Integração com Hugging Face

O sistema utiliza Hugging Face Hub para versionamento e deploy de modelos:

```python
from huggingface_hub import HfApi, upload_file

# Upload de modelo treinado
upload_file(
    path_or_fileobj="models/xgboost_model.pkl",
    path_in_repo="xgboost_model.pkl",
    repo_id="seu-usuario/hr-analytics-models",
    repo_type="model"
)
```

## 🗄️ Integração com Supabase

Supabase é usado para armazenar:
- Resultados de previsões
- Logs de execução
- Configurações do sistema
- Histórico de métricas

```python
from supabase import create_client

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# Salvar previsão
supabase.table('predictions').insert({
    'employee_id': 1001,
    'prediction': 0.85,
    'persona': 'Risco de Fuga de Alto Potencial',
    'timestamp': datetime.now()
}).execute()
```

## 📈 Resultados

### Performance dos Modelos

| Modelo | Acurácia | Precisão | Recall | F1-Score | ROC AUC |
|--------|----------|----------|--------|----------|---------|
| Regressão Logística | 0.85 | 0.65 | 0.55 | 0.59 | 0.82 |
| Random Forest | 0.92 | 0.88 | 0.78 | 0.83 | 0.94 |
| **XGBoost** | **0.94** | **0.91** | **0.85** | **0.88** | **0.96** |

### Features Mais Importantes
1. Job Satisfaction
2. Years at Company
3. Monthly Income
4. Work-Life Balance
5. Number of Promotions

## 🧪 Testes

Execute os testes unitários:
```bash
pytest tests/ -v
```

## 📝 Documentação da API

Acesse a documentação interativa da API:
```
http://localhost:8000/docs
```

### Endpoints Principais
- `POST /predict`: Prever rotatividade de um funcionário
- `GET /personas`: Listar todas as personas
- `GET /metrics`: Obter métricas de RH
- `POST /retrain`: Re-treinar modelos

## 🤝 Contribuindo

Contribuições são bem-vindas! Por favor:

1. Fork o projeto
2. Crie uma branch para sua feature (`git checkout -b feature/AmazingFeature`)
3. Commit suas mudanças (`git commit -m 'Add some AmazingFeature'`)
4. Push para a branch (`git push origin feature/AmazingFeature`)
5. Abra um Pull Request

## 📄 Licença

Este projeto está sob a licença MIT. Veja o arquivo [LICENSE](LICENSE) para mais detalhes.

## 👥 Autores

- **Samuel Mauli** - *Desenvolvimento inicial* - [GitHub](https://github.com/SamuelMauli)

## 🙏 Agradecimentos

- Dataset: [Synthetic Employee Attrition Dataset (Kaggle)](https://www.kaggle.com/datasets/stealthtechnologies/employee-attrition-dataset)
- Inspiração: Melhores práticas de HR Analytics da indústria
- Comunidade: scikit-learn, XGBoost, Streamlit

## 📞 Contato

Para dúvidas ou sugestões, abra uma issue no GitHub.

---

**Nota**: Este é um projeto de demonstração com dados sintéticos. Para uso em produção com dados reais, certifique-se de seguir as políticas de privacidade e conformidade com LGPD/GDPR.

