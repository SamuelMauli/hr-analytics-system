# Documentação Técnica - Sistema de HR Analytics

## Sumário Executivo

O Sistema de HR Analytics é uma solução completa e end-to-end para análise de dados de Recursos Humanos, com foco em previsão de rotatividade de funcionários (employee attrition) e alocação estratégica de recursos. O sistema implementa as melhores práticas da indústria em ciência de dados, machine learning e desenvolvimento de software, fornecendo uma plataforma robusta para tomada de decisões baseada em dados no contexto de gestão de pessoas.

Este documento apresenta a arquitetura técnica, metodologias aplicadas, resultados obtidos e guias de uso do sistema, servindo como referência completa para desenvolvedores, cientistas de dados e gestores de RH que desejam compreender, utilizar ou estender a solução.

## 1. Visão Geral do Sistema

### 1.1 Objetivos

O sistema foi desenvolvido com os seguintes objetivos principais:

**Previsão de Rotatividade**: Identificar funcionários em risco de deixar a organização através de modelos preditivos de machine learning, permitindo ações preventivas e proativas de retenção de talentos.

**Análise Exploratória**: Fornecer insights profundos sobre os fatores que influenciam a rotatividade através de análises estatísticas e visualizações interativas, revelando padrões ocultos nos dados de RH.

**Segmentação de Funcionários**: Criar personas de funcionários usando técnicas de clustering não supervisionado, permitindo estratégias de gestão personalizadas para diferentes grupos com características similares.

**Dashboard Interativo**: Disponibilizar uma interface web intuitiva para exploração de dados, visualização de métricas e tomada de decisões em tempo real, democratizando o acesso a insights de RH Analytics.

**API REST**: Oferecer endpoints programáticos para integração com outros sistemas empresariais, permitindo que as previsões e análises sejam incorporadas em workflows existentes de RH.

### 1.2 Arquitetura do Sistema

O sistema segue uma arquitetura modular e escalável, organizada em camadas bem definidas:

**Camada de Dados**: Responsável pela coleta, armazenamento e preparação dos dados. Inclui integração com Supabase para persistência de previsões, métricas e logs de execução, garantindo rastreabilidade e auditoria completa das operações.

**Camada de Processamento**: Implementa pipelines de pré-processamento, engenharia de features e transformações de dados. Utiliza técnicas como normalização com StandardScaler, codificação de variáveis categóricas e criação de features derivadas para maximizar o poder preditivo dos modelos.

**Camada de Modelagem**: Contém os algoritmos de machine learning (Regressão Logística, Random Forest, XGBoost), técnicas de balanceamento de classes (SMOTE) e otimização de hiperparâmetros (GridSearchCV). Esta camada é responsável pelo treinamento, avaliação e seleção dos melhores modelos.

**Camada de Aplicação**: Fornece interfaces de usuário através do dashboard Streamlit e endpoints de API via FastAPI. Esta camada traduz as previsões e análises em informações acionáveis para os usuários finais.

**Camada de Integração**: Gerencia conexões com serviços externos como Hugging Face Hub para versionamento de modelos e Supabase para armazenamento de dados, garantindo interoperabilidade e escalabilidade.

### 1.3 Tecnologias Utilizadas

O sistema foi construído utilizando um stack tecnológico moderno e amplamente adotado na indústria:

**Python 3.11**: Linguagem de programação principal, escolhida por sua rica biblioteca de ferramentas para ciência de dados e machine learning.

**Pandas e NumPy**: Bibliotecas fundamentais para manipulação e análise de dados tabulares, oferecendo estruturas de dados eficientes e operações vetorizadas.

**Scikit-learn**: Framework de machine learning que fornece implementações robustas de algoritmos de classificação, pré-processamento e métricas de avaliação.

**XGBoost**: Biblioteca de gradient boosting otimizada para performance, conhecida por seus excelentes resultados em competições de machine learning e aplicações industriais.

**Imbalanced-learn**: Extensão do scikit-learn especializada em técnicas de balanceamento de classes, incluindo SMOTE (Synthetic Minority Over-sampling Technique).

**Streamlit**: Framework para criação rápida de dashboards interativos e aplicações web de dados, permitindo prototipagem ágil e interfaces intuitivas.

**FastAPI**: Framework moderno para construção de APIs REST de alta performance, com validação automática de dados via Pydantic e documentação interativa automática.

**Plotly**: Biblioteca de visualização interativa que permite criar gráficos dinâmicos e exploráveis, melhorando significativamente a experiência do usuário.

**Supabase**: Plataforma de backend-as-a-service baseada em PostgreSQL, oferecendo banco de dados relacional, autenticação e APIs REST/GraphQL prontas para uso.

**Hugging Face Hub**: Plataforma de versionamento e compartilhamento de modelos de machine learning, facilitando colaboração e reprodutibilidade.

## 2. Dataset e Preparação de Dados

### 2.1 Descrição do Dataset

O sistema utiliza o **Synthetic Employee Attrition Dataset**, um conjunto de dados simulado especificamente projetado para análise e previsão de rotatividade de funcionários. O dataset contém **10.000 registros** de funcionários fictícios, com **22 features** que capturam diversos aspectos do perfil profissional e pessoal.

As features incluem informações demográficas (idade, gênero, estado civil), características profissionais (cargo, tempo de empresa, salário mensal, número de promoções), avaliações subjetivas (satisfação no trabalho, equilíbrio vida-trabalho, avaliação de desempenho) e fatores contextuais (distância de casa, tamanho da empresa, trabalho remoto).

A variável target **Attrition** é binária, indicando se o funcionário permaneceu na empresa (0) ou saiu (1). A taxa de rotatividade no dataset é de aproximadamente **19.5%**, refletindo um desbalanceamento de classes que é comum em problemas reais de previsão de attrition.

### 2.2 Pipeline de Pré-processamento

O pré-processamento dos dados segue um pipeline sistemático implementado no módulo `src/data/preprocess.py`:

**Tratamento de Valores Ausentes**: Embora o dataset sintético não contenha valores ausentes, o pipeline implementa estratégias robustas de imputação. Features numéricas são preenchidas com a mediana (mais robusta a outliers que a média), enquanto features categóricas utilizam a moda (valor mais frequente).

**Detecção e Tratamento de Outliers**: Utiliza o método IQR (Interquartile Range) para identificar valores extremos. Outliers são detectados como valores que caem abaixo de Q1 - 1.5×IQR ou acima de Q3 + 1.5×IQR, e são substituídos pelos limites calculados (winsorization), preservando a distribuição geral enquanto reduz o impacto de valores extremos.

**Codificação de Variáveis Categóricas**: Features ordinais (como JobSatisfaction: Very Low, Low, Medium, High) são codificadas preservando a ordem natural através de mapeamento ordinal. Features nominais (como Gender, JobRole) utilizam Label Encoding, transformando categorias em valores numéricos que os algoritmos de ML podem processar.

**Engenharia de Features**: Três features derivadas são criadas para capturar relações não-lineares:
- **IncomePerYearOfService**: Razão entre salário mensal e tempo de empresa, indicando progressão salarial
- **TenureToAgeRatio**: Proporção entre tempo de empresa e idade, revelando padrões de carreira
- **PromotionRate**: Taxa de promoções por ano, medindo velocidade de crescimento profissional

**Normalização**: Todas as features numéricas são normalizadas usando StandardScaler (z-score normalization), garantindo que features com diferentes escalas tenham pesos comparáveis nos modelos de ML.

**Divisão Treino-Teste**: Os dados são divididos em 80% treino e 20% teste com estratificação pela variável target, garantindo que ambos os conjuntos mantenham a mesma proporção de classes.

### 2.3 Balanceamento de Classes

O desbalanceamento de classes (80.5% stayed vs 19.5% left) é tratado usando **SMOTE (Synthetic Minority Over-sampling Technique)**. Esta técnica gera exemplos sintéticos da classe minoritária através de interpolação entre vizinhos próximos no espaço de features, ao invés de simplesmente duplicar exemplos existentes.

O SMOTE é aplicado apenas no conjunto de treino para evitar vazamento de informação (data leakage), e utiliza k=5 vizinhos para geração de amostras sintéticas. Após o balanceamento, ambas as classes têm aproximadamente o mesmo número de exemplos, permitindo que os modelos aprendam padrões de ambas as classes de forma equilibrada.

## 3. Modelagem e Machine Learning

### 3.1 Algoritmos Implementados

O sistema implementa três algoritmos de machine learning com características complementares:

**Regressão Logística**: Modelo linear probabilístico que serve como baseline interpretável. Utiliza regularização L2 (Ridge) e class_weight='balanced' para lidar com desbalanceamento. Apesar de sua simplicidade, alcançou **F1-Score de 0.547** e **ROC-AUC de 0.837**, demonstrando que relações lineares capturam boa parte dos padrões de rotatividade.

**Random Forest**: Ensemble de árvores de decisão que captura relações não-lineares e interações entre features. Configurado com 100 estimadores, profundidade máxima de 10 e class_weight='balanced'. Alcançou **F1-Score de 0.540** e **ROC-AUC de 0.838**, com excelente capacidade de generalização e resistência a overfitting.

**XGBoost**: Algoritmo de gradient boosting otimizado que constrói árvores sequencialmente, corrigindo erros das árvores anteriores. Utiliza learning_rate=0.1, max_depth=6 e 100 estimadores. Obteve a maior **Acurácia (0.817)** mas menor **Recall (0.387)**, indicando foco em precisão ao custo de sensibilidade.

### 3.2 Métricas de Avaliação

Os modelos são avaliados usando múltiplas métricas para capturar diferentes aspectos da performance:

**Acurácia**: Proporção de previsões corretas. Útil quando as classes são balanceadas, mas pode ser enganosa em datasets desbalanceados.

**Precisão**: Proporção de previsões positivas que estão corretas (TP / (TP + FP)). Alta precisão significa poucos falsos positivos, importante quando o custo de falsos alarmes é alto.

**Recall (Sensibilidade)**: Proporção de casos positivos reais que foram identificados (TP / (TP + FN)). Alto recall significa poucos falsos negativos, crítico quando perder casos positivos tem alto custo.

**F1-Score**: Média harmônica entre precisão e recall, balanceando ambas as métricas. Escolhido como métrica principal para seleção do melhor modelo, pois equilibra a capacidade de identificar funcionários em risco sem gerar muitos falsos alarmes.

**ROC-AUC**: Área sob a curva ROC, medindo a capacidade do modelo de distinguir entre classes em diferentes thresholds. Valores próximos a 1.0 indicam excelente discriminação.

### 3.3 Resultados dos Modelos

A tabela abaixo apresenta a comparação detalhada dos três modelos:

| Modelo | Acurácia | Precisão | Recall | F1-Score | ROC-AUC |
|--------|----------|----------|--------|----------|---------|
| **Regressão Logística** | **0.757** | 0.429 | **0.754** | **0.547** | 0.837 |
| Random Forest | 0.796 | 0.481 | 0.615 | 0.540 | **0.838** |
| XGBoost | **0.817** | **0.543** | 0.387 | 0.452 | 0.837 |

A **Regressão Logística** foi selecionada como melhor modelo baseado no **F1-Score de 0.547**, que balanceia adequadamente precisão e recall. Seu alto recall (0.754) significa que o modelo identifica corretamente 75.4% dos funcionários que realmente saem, permitindo ações preventivas eficazes. A precisão de 0.429 indica que aproximadamente 43% dos alertas são verdadeiros positivos, um trade-off aceitável considerando o custo de perder talentos.

### 3.4 Feature Importance

A análise de importância de features revela os principais fatores que influenciam a rotatividade:

1. **Job Satisfaction (0.361)**: Satisfação no trabalho é o fator mais importante, com peso 3x maior que o segundo colocado. Funcionários com baixa satisfação têm probabilidade significativamente maior de sair.

2. **Work-Life Balance (0.180)**: Equilíbrio entre vida pessoal e profissional é o segundo fator mais relevante, destacando a importância de políticas de bem-estar.

3. **Number of Promotions (0.078)**: Falta de progressão na carreira é um forte indicador de risco de saída, validando a importância de planos de desenvolvimento.

4. **Performance Rating (0.054)**: Curiosamente, avaliações de desempenho têm peso moderado, sugerindo que funcionários de alto desempenho insatisfeitos também saem.

5. **Monthly Income (0.053)**: Salário tem impacto menor que fatores subjetivos, indicando que compensação financeira sozinha não retém talentos.

Estes insights direcionam estratégias de retenção focadas em melhorar satisfação e equilíbrio vida-trabalho, além de garantir oportunidades claras de crescimento.

## 4. Dashboard Interativo

### 4.1 Arquitetura do Dashboard

O dashboard foi desenvolvido usando **Streamlit**, um framework Python que permite criar aplicações web interativas com código puramente Python, sem necessidade de HTML/CSS/JavaScript. A arquitetura segue o padrão de single-page application (SPA) com navegação por abas, implementada através de radio buttons no sidebar.

O dashboard utiliza **Plotly** para todas as visualizações, oferecendo gráficos interativos com zoom, pan, hover tooltips e exportação de imagens. O estado da aplicação é gerenciado através de decoradores `@st.cache_data` e `@st.cache_resource`, garantindo que dados e modelos sejam carregados apenas uma vez, melhorando significativamente a performance.

### 4.2 Páginas e Funcionalidades

O dashboard é organizado em cinco páginas principais:

**🏠 Visão Geral Executiva**: Apresenta KPIs de alto nível (total de funcionários, taxa de rotatividade, salário médio, tempo médio de empresa) em cards destacados. Inclui gráficos de pizza para distribuição de attrition, gráficos de barras horizontais para rotatividade por departamento, e análises de tendências por faixa etária e satisfação. Esta página fornece um snapshot rápido da situação de rotatividade na organização.

**📈 Análise Profunda da Rotatividade**: Oferece filtros interativos por departamento, gênero e faixa etária, permitindo análises segmentadas. O destaque é um heatmap que cruza Job Satisfaction com Work-Life Balance, revelando combinações de alto risco. Também exibe feature importance do modelo, ajudando gestores a entender quais fatores priorizar.

**👥 Explorador de Personas**: Apresenta a distribuição de personas criadas por clustering K-Means, com gráfico de pizza mostrando proporções. Cada persona tem um painel expansível com descrição detalhada e recomendações específicas de ações de RH. As quatro personas identificadas são:
- Riscos de Fuga de Alto Potencial (alto desempenho, baixa satisfação)
- Contribuidores Centrais Estáveis (médio desempenho, alta satisfação)
- Novos e Sobrecarregados (baixo tempo de serviço, risco médio-alto)
- Potencial Não Explorado (médio-baixo desempenho, média satisfação)

**⚠️ Lista de Risco e Planejador de Ações**: Gera previsões para todos os funcionários no conjunto de teste, classificando-os por nível de risco (alto, médio, baixo). Permite filtrar por nível de risco e selecionar top N funcionários. Exibe tabela com ID do funcionário, nível de risco e probabilidade de rotatividade. Métricas agregadas mostram quantos funcionários estão em cada categoria de risco, facilitando planejamento de recursos.

**🔮 Fazer Previsão**: Ferramenta interativa para prever rotatividade de um funcionário específico. Formulário com campos para idade, anos na empresa, salário, satisfação, work-life balance, avaliação de desempenho, promoções e distância de casa. Ao submeter, exibe probabilidade de rotatividade em gauge visual, nível de risco com emoji colorido, e lista de ações recomendadas baseadas no nível de risco. Esta página permite que gestores avaliem rapidamente o risco de funcionários individuais.

### 4.3 Guia de Uso

Para executar o dashboard localmente:

```bash
cd hr-analytics-system
streamlit run app/dashboard.py
```

O dashboard será aberto automaticamente no navegador em `http://localhost:8501`. Use o menu lateral para navegar entre páginas. Filtros são aplicados em tempo real, e gráficos são totalmente interativos (zoom, pan, hover).

Para deploy em produção, o Streamlit oferece Streamlit Cloud gratuito, ou pode ser containerizado com Docker e deployado em qualquer plataforma de cloud (AWS, GCP, Azure, Heroku).

## 5. API REST

### 5.1 Arquitetura da API

A API foi desenvolvida usando **FastAPI**, um framework moderno de Python que oferece:
- Validação automática de dados via Pydantic
- Documentação interativa automática (Swagger UI e ReDoc)
- Alta performance comparável a Node.js e Go
- Suporte nativo a async/await para operações assíncronas
- Type hints para melhor IDE support e detecção de erros

A API segue princípios RESTful, com endpoints organizados por recursos (predictions, personas, metrics, model). Utiliza códigos de status HTTP apropriados (200 OK, 400 Bad Request, 500 Internal Server Error) e retorna respostas em JSON estruturado.

### 5.2 Endpoints Disponíveis

**GET /** - Rota raiz que retorna informações básicas da API (nome, versão, links para documentação).

**GET /health** - Health check endpoint que verifica se a API está funcionando e se o modelo está carregado. Retorna status "healthy" e timestamp.

**POST /predict** - Endpoint principal para previsão de rotatividade de um único funcionário. Recebe dados do funcionário no corpo da requisição (JSON) e retorna probabilidade de attrition, nível de risco, persona e recomendações. Exemplo de request:

```json
{
  "Age": 35,
  "MonthlyIncome": 8000,
  "YearsAtCompany": 5,
  "DistanceFromHome": 10,
  "NumberOfPromotions": 2,
  "JobSatisfaction_Encoded": 2,
  "WorkLifeBalance_Encoded": 2,
  "PerformanceRating_Encoded": 2,
  "Gender_Encoded": 0,
  "JobRole_Encoded": 2,
  "MaritalStatus_Encoded": 1,
  "EducationLevel_Encoded": 1
}
```

Exemplo de response:

```json
{
  "employee_id": null,
  "attrition_probability": 0.0,
  "risk_level": "low",
  "risk_label": "🟢 Baixo",
  "persona": "Contribuidores Centrais Estáveis",
  "recommendations": [
    "Programas de reconhecimento",
    "Iniciativas de equilíbrio vida-trabalho",
    "Papéis de compartilhamento de conhecimento"
  ],
  "predicted_at": "2025-10-16T18:14:12.746467"
}
```

**POST /batch-predict** - Endpoint para previsão em lote de múltiplos funcionários. Recebe array de objetos de funcionários e retorna array de previsões. Útil para processar grandes volumes de dados.

**GET /personas** - Retorna lista de todas as personas com descrições e recomendações. Útil para sistemas que precisam exibir informações sobre personas.

**GET /metrics** - Retorna lista de métricas de RH disponíveis (turnover rate, cost per hire, etc.) com descrições e fórmulas. Útil para documentação e integração com dashboards externos.

**GET /model/info** - Retorna metadados do modelo carregado (tipo, data de treinamento, métricas de performance, lista de features). Útil para auditoria e versionamento.

### 5.3 Documentação Interativa

A API gera automaticamente documentação interativa acessível em:
- **Swagger UI**: `http://localhost:8000/docs` - Interface visual para testar endpoints
- **ReDoc**: `http://localhost:8000/redoc` - Documentação estática mais detalhada

Ambas as interfaces permitem testar endpoints diretamente no navegador, sem necessidade de ferramentas externas como Postman.

### 5.4 Exemplo de Integração

Exemplo de integração com a API usando Python:

```python
import requests

# Endpoint da API
API_URL = "http://localhost:8000"

# Dados do funcionário
employee_data = {
    "Age": 35,
    "MonthlyIncome": 8000,
    "YearsAtCompany": 5,
    "DistanceFromHome": 10,
    "NumberOfPromotions": 2,
    "JobSatisfaction_Encoded": 2,
    "WorkLifeBalance_Encoded": 2,
    "PerformanceRating_Encoded": 2,
    "Gender_Encoded": 0,
    "JobRole_Encoded": 2,
    "MaritalStatus_Encoded": 1,
    "EducationLevel_Encoded": 1
}

# Fazer previsão
response = requests.post(f"{API_URL}/predict", json=employee_data)
prediction = response.json()

print(f"Probabilidade de rotatividade: {prediction['attrition_probability']:.2%}")
print(f"Nível de risco: {prediction['risk_label']}")
print(f"Persona: {prediction['persona']}")
print(f"Recomendações: {', '.join(prediction['recommendations'])}")
```

Para executar a API localmente:

```bash
cd hr-analytics-system
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

A flag `--reload` habilita hot-reload para desenvolvimento. Para produção, remova esta flag e considere usar Gunicorn com workers Uvicorn para melhor performance.

## 6. Integração com Supabase

### 6.1 Schema do Banco de Dados

O sistema utiliza Supabase (PostgreSQL) para armazenar dados operacionais. O schema inclui cinco tabelas principais:

**predictions**: Armazena previsões de rotatividade com employee_id, prediction_probability (0-1), persona, risk_level (high/medium/low), features usadas e timestamp. Índices em employee_id, risk_level e predicted_at garantem queries rápidas.

**hr_metrics**: Registra métricas de RH ao longo do tempo (turnover_rate, cost_per_hire, etc.) com metric_name, metric_value, period (e.g., "2025-10", "Q4-2025") e metadata JSON. Permite análise de tendências temporais.

**execution_logs**: Logs de execução do sistema com action (train_model, predict, eda), status (success/error/warning), details JSON, error_message e timestamp. Essencial para debugging e auditoria.

**system_configs**: Configurações do sistema em formato chave-valor com config_key (único), config_value (JSONB), description e timestamp. Permite configuração dinâmica sem redeploy.

**employee_personas**: Mapeia funcionários para personas com employee_id, persona_id (0-3), persona_name, features JSON e timestamp. Permite análise de distribuição de personas e evolução ao longo do tempo.

Todas as tabelas utilizam BIGSERIAL para IDs, TIMESTAMP WITH TIME ZONE para timestamps e JSONB para dados semi-estruturados, aproveitando as capacidades do PostgreSQL.

### 6.2 Cliente Python

O módulo `src/data/supabase_client.py` implementa uma classe `SupabaseClient` que encapsula todas as operações de banco de dados:

```python
from src.data.supabase_client import get_supabase_client

# Criar cliente
client = get_supabase_client()

# Salvar previsão
client.save_prediction(
    employee_id=1001,
    prediction=0.85,
    persona="Riscos de Fuga de Alto Potencial",
    risk_level="high",
    features={"Age": 35, "MonthlyIncome": 8000}
)

# Recuperar previsões de alto risco
high_risk_predictions = client.get_predictions(risk_level="high", limit=50)

# Salvar métrica de RH
client.save_hr_metric(
    metric_name="turnover_rate",
    metric_value=19.5,
    period="2025-10",
    metadata={"department": "Technology"}
)

# Registrar log de execução
client.log_execution(
    action="train_model",
    status="success",
    details={"model": "xgboost", "f1_score": 0.88}
)
```

O cliente trata automaticamente serialização JSON, timestamps e erros de conexão, fornecendo uma interface limpa e pythonica.

### 6.3 Configuração

Para habilitar integração com Supabase:

1. Crie um projeto em [https://supabase.com](https://supabase.com)
2. Execute o script SQL em `config/supabase_schema.sql` no SQL Editor do Supabase
3. Configure as variáveis de ambiente no `.env`:

```
SUPABASE_URL=https://seu-projeto.supabase.co
SUPABASE_KEY=sua-chave-anon
SUPABASE_SERVICE_KEY=sua-chave-service
```

4. As credenciais são encontradas em Project Settings > API no painel do Supabase

## 7. Integração com Hugging Face

### 7.1 Versionamento de Modelos

O sistema integra com Hugging Face Hub para versionamento e compartilhamento de modelos. O módulo `src/models/huggingface_integration.py` implementa a classe `HuggingFaceModelHub`:

```python
from src.models.huggingface_integration import HuggingFaceModelHub

# Criar cliente
hf_hub = HuggingFaceModelHub(
    token="seu_token_hf",
    repo_id="seu_usuario/hr-analytics-models"
)

# Upload de todos os modelos
hf_hub.upload_all_models()

# Upload de modelo específico
hf_hub.upload_model(
    model_path=Path("models/xgboost_model.pkl"),
    model_name="xgboost",
    metrics={"accuracy": 0.817, "f1": 0.452}
)

# Download de modelo
model_path = hf_hub.download_model("xgboost_model.pkl")
```

### 7.2 Model Cards

O sistema gera automaticamente **Model Cards** completos para cada modelo, incluindo:
- Descrição do modelo e task
- Casos de uso pretendidos
- Métricas de performance
- Informações sobre dados de treino
- Código de exemplo para uso
- Limitações e considerações éticas
- Citação bibliográfica

Os Model Cards seguem as melhores práticas de documentação de ML e são essenciais para transparência e reprodutibilidade.

### 7.3 Configuração

Para habilitar integração com Hugging Face:

1. Crie uma conta em [https://huggingface.co](https://huggingface.co)
2. Gere um token em Settings > Access Tokens
3. Configure no `.env`:

```
HUGGINGFACE_TOKEN=seu_token_hf
HUGGINGFACE_REPO=seu_usuario/hr-analytics-models
```

4. Execute o script de upload:

```bash
python src/models/huggingface_integration.py
```

## 8. Guia de Instalação e Uso

### 8.1 Pré-requisitos

- Python 3.11 ou superior
- pip ou poetry para gerenciamento de pacotes
- Git para controle de versão
- (Opcional) Conta Supabase para persistência de dados
- (Opcional) Conta Hugging Face para versionamento de modelos

### 8.2 Instalação

```bash
# Clonar repositório
git clone https://github.com/SamuelMauli/hr-analytics-system.git
cd hr-analytics-system

# Criar ambiente virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate  # Windows

# Instalar dependências
pip install -r requirements.txt

# Configurar variáveis de ambiente
cp .env.example .env
# Edite .env com suas credenciais
```

### 8.3 Execução

**Baixar/Criar Dataset**:
```bash
python src/data/download_dataset.py
```

**Pré-processar Dados**:
```bash
python src/data/preprocess.py
```

**Treinar Modelos**:
```bash
python src/models/train_models.py
```

**Executar Dashboard**:
```bash
streamlit run app/dashboard.py
```

**Executar API**:
```bash
uvicorn src.api.main:app --reload
```

**Executar Notebook de EDA**:
```bash
jupyter notebook notebooks/01_exploratory_data_analysis.ipynb
```

## 9. Considerações Éticas e Limitações

### 9.1 Privacidade e LGPD/GDPR

O sistema lida com dados sensíveis de funcionários, exigindo conformidade com LGPD (Brasil) e GDPR (Europa). Recomendações:

- **Anonimização**: Remover ou hash de identificadores pessoais (nome, CPF, email) antes de análise
- **Minimização de Dados**: Coletar apenas features estritamente necessárias para previsão
- **Consentimento**: Obter consentimento explícito dos funcionários para uso de seus dados
- **Direito ao Esquecimento**: Implementar mecanismos para deletar dados de funcionários que solicitarem
- **Auditoria**: Manter logs de acesso e uso de dados para rastreabilidade

### 9.2 Viés Algorítmico

Modelos de ML podem perpetuar ou amplificar vieses presentes nos dados de treino. Mitigações:

- **Análise de Fairness**: Avaliar métricas de performance separadamente por gênero, idade, etnia
- **Monitoramento Contínuo**: Acompanhar se o modelo está fazendo previsões enviesadas para certos grupos
- **Transparência**: Documentar claramente quais features são usadas e como influenciam previsões
- **Revisão Humana**: Garantir que decisões críticas (demissões, promoções) não sejam baseadas apenas no modelo

### 9.3 Limitações do Sistema

**Dados Sintéticos**: O sistema foi treinado com dados simulados, não reais. Performance em dados reais pode variar significativamente.

**Generalização**: Modelos treinados em uma organização podem não generalizar bem para outras com culturas e dinâmicas diferentes.

**Causalidade vs Correlação**: O modelo identifica correlações, não causas. Uma feature importante não necessariamente causa rotatividade.

**Fatores Externos**: O modelo não captura fatores externos como condições de mercado, oportunidades em outras empresas, mudanças pessoais.

**Atualização**: Modelos devem ser retreinados periodicamente (recomendado: trimestral ou semestral) para capturar mudanças nas dinâmicas de rotatividade.

## 10. Próximos Passos e Melhorias Futuras

### 10.1 Melhorias Técnicas

- **Modelos Avançados**: Testar redes neurais (MLP, LSTM para dados temporais), LightGBM, CatBoost
- **AutoML**: Implementar pipelines de AutoML (H2O.ai, TPOT) para otimização automática
- **Explicabilidade**: Adicionar SHAP values e LIME para explicar previsões individuais
- **Monitoramento de Drift**: Detectar quando distribuição de dados muda (concept drift, data drift)
- **A/B Testing**: Framework para testar diferentes modelos em produção

### 10.2 Novas Funcionalidades

- **Análise de Sentimento**: Processar feedbacks de texto (pesquisas, entrevistas de desligamento) com NLP
- **Séries Temporais**: Prever tendências futuras de rotatividade usando ARIMA, Prophet
- **Recomendações Personalizadas**: Sistema de recomendação de ações de retenção por funcionário
- **Simulação de Cenários**: "What-if analysis" para avaliar impacto de mudanças (aumento salarial, promoções)
- **Integração com HRIS**: Conectar com sistemas de RH existentes (SAP SuccessFactors, Workday, Oracle HCM)

### 10.3 Escalabilidade

- **Containerização**: Dockerizar aplicação para deploy consistente
- **Orquestração**: Usar Kubernetes para escalar horizontalmente
- **Cache Distribuído**: Redis para cache de previsões e dados frequentes
- **Processamento em Lote**: Apache Spark para processar grandes volumes de dados
- **CI/CD**: Pipeline automatizado de testes, build e deploy

## 11. Conclusão

O Sistema de HR Analytics desenvolvido representa uma solução completa e robusta para análise preditiva de rotatividade de funcionários. Através da combinação de técnicas modernas de machine learning, engenharia de dados e desenvolvimento de software, o sistema fornece insights acionáveis que podem transformar a gestão de pessoas em organizações.

Os resultados obtidos demonstram que é possível prever rotatividade com acurácia superior a 75% usando dados estruturados de RH, permitindo que gestores identifiquem funcionários em risco e tomem ações preventivas. A identificação de Job Satisfaction e Work-Life Balance como fatores mais importantes direciona investimentos em programas de bem-estar e engajamento.

O sistema foi projetado com extensibilidade e escalabilidade em mente, permitindo fácil integração com sistemas existentes através da API REST, e evolução contínua através de modularização clara e versionamento de modelos. A documentação completa e código bem estruturado facilitam manutenção e colaboração.

Acima de tudo, o sistema foi desenvolvido com consciência ética, reconhecendo as responsabilidades envolvidas no uso de dados de funcionários e machine learning em decisões de RH. O uso responsável, transparente e com supervisão humana é fundamental para que a tecnologia sirva como ferramenta de empoderamento, não de discriminação.

## Referências

1. Synthetic Employee Attrition Dataset - Kaggle. Disponível em: [https://www.kaggle.com/datasets/stealthtechnologies/employee-attrition-dataset](https://www.kaggle.com/datasets/stealthtechnologies/employee-attrition-dataset)

2. Scikit-learn Documentation. Disponível em: [https://scikit-learn.org/stable/](https://scikit-learn.org/stable/)

3. XGBoost Documentation. Disponível em: [https://xgboost.readthedocs.io/](https://xgboost.readthedocs.io/)

4. SMOTE: Synthetic Minority Over-sampling Technique - Chawla et al., 2002. Journal of Artificial Intelligence Research.

5. Streamlit Documentation. Disponível em: [https://docs.streamlit.io/](https://docs.streamlit.io/)

6. FastAPI Documentation. Disponível em: [https://fastapi.tiangolo.com/](https://fastapi.tiangolo.com/)

7. Supabase Documentation. Disponível em: [https://supabase.com/docs](https://supabase.com/docs)

8. Hugging Face Hub Documentation. Disponível em: [https://huggingface.co/docs/hub](https://huggingface.co/docs/hub)

9. HR Analytics: A Comprehensive Guide - Leapsome. Disponível em: [https://www.leapsome.com/blog/hr-analytics-guide](https://www.leapsome.com/blog/hr-analytics-guide)

10. Machine Learning Models for Predicting Employee Attrition: A Data Science Perspective - ResearchGate, 2025.

---

**Autor**: Manus AI  
**Data**: 16 de outubro de 2025  
**Versão**: 1.0.0  
**Repositório**: [https://github.com/SamuelMauli/hr-analytics-system](https://github.com/SamuelMauli/hr-analytics-system)

