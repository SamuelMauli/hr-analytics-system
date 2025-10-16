# Guia Rápido de Início - HR Analytics System

## 🚀 Início Rápido em 5 Minutos

Este guia permite que você execute o sistema de HR Analytics em poucos minutos.

### Pré-requisitos

- Python 3.11 ou superior instalado
- Git instalado
- 2GB de espaço em disco

### Passo 1: Clonar o Repositório

```bash
git clone https://github.com/SamuelMauli/hr-analytics-system.git
cd hr-analytics-system
```

### Passo 2: Instalar Dependências

```bash
# Criar ambiente virtual (recomendado)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate  # Windows

# Instalar pacotes
pip install -r requirements.txt
```

### Passo 3: Preparar Dados e Treinar Modelos

```bash
# Criar dataset sintético
python src/data/download_dataset.py

# Pré-processar dados
python src/data/preprocess.py

# Treinar modelos (leva ~2-3 minutos)
python src/models/train_models.py
```

### Passo 4: Executar o Dashboard

```bash
streamlit run app/dashboard.py
```

O dashboard abrirá automaticamente em `http://localhost:8501`

### Passo 5 (Opcional): Executar a API

Em outro terminal:

```bash
uvicorn src.api.main:app --reload
```

A API estará disponível em `http://localhost:8000`
- Documentação interativa: `http://localhost:8000/docs`

## 📊 O que você pode fazer agora?

### No Dashboard:

1. **Visão Geral**: Veja KPIs de rotatividade, distribuições e tendências
2. **Análise Profunda**: Explore heatmaps e feature importance
3. **Explorador de Personas**: Conheça os 4 tipos de funcionários identificados
4. **Lista de Risco**: Identifique funcionários em risco de sair
5. **Fazer Previsão**: Teste previsões para funcionários individuais

### Na API:

Teste o endpoint de previsão:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
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
  }'
```

## 🔧 Configurações Opcionais

### Integração com Supabase

1. Crie uma conta em [https://supabase.com](https://supabase.com)
2. Crie um novo projeto
3. Execute o SQL em `config/supabase_schema.sql` no SQL Editor
4. Copie `.env.example` para `.env` e adicione suas credenciais:

```
SUPABASE_URL=https://seu-projeto.supabase.co
SUPABASE_KEY=sua-chave-anon
```

### Integração com Hugging Face

1. Crie uma conta em [https://huggingface.co](https://huggingface.co)
2. Gere um token em Settings > Access Tokens
3. Adicione ao `.env`:

```
HUGGINGFACE_TOKEN=seu_token
HUGGINGFACE_REPO=seu_usuario/hr-analytics-models
```

4. Execute o upload:

```bash
python src/models/huggingface_integration.py
```

## 📚 Próximos Passos

- Leia a [Documentação Completa](DOCUMENTATION.md) para entender a arquitetura
- Explore o [Notebook de EDA](notebooks/01_exploratory_data_analysis.ipynb)
- Customize os modelos em `src/models/train_models.py`
- Adapte o dashboard em `app/dashboard.py`
- Estenda a API em `src/api/main.py`

## 🆘 Problemas Comuns

### Erro: "Module not found"
```bash
pip install -r requirements.txt
```

### Erro: "Model not found"
```bash
python src/models/train_models.py
```

### Erro: "Dataset not found"
```bash
python src/data/download_dataset.py
python src/data/preprocess.py
```

### Dashboard não abre
Verifique se a porta 8501 está livre:
```bash
lsof -i :8501  # Linux/Mac
netstat -ano | findstr :8501  # Windows
```

### API não responde
Verifique se a porta 8000 está livre:
```bash
lsof -i :8000  # Linux/Mac
netstat -ano | findstr :8000  # Windows
```

## 📞 Suporte

- **Issues**: [https://github.com/SamuelMauli/hr-analytics-system/issues](https://github.com/SamuelMauli/hr-analytics-system/issues)
- **Documentação**: [DOCUMENTATION.md](DOCUMENTATION.md)
- **README**: [README.md](README.md)

## 📄 Licença

Este projeto está sob a licença MIT. Veja [LICENSE](LICENSE) para mais detalhes.

---

**Desenvolvido com ❤️ por Manus AI**

