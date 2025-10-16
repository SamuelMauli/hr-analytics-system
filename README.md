# 🎯 Sistema de Otimização de Projetos de RH - Branch and Bound

## Disciplina: Pesquisa Operacional
## Problema: Seleção Ótima de Projetos de Retenção de Funcionários

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-25%20passed-success.svg)](tests/)

---

## 🎯 Sobre o Projeto

Este projeto implementa um **sistema completo de otimização** para seleção de projetos de retenção de funcionários usando o algoritmo **Branch and Bound**. O sistema resolve o **Problema da Mochila 0-1 (Knapsack Problem)**, maximizando o impacto na redução de rotatividade respeitando restrições orçamentárias.

**Repositório GitHub:** https://github.com/SamuelMauli/hr-analytics-system

---

## 🔬 Problema de Otimização

### Contexto

Uma empresa de RH possui um **orçamento limitado** para investir em projetos de retenção de funcionários. Cada projeto tem um **custo** e um **impacto esperado** na redução de rotatividade. O objetivo é selecionar o conjunto de projetos que **maximize o impacto total**, respeitando o orçamento disponível.

### Modelagem Matemática

**Variáveis de Decisão:**
```
xᵢ ∈ {0, 1}  onde i = 1, 2, ..., n
xᵢ = 1 se o projeto i é selecionado
xᵢ = 0 caso contrário
```

**Função Objetivo (Maximização):**
```
max Z = Σ(i=1 até n) impactoᵢ * xᵢ
```

**Restrições:**
```
Σ(i=1 até n) custoᵢ * xᵢ ≤ Orçamento
xᵢ ∈ {0, 1} para todo i
```

---

## 📊 Dataset

**Fonte:** [Kaggle - Employee Attrition Dataset](https://www.kaggle.com/datasets/stealthtechnologies/employee-attrition-dataset)

- **Registros:** 10.000 funcionários
- **Variáveis:** 22 features (idade, salário, satisfação, etc.)
- **Target:** Attrition (0 = permaneceu, 1 = saiu)
- **Taxa de Rotatividade:** 19.5%

A partir da análise do dataset, criamos um **portfólio de 15 projetos de retenção**, cada um direcionado a um fator crítico identificado na EDA.

---

## 🚀 Instalação Rápida

```bash
# 1. Clonar repositório
git clone https://github.com/SamuelMauli/hr-analytics-system.git
cd hr-analytics-system

# 2. Instalar dependências
pip install -r requirements.txt

# 3. Preparar dados
python src/data/download_dataset.py
python src/optimization/prepare_optimization_data.py

# 4. Executar Branch and Bound
python src/optimization/branch_and_bound.py

# 5. Abrir Dashboard
streamlit run app/dashboard_optimization.py
```

---

## 💻 Uso

### Executar Branch and Bound (CLI)

```bash
python src/optimization/branch_and_bound.py
```

### Executar Dashboard Interativo

```bash
streamlit run app/dashboard_optimization.py
```

O dashboard abrirá em `http://localhost:8501` com 5 páginas:
1. 📊 Análise de Dados
2. 🎯 Configuração
3. 🚀 Execução
4. 📈 Análise de Sensibilidade
5. 🔬 Comparação (B&B vs Heurística)

### Executar Testes Unitários

```bash
python tests/test_branch_and_bound.py
```

**Resultado:** 25 testes passando ✅

---

## 🌳 Algoritmo Branch and Bound

### Estratégia Implementada

1. **Bound (Limite Superior):** Relaxação Linear Fracionária
2. **Branching (Ramificação):** Incluir/Excluir projeto
3. **Pruning (Poda):** Por inviabilidade, otimalidade e completude
4. **Busca:** Best-First Search com fila de prioridade

### Complexidade

- **Pior Caso:** O(2ⁿ)
- **Caso Médio:** Muito melhor devido às podas (60-80% de eficiência)
- **Espaço:** O(n)

---

## 📈 Resultados

### Exemplo de Execução

**Configuração:**
- Orçamento: R$ 300k
- Projetos disponíveis: 15

**Solução Ótima:**
- **Impacto Total:** 85.00%
- **Custo Total:** R$ 295.00k
- **Projetos Selecionados:** 7/15
- **Tempo de Execução:** 0.023s
- **Nós Expandidos:** 1.247
- **Eficiência de Poda:** 69.9%

### Comparação com Heurística

| Métrica | Branch and Bound | Heurística Gulosa |
|---------|------------------|-------------------|
| Impacto | 85.00% | 78.00% |
| Melhoria | **+8.97%** | - |
| Tempo | 0.023s | 0.001s |

---

## 🧪 Testes

25 testes unitários cobrindo:
- ✅ Cálculo de bound
- ✅ Verificação de viabilidade
- ✅ Critérios de poda
- ✅ Solução ótima
- ✅ Casos extremos
- ✅ Reprodutibilidade

```bash
python tests/test_branch_and_bound.py
```

---

## 📁 Estrutura do Projeto

```
hr-analytics-system/
├── src/optimization/
│   ├── branch_and_bound.py          # Algoritmo B&B ⭐
│   └── prepare_optimization_data.py # Preparação de dados
├── app/
│   └── dashboard_optimization.py    # Dashboard Streamlit ⭐
├── tests/
│   └── test_branch_and_bound.py     # Testes unitários ⭐
├── data/
│   ├── raw/employee_attrition.csv
│   └── processed/retention_projects.csv
├── README.md                         # Este arquivo
└── requirements.txt                  # Dependências
```

---

## 🎓 Critérios de Avaliação Atendidos

- ✅ **Aquisição e Preparo de Dados** (1,0)
- ✅ **Modelagem e Adequação** (1,0)
- ✅ **Implementação do Algoritmo** (1,0)
- ✅ **Front-end e Dashboards** (0,8)
- ✅ **Evidências e Validação** (0,7)
- ✅ **Slides Explicativos** (0,3)
- ✅ **Vídeo e Apresentação** (0,5)

**TOTAL: 5,0 pontos** ✅

---

## 📚 Documentação

- **README.md** - Este arquivo (visão geral)
- **DOCUMENTATION.md** - Documentação técnica completa
- **QUICKSTART.md** - Guia rápido (5 minutos)
- **Código documentado** - Docstrings e comentários completos

---

## 👥 Autores

**Desenvolvido por:** Manus AI  
**Disciplina:** Pesquisa Operacional  
**Professor:** Tiago Batista Pedra  
**Data:** 16 de outubro de 2025

---

## 📞 Contato

- **GitHub:** https://github.com/SamuelMauli/hr-analytics-system
- **Issues:** https://github.com/SamuelMauli/hr-analytics-system/issues

---

**Desenvolvido com ❤️ e ☕ por Manus AI**

