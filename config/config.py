"""
Configurações centralizadas do sistema de HR Analytics
"""
import os
from pathlib import Path
from typing import List, Dict, Any
from dotenv import load_dotenv

# Carregar variáveis de ambiente
load_dotenv()

# Diretórios base
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"

# Criar diretórios se não existirem
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, LOGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Supabase Configuration
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY", "")

# Hugging Face Configuration
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN", "")
HUGGINGFACE_REPO = os.getenv("HUGGINGFACE_REPO", "hr-analytics-models")

# API Configuration
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
API_RELOAD = os.getenv("API_RELOAD", "True").lower() == "true"

# Model Configuration
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "xgboost_model.pkl")
MODEL_PATH = MODELS_DIR / DEFAULT_MODEL

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Application
APP_NAME = os.getenv("APP_NAME", "HR Analytics System")
APP_VERSION = os.getenv("APP_VERSION", "1.0.0")
DEBUG = os.getenv("DEBUG", "False").lower() == "true"

# Dataset Configuration
DATASET_URL = "https://www.kaggle.com/datasets/stealthtechnologies/employee-attrition-dataset"
DATASET_FILE = "employee_attrition.csv"

# Features do Dataset
NUMERIC_FEATURES = [
    "Age",
    "MonthlyIncome",
    "YearsAtCompany",
    "DistanceFromHome",
    "NumberOfPromotions"
]

CATEGORICAL_FEATURES = [
    "Gender",
    "JobRole",
    "WorkLifeBalance",
    "JobSatisfaction",
    "PerformanceRating",
    "EducationLevel",
    "MaritalStatus"
]

ORDINAL_FEATURES = {
    "WorkLifeBalance": ["Poor", "Below Average", "Good", "Excellent"],
    "JobSatisfaction": ["Very Low", "Low", "Medium", "High"],
    "PerformanceRating": ["Low", "Below Average", "Average", "High"],
    "EducationLevel": ["High School", "Bachelor's Degree", "PhD"]
}

TARGET_FEATURE = "Attrition"

# Engenharia de Features
ENGINEERED_FEATURES = [
    "IncomePerYearOfService",
    "TenureToAgeRatio",
    "PromotionRate"
]

# Configurações de Modelo
MODEL_CONFIGS: Dict[str, Dict[str, Any]] = {
    "logistic_regression": {
        "name": "Regressão Logística",
        "params": {
            "max_iter": 1000,
            "random_state": 42,
            "class_weight": "balanced"
        }
    },
    "random_forest": {
        "name": "Random Forest",
        "params": {
            "n_estimators": 100,
            "max_depth": 10,
            "random_state": 42,
            "class_weight": "balanced",
            "n_jobs": -1
        }
    },
    "xgboost": {
        "name": "XGBoost",
        "params": {
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1,
            "random_state": 42,
            "use_label_encoder": False,
            "eval_metric": "logloss"
        }
    }
}

# Configurações de GridSearch
GRID_SEARCH_PARAMS: Dict[str, Dict[str, List[Any]]] = {
    "random_forest": {
        "n_estimators": [50, 100, 200],
        "max_depth": [5, 10, 15],
        "min_samples_split": [2, 5, 10]
    },
    "xgboost": {
        "n_estimators": [50, 100, 200],
        "max_depth": [3, 6, 9],
        "learning_rate": [0.01, 0.1, 0.3]
    }
}

# Configurações de SMOTE
SMOTE_CONFIG = {
    "sampling_strategy": "auto",
    "random_state": 42,
    "k_neighbors": 5
}

# Configurações de K-Means
KMEANS_CONFIG = {
    "n_clusters": 4,
    "random_state": 42,
    "max_iter": 300,
    "n_init": 10
}

# Personas de Funcionários
EMPLOYEE_PERSONAS = {
    0: {
        "name": "Riscos de Fuga de Alto Potencial",
        "description": "Alto desempenho, baixa satisfação, risco alto de rotatividade",
        "recommendations": [
            "Oportunidades de liderança",
            "Mentoria executiva",
            "Projetos de inovação",
            "Plano de carreira acelerado"
        ]
    },
    1: {
        "name": "Contribuidores Centrais Estáveis",
        "description": "Médio desempenho, alta satisfação, baixo risco de rotatividade",
        "recommendations": [
            "Programas de reconhecimento",
            "Iniciativas de equilíbrio vida-trabalho",
            "Papéis de compartilhamento de conhecimento"
        ]
    },
    2: {
        "name": "Novos e Sobrecarregados",
        "description": "Baixo tempo de serviço, risco médio-alto de rotatividade",
        "recommendations": [
            "Onboarding aprimorado",
            "Treinamento direcionado",
            "Mentoria de um colega"
        ]
    },
    3: {
        "name": "Potencial Não Explorado",
        "description": "Médio-baixo desempenho, média satisfação",
        "recommendations": [
            "Planos de desenvolvimento de habilidades",
            "Projetos de stretch",
            "Feedback mais frequente"
        ]
    }
}

# Métricas de RH
HR_METRICS = {
    "turnover_rate": {
        "name": "Taxa de Rotatividade",
        "description": "Porcentagem de funcionários que deixaram a organização",
        "formula": "(Saídas / Headcount médio) * 100"
    },
    "cost_per_hire": {
        "name": "Custo por Contratação",
        "description": "Custo total para recrutar um novo funcionário",
        "formula": "Despesas de recrutamento / Número de contratações"
    },
    "absenteeism_rate": {
        "name": "Taxa de Absenteísmo",
        "description": "Frequência com que funcionários faltam ao trabalho",
        "formula": "(Dias de ausência / Dias úteis) * 100"
    },
    "revenue_per_employee": {
        "name": "Receita por Funcionário",
        "description": "Eficiência da força de trabalho na geração de receita",
        "formula": "Receita total / Headcount"
    },
    "enps": {
        "name": "Employee Net Promoter Score",
        "description": "Probabilidade de funcionários recomendarem a empresa",
        "formula": "% Promotores - % Detratores"
    }
}

# Configurações do Dashboard
DASHBOARD_CONFIG = {
    "title": "HR Analytics Dashboard",
    "layout": "wide",
    "initial_sidebar_state": "expanded",
    "page_icon": "📊"
}

# Configurações de Visualização
PLOT_CONFIG = {
    "color_palette": "viridis",
    "figure_size": (12, 6),
    "dpi": 100,
    "style": "seaborn-v0_8-darkgrid"
}

# Thresholds de Risco
RISK_THRESHOLDS = {
    "high": 0.7,
    "medium": 0.4,
    "low": 0.0
}

