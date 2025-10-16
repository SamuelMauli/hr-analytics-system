"""
API REST para o Sistema de HR Analytics
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

from config.config import (
    MODELS_DIR, PROCESSED_DATA_DIR, APP_NAME, APP_VERSION,
    EMPLOYEE_PERSONAS, HR_METRICS, RISK_THRESHOLDS
)

# Criar aplicação FastAPI
app = FastAPI(
    title=APP_NAME,
    version=APP_VERSION,
    description="API REST para previsão de rotatividade de funcionários e análise de RH",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Modelos Pydantic
class EmployeeInput(BaseModel):
    """Modelo de entrada para previsão"""
    Age: int = Field(..., ge=18, le=65, description="Idade do funcionário")
    MonthlyIncome: float = Field(..., ge=0, description="Salário mensal")
    YearsAtCompany: int = Field(..., ge=0, description="Anos na empresa")
    DistanceFromHome: float = Field(..., ge=0, description="Distância de casa (km)")
    NumberOfPromotions: int = Field(..., ge=0, description="Número de promoções")
    JobSatisfaction_Encoded: int = Field(..., ge=0, le=3, description="Satisfação no trabalho (0-3)")
    WorkLifeBalance_Encoded: int = Field(..., ge=0, le=3, description="Work-Life Balance (0-3)")
    PerformanceRating_Encoded: int = Field(..., ge=0, le=3, description="Avaliação de desempenho (0-3)")
    Gender_Encoded: int = Field(..., ge=0, le=1, description="Gênero (0-1)")
    JobRole_Encoded: int = Field(..., ge=0, le=4, description="Cargo (0-4)")
    MaritalStatus_Encoded: int = Field(..., ge=0, le=2, description="Estado civil (0-2)")
    EducationLevel_Encoded: int = Field(..., ge=0, le=2, description="Nível educacional (0-2)")
    
    class Config:
        schema_extra = {
            "example": {
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
        }

class PredictionOutput(BaseModel):
    """Modelo de saída para previsão"""
    employee_id: Optional[int] = None
    attrition_probability: float = Field(..., description="Probabilidade de rotatividade (0-1)")
    risk_level: str = Field(..., description="Nível de risco (high, medium, low)")
    risk_label: str = Field(..., description="Label do risco")
    persona: Optional[str] = None
    recommendations: List[str] = Field(..., description="Ações recomendadas")
    predicted_at: str = Field(..., description="Timestamp da previsão")

class MetricOutput(BaseModel):
    """Modelo de saída para métricas"""
    metric_name: str
    metric_value: float
    description: str
    formula: Optional[str] = None

# Carregar modelo ao iniciar
model = None
feature_names = []

@app.on_event("startup")
async def load_model():
    """Carrega o modelo ao iniciar a aplicação"""
    global model, feature_names
    
    try:
        model_path = MODELS_DIR / 'best_model.pkl'
        model = joblib.load(model_path)
        
        # Carregar nomes das features
        with open(PROCESSED_DATA_DIR / 'feature_names.txt', 'r') as f:
            feature_names = [line.strip() for line in f.readlines()]
        
        print(f"✓ Modelo carregado: {model_path}")
        print(f"✓ Features: {len(feature_names)}")
    except Exception as e:
        print(f"❌ Erro ao carregar modelo: {e}")

# Rotas
@app.get("/", tags=["Root"])
async def root():
    """Rota raiz da API"""
    return {
        "message": "HR Analytics API",
        "version": APP_VERSION,
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", tags=["Health"])
async def health_check():
    """Verifica o status da API"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/predict", response_model=PredictionOutput, tags=["Prediction"])
async def predict_attrition(employee: EmployeeInput):
    """
    Prevê a probabilidade de rotatividade de um funcionário
    
    - **employee**: Dados do funcionário
    - **returns**: Previsão com probabilidade, nível de risco e recomendações
    """
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Modelo não carregado"
        )
    
    try:
        # Preparar dados para previsão
        input_data = pd.DataFrame([employee.dict()])
        
        # Calcular features engineered
        input_data['IncomePerYearOfService'] = input_data['MonthlyIncome'] / (input_data['YearsAtCompany'] + 1)
        input_data['TenureToAgeRatio'] = input_data['YearsAtCompany'] / input_data['Age']
        input_data['PromotionRate'] = input_data['NumberOfPromotions'] / (input_data['YearsAtCompany'] + 1)
        
        # Reordenar colunas conforme feature_names
        input_data = input_data[feature_names]
        
        # Fazer previsão
        prediction_proba = model.predict_proba(input_data)[0, 1]
        
        # Calcular nível de risco
        if prediction_proba >= RISK_THRESHOLDS['high']:
            risk_level = 'high'
            risk_label = '🔴 Alto'
        elif prediction_proba >= RISK_THRESHOLDS['medium']:
            risk_level = 'medium'
            risk_label = '🟡 Médio'
        else:
            risk_level = 'low'
            risk_label = '🟢 Baixo'
        
        # Determinar persona (simplificado)
        persona_id = 0 if prediction_proba > 0.7 else 1
        persona_name = EMPLOYEE_PERSONAS[persona_id]['name']
        recommendations = EMPLOYEE_PERSONAS[persona_id]['recommendations']
        
        return PredictionOutput(
            attrition_probability=float(prediction_proba),
            risk_level=risk_level,
            risk_label=risk_label,
            persona=persona_name,
            recommendations=recommendations,
            predicted_at=datetime.now().isoformat()
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erro ao fazer previsão: {str(e)}"
        )

@app.get("/personas", tags=["Personas"])
async def get_personas():
    """
    Retorna todas as personas de funcionários
    
    - **returns**: Lista de personas com descrições e recomendações
    """
    return {
        "personas": [
            {
                "id": persona_id,
                "name": persona_info['name'],
                "description": persona_info['description'],
                "recommendations": persona_info['recommendations']
            }
            for persona_id, persona_info in EMPLOYEE_PERSONAS.items()
        ]
    }

@app.get("/metrics", response_model=List[MetricOutput], tags=["Metrics"])
async def get_hr_metrics():
    """
    Retorna todas as métricas de RH disponíveis
    
    - **returns**: Lista de métricas com descrições e fórmulas
    """
    return [
        MetricOutput(
            metric_name=metric_name,
            metric_value=0.0,  # Valor placeholder
            description=metric_info['description'],
            formula=metric_info.get('formula')
        )
        for metric_name, metric_info in HR_METRICS.items()
    ]

@app.get("/model/info", tags=["Model"])
async def get_model_info():
    """
    Retorna informações sobre o modelo carregado
    
    - **returns**: Metadados do modelo
    """
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Modelo não carregado"
        )
    
    try:
        import json
        metadata_path = MODELS_DIR / "model_metadata.json"
        
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            return {
                "model_type": metadata.get('best_model'),
                "training_date": metadata.get('training_date'),
                "metrics": metadata.get('results', {}).get(metadata.get('best_model'), {}),
                "features": feature_names
            }
        else:
            return {
                "model_type": "Unknown",
                "features": feature_names
            }
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erro ao obter informações do modelo: {str(e)}"
        )

@app.post("/batch-predict", tags=["Prediction"])
async def batch_predict(employees: List[EmployeeInput]):
    """
    Prevê a rotatividade para múltiplos funcionários
    
    - **employees**: Lista de dados de funcionários
    - **returns**: Lista de previsões
    """
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Modelo não carregado"
        )
    
    predictions = []
    
    for employee in employees:
        try:
            prediction = await predict_attrition(employee)
            predictions.append(prediction)
        except Exception as e:
            predictions.append({
                "error": str(e),
                "employee_data": employee.dict()
            })
    
    return {
        "total": len(employees),
        "predictions": predictions,
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)

