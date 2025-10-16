"""
Cliente Supabase para armazenamento de dados do HR Analytics
"""
import os
from datetime import datetime
from typing import Dict, List, Any, Optional
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

class SupabaseClient:
    """Cliente para interação com Supabase"""
    
    def __init__(self):
        """Inicializa o cliente Supabase"""
        self.url = os.getenv("SUPABASE_URL", "")
        self.key = os.getenv("SUPABASE_KEY", "")
        
        if not self.url or not self.key:
            raise ValueError("SUPABASE_URL e SUPABASE_KEY devem estar configurados no .env")
        
        self.client: Client = create_client(self.url, self.key)
    
    # ========== Previsões ==========
    
    def save_prediction(self, employee_id: int, prediction: float, 
                       persona: str, risk_level: str, 
                       features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Salva uma previsão de rotatividade
        
        Args:
            employee_id: ID do funcionário
            prediction: Probabilidade de rotatividade (0-1)
            persona: Persona do funcionário
            risk_level: Nível de risco (high, medium, low)
            features: Features usadas na previsão
            
        Returns:
            Resultado da inserção
        """
        data = {
            "employee_id": employee_id,
            "prediction_probability": prediction,
            "persona": persona,
            "risk_level": risk_level,
            "features": features,
            "predicted_at": datetime.now().isoformat()
        }
        
        response = self.client.table("predictions").insert(data).execute()
        return response.data
    
    def get_predictions(self, employee_id: Optional[int] = None,
                       risk_level: Optional[str] = None,
                       limit: int = 100) -> List[Dict[str, Any]]:
        """
        Recupera previsões do banco
        
        Args:
            employee_id: Filtrar por ID do funcionário (opcional)
            risk_level: Filtrar por nível de risco (opcional)
            limit: Número máximo de resultados
            
        Returns:
            Lista de previsões
        """
        query = self.client.table("predictions").select("*")
        
        if employee_id:
            query = query.eq("employee_id", employee_id)
        
        if risk_level:
            query = query.eq("risk_level", risk_level)
        
        response = query.limit(limit).order("predicted_at", desc=True).execute()
        return response.data
    
    # ========== Métricas de RH ==========
    
    def save_hr_metric(self, metric_name: str, metric_value: float,
                      period: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Salva uma métrica de RH
        
        Args:
            metric_name: Nome da métrica (turnover_rate, cost_per_hire, etc.)
            metric_value: Valor da métrica
            period: Período da métrica (e.g., "2025-10", "Q4-2025")
            metadata: Metadados adicionais
            
        Returns:
            Resultado da inserção
        """
        data = {
            "metric_name": metric_name,
            "metric_value": metric_value,
            "period": period,
            "metadata": metadata or {},
            "recorded_at": datetime.now().isoformat()
        }
        
        response = self.client.table("hr_metrics").insert(data).execute()
        return response.data
    
    def get_hr_metrics(self, metric_name: Optional[str] = None,
                      period: Optional[str] = None,
                      limit: int = 100) -> List[Dict[str, Any]]:
        """
        Recupera métricas de RH
        
        Args:
            metric_name: Filtrar por nome da métrica (opcional)
            period: Filtrar por período (opcional)
            limit: Número máximo de resultados
            
        Returns:
            Lista de métricas
        """
        query = self.client.table("hr_metrics").select("*")
        
        if metric_name:
            query = query.eq("metric_name", metric_name)
        
        if period:
            query = query.eq("period", period)
        
        response = query.limit(limit).order("recorded_at", desc=True).execute()
        return response.data
    
    # ========== Logs de Execução ==========
    
    def log_execution(self, action: str, status: str, 
                     details: Optional[Dict[str, Any]] = None,
                     error: Optional[str] = None) -> Dict[str, Any]:
        """
        Registra log de execução do sistema
        
        Args:
            action: Ação executada (e.g., "train_model", "predict", "eda")
            status: Status da execução (success, error, warning)
            details: Detalhes adicionais
            error: Mensagem de erro (se houver)
            
        Returns:
            Resultado da inserção
        """
        data = {
            "action": action,
            "status": status,
            "details": details or {},
            "error_message": error,
            "executed_at": datetime.now().isoformat()
        }
        
        response = self.client.table("execution_logs").insert(data).execute()
        return response.data
    
    def get_execution_logs(self, action: Optional[str] = None,
                          status: Optional[str] = None,
                          limit: int = 100) -> List[Dict[str, Any]]:
        """
        Recupera logs de execução
        
        Args:
            action: Filtrar por ação (opcional)
            status: Filtrar por status (opcional)
            limit: Número máximo de resultados
            
        Returns:
            Lista de logs
        """
        query = self.client.table("execution_logs").select("*")
        
        if action:
            query = query.eq("action", action)
        
        if status:
            query = query.eq("status", status)
        
        response = query.limit(limit).order("executed_at", desc=True).execute()
        return response.data
    
    # ========== Configurações do Sistema ==========
    
    def save_config(self, config_key: str, config_value: Any,
                   description: Optional[str] = None) -> Dict[str, Any]:
        """
        Salva uma configuração do sistema
        
        Args:
            config_key: Chave da configuração
            config_value: Valor da configuração
            description: Descrição da configuração
            
        Returns:
            Resultado da inserção/atualização
        """
        data = {
            "config_key": config_key,
            "config_value": config_value,
            "description": description,
            "updated_at": datetime.now().isoformat()
        }
        
        # Upsert: insere se não existir, atualiza se existir
        response = self.client.table("system_configs").upsert(
            data, 
            on_conflict="config_key"
        ).execute()
        return response.data
    
    def get_config(self, config_key: str) -> Optional[Any]:
        """
        Recupera uma configuração do sistema
        
        Args:
            config_key: Chave da configuração
            
        Returns:
            Valor da configuração ou None se não encontrada
        """
        response = self.client.table("system_configs").select(
            "config_value"
        ).eq("config_key", config_key).execute()
        
        if response.data:
            return response.data[0]["config_value"]
        return None
    
    # ========== Personas de Funcionários ==========
    
    def save_employee_persona(self, employee_id: int, persona_id: int,
                             persona_name: str, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Salva a persona de um funcionário
        
        Args:
            employee_id: ID do funcionário
            persona_id: ID da persona (0-3)
            persona_name: Nome da persona
            features: Features do funcionário
            
        Returns:
            Resultado da inserção
        """
        data = {
            "employee_id": employee_id,
            "persona_id": persona_id,
            "persona_name": persona_name,
            "features": features,
            "assigned_at": datetime.now().isoformat()
        }
        
        response = self.client.table("employee_personas").insert(data).execute()
        return response.data
    
    def get_employee_personas(self, persona_id: Optional[int] = None,
                             limit: int = 100) -> List[Dict[str, Any]]:
        """
        Recupera personas de funcionários
        
        Args:
            persona_id: Filtrar por ID da persona (opcional)
            limit: Número máximo de resultados
            
        Returns:
            Lista de personas
        """
        query = self.client.table("employee_personas").select("*")
        
        if persona_id is not None:
            query = query.eq("persona_id", persona_id)
        
        response = query.limit(limit).order("assigned_at", desc=True).execute()
        return response.data


# Função auxiliar para criar instância do cliente
def get_supabase_client() -> SupabaseClient:
    """Retorna uma instância do cliente Supabase"""
    return SupabaseClient()

