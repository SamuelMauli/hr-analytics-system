-- Schema para o Sistema de HR Analytics no Supabase

-- Tabela de Previsões de Rotatividade
CREATE TABLE IF NOT EXISTS predictions (
    id BIGSERIAL PRIMARY KEY,
    employee_id INTEGER NOT NULL,
    prediction_probability DECIMAL(5, 4) NOT NULL CHECK (prediction_probability >= 0 AND prediction_probability <= 1),
    persona VARCHAR(100),
    risk_level VARCHAR(20) CHECK (risk_level IN ('high', 'medium', 'low')),
    features JSONB,
    predicted_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Índices para melhor performance
CREATE INDEX IF NOT EXISTS idx_predictions_employee_id ON predictions(employee_id);
CREATE INDEX IF NOT EXISTS idx_predictions_risk_level ON predictions(risk_level);
CREATE INDEX IF NOT EXISTS idx_predictions_predicted_at ON predictions(predicted_at DESC);

-- Tabela de Métricas de RH
CREATE TABLE IF NOT EXISTS hr_metrics (
    id BIGSERIAL PRIMARY KEY,
    metric_name VARCHAR(100) NOT NULL,
    metric_value DECIMAL(15, 4) NOT NULL,
    period VARCHAR(50) NOT NULL,
    metadata JSONB,
    recorded_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Índices
CREATE INDEX IF NOT EXISTS idx_hr_metrics_name ON hr_metrics(metric_name);
CREATE INDEX IF NOT EXISTS idx_hr_metrics_period ON hr_metrics(period);
CREATE INDEX IF NOT EXISTS idx_hr_metrics_recorded_at ON hr_metrics(recorded_at DESC);

-- Tabela de Logs de Execução
CREATE TABLE IF NOT EXISTS execution_logs (
    id BIGSERIAL PRIMARY KEY,
    action VARCHAR(100) NOT NULL,
    status VARCHAR(20) CHECK (status IN ('success', 'error', 'warning', 'info')),
    details JSONB,
    error_message TEXT,
    executed_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Índices
CREATE INDEX IF NOT EXISTS idx_execution_logs_action ON execution_logs(action);
CREATE INDEX IF NOT EXISTS idx_execution_logs_status ON execution_logs(status);
CREATE INDEX IF NOT EXISTS idx_execution_logs_executed_at ON execution_logs(executed_at DESC);

-- Tabela de Configurações do Sistema
CREATE TABLE IF NOT EXISTS system_configs (
    id BIGSERIAL PRIMARY KEY,
    config_key VARCHAR(100) UNIQUE NOT NULL,
    config_value JSONB NOT NULL,
    description TEXT,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Índice
CREATE UNIQUE INDEX IF NOT EXISTS idx_system_configs_key ON system_configs(config_key);

-- Tabela de Personas de Funcionários
CREATE TABLE IF NOT EXISTS employee_personas (
    id BIGSERIAL PRIMARY KEY,
    employee_id INTEGER NOT NULL,
    persona_id INTEGER NOT NULL CHECK (persona_id >= 0 AND persona_id <= 3),
    persona_name VARCHAR(100) NOT NULL,
    features JSONB,
    assigned_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Índices
CREATE INDEX IF NOT EXISTS idx_employee_personas_employee_id ON employee_personas(employee_id);
CREATE INDEX IF NOT EXISTS idx_employee_personas_persona_id ON employee_personas(persona_id);
CREATE INDEX IF NOT EXISTS idx_employee_personas_assigned_at ON employee_personas(assigned_at DESC);

-- Comentários nas tabelas
COMMENT ON TABLE predictions IS 'Armazena previsões de rotatividade de funcionários';
COMMENT ON TABLE hr_metrics IS 'Armazena métricas de RH ao longo do tempo';
COMMENT ON TABLE execution_logs IS 'Logs de execução do sistema de analytics';
COMMENT ON TABLE system_configs IS 'Configurações do sistema';
COMMENT ON TABLE employee_personas IS 'Personas atribuídas aos funcionários';

-- Comentários nas colunas principais
COMMENT ON COLUMN predictions.prediction_probability IS 'Probabilidade de rotatividade (0-1)';
COMMENT ON COLUMN predictions.risk_level IS 'Nível de risco: high, medium, low';
COMMENT ON COLUMN hr_metrics.period IS 'Período da métrica (e.g., 2025-10, Q4-2025)';
COMMENT ON COLUMN employee_personas.persona_id IS 'ID da persona (0-3)';

-- Habilitar Row Level Security (RLS) - Opcional, mas recomendado
-- ALTER TABLE predictions ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE hr_metrics ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE execution_logs ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE system_configs ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE employee_personas ENABLE ROW LEVEL SECURITY;

-- Políticas de acesso (exemplo - ajustar conforme necessário)
-- CREATE POLICY "Enable read access for all users" ON predictions FOR SELECT USING (true);
-- CREATE POLICY "Enable insert for authenticated users only" ON predictions FOR INSERT WITH CHECK (auth.role() = 'authenticated');

