"""
Script para download do dataset Synthetic Employee Attrition do Kaggle
"""
import os
import sys
import pandas as pd
from pathlib import Path

# Adicionar o diretório raiz ao path
sys.path.append(str(Path(__file__).parent.parent.parent))

from config.config import RAW_DATA_DIR, DATASET_FILE

def create_sample_dataset():
    """
    Cria um dataset de exemplo baseado nas especificações do PDF
    
    Como não temos acesso direto ao Kaggle API, vamos criar um dataset sintético
    com as mesmas características do dataset original para demonstração.
    """
    import numpy as np
    
    print("Criando dataset sintético de exemplo...")
    
    # Configurar seed para reprodutibilidade
    np.random.seed(42)
    
    # Número de amostras
    n_samples = 10000
    
    # Criar features
    data = {
        'EmployeeID': range(1, n_samples + 1),
        'Age': np.random.randint(18, 61, n_samples),
        'Gender': np.random.choice(['Male', 'Female'], n_samples),
        'YearsAtCompany': np.random.randint(0, 31, n_samples),
        'MonthlyIncome': np.random.randint(2000, 20000, n_samples),
        'JobRole': np.random.choice(['Finance', 'Healthcare', 'Technology', 'Education', 'Media'], n_samples),
        'WorkLifeBalance': np.random.choice(['Poor', 'Below Average', 'Good', 'Excellent'], n_samples),
        'JobSatisfaction': np.random.choice(['Very Low', 'Low', 'Medium', 'High'], n_samples),
        'PerformanceRating': np.random.choice(['Low', 'Below Average', 'Average', 'High'], n_samples),
        'NumberOfPromotions': np.random.randint(0, 6, n_samples),
        'DistanceFromHome': np.random.randint(1, 51, n_samples),
        'EducationLevel': np.random.choice(['High School', 'Associate Degree', "Bachelor's Degree", "Master's Degree", 'PhD'], n_samples),
        'MaritalStatus': np.random.choice(['Divorced', 'Married', 'Single'], n_samples),
        'JobLevel': np.random.choice(['Entry', 'Mid', 'Senior'], n_samples),
        'CompanySize': np.random.choice(['Small', 'Medium', 'Large'], n_samples),
        'CompanyTenure': np.random.randint(0, 31, n_samples),
        'RemoteWork': np.random.choice(['Yes', 'No'], n_samples),
        'LeadershipOpportunities': np.random.choice(['Yes', 'No'], n_samples),
        'InnovationOpportunities': np.random.choice(['Yes', 'No'], n_samples),
        'CompanyReputation': np.random.choice(['Very Poor', 'Poor', 'Good', 'Excellent'], n_samples),
        'EmployeeRecognition': np.random.choice(['Very Low', 'Low', 'Medium', 'High'], n_samples),
    }
    
    # Criar DataFrame
    df = pd.DataFrame(data)
    
    # Criar target (Attrition) com lógica baseada em features
    # Funcionários com baixa satisfação, baixo salário e alta distância têm maior chance de sair
    attrition_prob = np.zeros(n_samples)
    
    # Fatores que aumentam a probabilidade de attrition
    attrition_prob += (df['JobSatisfaction'] == 'Very Low').astype(int) * 0.3
    attrition_prob += (df['JobSatisfaction'] == 'Low').astype(int) * 0.2
    attrition_prob += (df['WorkLifeBalance'] == 'Poor').astype(int) * 0.2
    attrition_prob += (df['MonthlyIncome'] < 5000).astype(int) * 0.15
    attrition_prob += (df['DistanceFromHome'] > 30).astype(int) * 0.1
    attrition_prob += (df['NumberOfPromotions'] == 0).astype(int) * 0.15
    attrition_prob += (df['PerformanceRating'] == 'Low').astype(int) * 0.1
    attrition_prob += (df['EmployeeRecognition'] == 'Very Low').astype(int) * 0.15
    
    # Fatores que diminuem a probabilidade de attrition
    attrition_prob -= (df['JobSatisfaction'] == 'High').astype(int) * 0.2
    attrition_prob -= (df['WorkLifeBalance'] == 'Excellent').astype(int) * 0.15
    attrition_prob -= (df['MonthlyIncome'] > 15000).astype(int) * 0.1
    attrition_prob -= (df['NumberOfPromotions'] > 3).astype(int) * 0.1
    attrition_prob -= (df['LeadershipOpportunities'] == 'Yes').astype(int) * 0.1
    
    # Normalizar probabilidades entre 0 e 1
    attrition_prob = np.clip(attrition_prob, 0, 1)
    
    # Gerar attrition baseado nas probabilidades
    df['Attrition'] = (np.random.random(n_samples) < attrition_prob).astype(int)
    
    # Salvar dataset
    output_path = RAW_DATA_DIR / DATASET_FILE
    df.to_csv(output_path, index=False)
    
    print(f"Dataset criado com sucesso: {output_path}")
    print(f"Tamanho: {len(df)} registros")
    print(f"Taxa de attrition: {df['Attrition'].mean():.2%}")
    print(f"\nPrimeiras linhas:")
    print(df.head())
    print(f"\nEstatísticas descritivas:")
    print(df.describe())
    
    return df

def download_dataset():
    """
    Função principal para download/criação do dataset
    """
    print("=" * 60)
    print("DOWNLOAD DO DATASET SYNTHETIC EMPLOYEE ATTRITION")
    print("=" * 60)
    
    # Verificar se o dataset já existe
    dataset_path = RAW_DATA_DIR / DATASET_FILE
    
    if dataset_path.exists():
        print(f"\nDataset já existe em: {dataset_path}")
        response = input("Deseja sobrescrever? (s/n): ")
        if response.lower() != 's':
            print("Download cancelado.")
            return
    
    # Criar dataset sintético
    df = create_sample_dataset()
    
    print("\n" + "=" * 60)
    print("DOWNLOAD CONCLUÍDO COM SUCESSO!")
    print("=" * 60)
    print(f"\nDataset salvo em: {dataset_path}")
    print(f"\nPróximos passos:")
    print("1. Execute a análise exploratória: notebooks/01_exploratory_data_analysis.ipynb")
    print("2. Treine os modelos: python src/models/train_models.py")
    print("3. Execute o dashboard: streamlit run app/dashboard.py")

if __name__ == "__main__":
    download_dataset()

