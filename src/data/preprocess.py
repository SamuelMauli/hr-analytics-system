"""
Pré-processamento de dados para o sistema de HR Analytics
"""
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# Adicionar o diretório raiz ao path
sys.path.append(str(Path(__file__).parent.parent.parent))

from config.config import (
    RAW_DATA_DIR, PROCESSED_DATA_DIR, DATASET_FILE,
    NUMERIC_FEATURES, CATEGORICAL_FEATURES, ORDINAL_FEATURES, TARGET_FEATURE
)

class DataPreprocessor:
    """Classe para pré-processamento de dados de RH"""
    
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        
    def load_data(self, file_path: str) -> pd.DataFrame:
        """Carrega o dataset"""
        print(f"Carregando dados de: {file_path}")
        df = pd.read_csv(file_path)
        print(f"Dataset carregado: {df.shape[0]} linhas, {df.shape[1]} colunas")
        return df
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Trata valores ausentes"""
        print("\nTratando valores ausentes...")
        missing_before = df.isnull().sum().sum()
        
        # Para features numéricas: preencher com mediana
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isnull().sum() > 0:
                df[col].fillna(df[col].median(), inplace=True)
        
        # Para features categóricas: preencher com moda
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df[col].isnull().sum() > 0:
                df[col].fillna(df[col].mode()[0], inplace=True)
        
        missing_after = df.isnull().sum().sum()
        print(f"Valores ausentes antes: {missing_before}")
        print(f"Valores ausentes depois: {missing_after}")
        
        return df
    
    def detect_outliers(self, df: pd.DataFrame, columns: list) -> pd.DataFrame:
        """Detecta e trata outliers usando IQR"""
        print("\nDetectando e tratando outliers...")
        
        for col in columns:
            if col in df.columns and df[col].dtype in [np.int64, np.float64]:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
                
                if outliers > 0:
                    print(f"  {col}: {outliers} outliers detectados")
                    # Substituir outliers pelos limites
                    df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
        
        return df
    
    def encode_ordinal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Codifica features ordinais"""
        print("\nCodificando features ordinais...")
        
        for feature, order in ORDINAL_FEATURES.items():
            if feature in df.columns:
                # Criar mapeamento ordinal
                ordinal_map = {value: idx for idx, value in enumerate(order)}
                df[f'{feature}_Encoded'] = df[feature].map(ordinal_map)
                print(f"  {feature}: {order}")
        
        return df
    
    def encode_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Codifica features categóricas usando Label Encoding"""
        print("\nCodificando features categóricas...")
        
        for feature in CATEGORICAL_FEATURES:
            if feature in df.columns and feature not in ORDINAL_FEATURES:
                le = LabelEncoder()
                df[f'{feature}_Encoded'] = le.fit_transform(df[feature])
                self.label_encoders[feature] = le
                print(f"  {feature}: {len(le.classes_)} classes")
        
        return df
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cria novas features através de engenharia de features"""
        print("\nCriando features engineered...")
        
        # IncomePerYearOfService
        df['IncomePerYearOfService'] = df['MonthlyIncome'] / (df['YearsAtCompany'] + 1)
        print("  IncomePerYearOfService: MonthlyIncome / (YearsAtCompany + 1)")
        
        # TenureToAgeRatio
        df['TenureToAgeRatio'] = df['YearsAtCompany'] / df['Age']
        print("  TenureToAgeRatio: YearsAtCompany / Age")
        
        # PromotionRate
        df['PromotionRate'] = df['NumberOfPromotions'] / (df['YearsAtCompany'] + 1)
        print("  PromotionRate: NumberOfPromotions / (YearsAtCompany + 1)")
        
        # AgeGroup
        df['AgeGroup'] = pd.cut(df['Age'], bins=[0, 25, 35, 45, 55, 100], 
                                labels=['18-25', '26-35', '36-45', '46-55', '56+'])
        print("  AgeGroup: Categorização por faixa etária")
        
        # IncomeGroup
        df['IncomeGroup'] = pd.cut(df['MonthlyIncome'], bins=[0, 5000, 10000, 15000, 100000],
                                    labels=['Low', 'Medium', 'High', 'Very High'])
        print("  IncomeGroup: Categorização por faixa salarial")
        
        return df
    
    def prepare_features(self, df: pd.DataFrame) -> tuple:
        """Prepara features para modelagem"""
        print("\nPreparando features para modelagem...")
        
        # Selecionar features numéricas e encoded
        feature_cols = []
        
        # Features numéricas originais
        for col in NUMERIC_FEATURES:
            if col in df.columns:
                feature_cols.append(col)
        
        # Features encoded
        encoded_cols = [col for col in df.columns if col.endswith('_Encoded')]
        feature_cols.extend(encoded_cols)
        
        # Features engineered
        engineered_cols = ['IncomePerYearOfService', 'TenureToAgeRatio', 'PromotionRate']
        for col in engineered_cols:
            if col in df.columns:
                feature_cols.append(col)
        
        # Preparar X e y
        X = df[feature_cols].copy()
        y = df[TARGET_FEATURE].copy()
        
        print(f"Features selecionadas: {len(feature_cols)}")
        print(f"Distribuição do target:")
        print(f"  Classe 0 (Stayed): {(y == 0).sum()} ({(y == 0).mean():.2%})")
        print(f"  Classe 1 (Left): {(y == 1).sum()} ({(y == 1).mean():.2%})")
        
        return X, y, feature_cols
    
    def split_data(self, X: pd.DataFrame, y: pd.Series, 
                   test_size: float = 0.2, random_state: int = 42) -> tuple:
        """Divide os dados em treino e teste"""
        print(f"\nDividindo dados (test_size={test_size})...")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"Treino: {X_train.shape[0]} amostras")
        print(f"Teste: {X_test.shape[0]} amostras")
        
        return X_train, X_test, y_train, y_test
    
    def scale_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> tuple:
        """Normaliza as features"""
        print("\nNormalizando features...")
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Converter de volta para DataFrame
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
        
        print("Features normalizadas com StandardScaler")
        
        return X_train_scaled, X_test_scaled
    
    def process_pipeline(self, input_file: str, output_dir: str) -> dict:
        """Pipeline completo de pré-processamento"""
        print("=" * 60)
        print("PIPELINE DE PRÉ-PROCESSAMENTO DE DADOS")
        print("=" * 60)
        
        # 1. Carregar dados
        df = self.load_data(input_file)
        
        # 2. Tratar valores ausentes
        df = self.handle_missing_values(df)
        
        # 3. Detectar e tratar outliers
        df = self.detect_outliers(df, NUMERIC_FEATURES)
        
        # 4. Codificar features ordinais
        df = self.encode_ordinal_features(df)
        
        # 5. Codificar features categóricas
        df = self.encode_categorical_features(df)
        
        # 6. Engenharia de features
        df = self.engineer_features(df)
        
        # 7. Preparar features
        X, y, feature_cols = self.prepare_features(df)
        
        # 8. Dividir dados
        X_train, X_test, y_train, y_test = self.split_data(X, y)
        
        # 9. Normalizar features
        X_train_scaled, X_test_scaled = self.scale_features(X_train, X_test)
        
        # 10. Salvar dados processados
        print("\nSalvando dados processados...")
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Salvar datasets
        df.to_csv(output_path / 'employee_attrition_processed.csv', index=False)
        X_train_scaled.to_csv(output_path / 'X_train.csv', index=False)
        X_test_scaled.to_csv(output_path / 'X_test.csv', index=False)
        y_train.to_csv(output_path / 'y_train.csv', index=False)
        y_test.to_csv(output_path / 'y_test.csv', index=False)
        
        # Salvar lista de features
        with open(output_path / 'feature_names.txt', 'w') as f:
            f.write('\n'.join(feature_cols))
        
        print(f"Dados salvos em: {output_path}")
        
        print("\n" + "=" * 60)
        print("PRÉ-PROCESSAMENTO CONCLUÍDO COM SUCESSO!")
        print("=" * 60)
        
        return {
            'df': df,
            'X_train': X_train_scaled,
            'X_test': X_test_scaled,
            'y_train': y_train,
            'y_test': y_test,
            'feature_cols': feature_cols
        }


def main():
    """Função principal"""
    # Caminhos
    input_file = RAW_DATA_DIR / DATASET_FILE
    output_dir = PROCESSED_DATA_DIR
    
    # Executar pipeline
    preprocessor = DataPreprocessor()
    results = preprocessor.process_pipeline(str(input_file), str(output_dir))
    
    print("\nPróximos passos:")
    print("1. Executar análise exploratória: jupyter notebook notebooks/01_exploratory_data_analysis.ipynb")
    print("2. Treinar modelos: python src/models/train_models.py")


if __name__ == "__main__":
    main()

