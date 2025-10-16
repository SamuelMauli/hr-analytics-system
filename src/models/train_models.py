"""
Treinamento de modelos de Machine Learning para previsão de rotatividade
"""
import sys
import os
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# ML libraries
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from imblearn.over_sampling import SMOTE

# Adicionar o diretório raiz ao path
sys.path.append(str(Path(__file__).parent.parent.parent))

from config.config import (
    PROCESSED_DATA_DIR, MODELS_DIR, MODEL_CONFIGS,
    GRID_SEARCH_PARAMS, SMOTE_CONFIG
)

class ModelTrainer:
    """Classe para treinamento e avaliação de modelos"""
    
    def __init__(self):
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_model_name = None
        
    def load_data(self):
        """Carrega dados processados"""
        print("Carregando dados processados...")
        
        X_train = pd.read_csv(PROCESSED_DATA_DIR / 'X_train.csv')
        X_test = pd.read_csv(PROCESSED_DATA_DIR / 'X_test.csv')
        y_train = pd.read_csv(PROCESSED_DATA_DIR / 'y_train.csv').values.ravel()
        y_test = pd.read_csv(PROCESSED_DATA_DIR / 'y_test.csv').values.ravel()
        
        # Tratar valores NaN
        X_train = X_train.fillna(X_train.median())
        X_test = X_test.fillna(X_test.median())
        
        print(f"Treino: {X_train.shape}, Teste: {X_test.shape}")
        print(f"Distribuição y_train: {np.bincount(y_train)}")
        print(f"Distribuição y_test: {np.bincount(y_test)}")
        
        return X_train, X_test, y_train, y_test
    
    def apply_smote(self, X_train, y_train):
        """Aplica SMOTE para balanceamento de classes"""
        print("\nAplicando SMOTE para balanceamento...")
        
        smote = SMOTE(**SMOTE_CONFIG)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
        
        print(f"Antes do SMOTE: {np.bincount(y_train)}")
        print(f"Depois do SMOTE: {np.bincount(y_train_balanced)}")
        
        return X_train_balanced, y_train_balanced
    
    def train_logistic_regression(self, X_train, y_train):
        """Treina modelo de Regressão Logística"""
        print("\n" + "="*60)
        print("Treinando Regressão Logística...")
        print("="*60)
        
        model = LogisticRegression(**MODEL_CONFIGS['logistic_regression']['params'])
        model.fit(X_train, y_train)
        
        self.models['logistic_regression'] = model
        print("Regressão Logística treinada com sucesso!")
        
        return model
    
    def train_random_forest(self, X_train, y_train, use_grid_search=True):
        """Treina modelo Random Forest com GridSearch"""
        print("\n" + "="*60)
        print("Treinando Random Forest...")
        print("="*60)
        
        if use_grid_search:
            print("Executando GridSearchCV...")
            rf = RandomForestClassifier(random_state=42, n_jobs=-1)
            grid_search = GridSearchCV(
                rf, 
                GRID_SEARCH_PARAMS['random_forest'],
                cv=5,
                scoring='f1',
                n_jobs=-1,
                verbose=1
            )
            grid_search.fit(X_train, y_train)
            
            model = grid_search.best_estimator_
            print(f"Melhores parâmetros: {grid_search.best_params_}")
            print(f"Melhor score (F1): {grid_search.best_score_:.4f}")
        else:
            model = RandomForestClassifier(**MODEL_CONFIGS['random_forest']['params'])
            model.fit(X_train, y_train)
        
        self.models['random_forest'] = model
        print("Random Forest treinado com sucesso!")
        
        return model
    
    def train_xgboost(self, X_train, y_train, use_grid_search=True):
        """Treina modelo XGBoost com GridSearch"""
        print("\n" + "="*60)
        print("Treinando XGBoost...")
        print("="*60)
        
        if use_grid_search:
            print("Executando GridSearchCV...")
            xgb = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
            grid_search = GridSearchCV(
                xgb,
                GRID_SEARCH_PARAMS['xgboost'],
                cv=5,
                scoring='f1',
                n_jobs=-1,
                verbose=1
            )
            grid_search.fit(X_train, y_train)
            
            model = grid_search.best_estimator_
            print(f"Melhores parâmetros: {grid_search.best_params_}")
            print(f"Melhor score (F1): {grid_search.best_score_:.4f}")
        else:
            model = XGBClassifier(**MODEL_CONFIGS['xgboost']['params'])
            model.fit(X_train, y_train)
        
        self.models['xgboost'] = model
        print("XGBoost treinado com sucesso!")
        
        return model
    
    def evaluate_model(self, model, X_test, y_test, model_name):
        """Avalia um modelo"""
        print(f"\nAvaliando {model_name}...")
        
        # Fazer previsões
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calcular métricas
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba)
        }
        
        # Matriz de confusão
        cm = confusion_matrix(y_test, y_pred)
        
        # Armazenar resultados
        self.results[model_name] = {
            'metrics': metrics,
            'confusion_matrix': cm,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
        
        # Imprimir resultados
        print(f"\nMétricas de {model_name}:")
        for metric, value in metrics.items():
            print(f"  {metric.capitalize()}: {value:.4f}")
        
        print(f"\nMatriz de Confusão:")
        print(cm)
        
        print(f"\nRelatório de Classificação:")
        print(classification_report(y_test, y_pred, target_names=['Stayed', 'Left']))
        
        return metrics
    
    def get_feature_importance(self, model, feature_names, model_name, top_n=15):
        """Obtém feature importance"""
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1][:top_n]
            
            print(f"\nTop {top_n} Features Mais Importantes ({model_name}):")
            for i, idx in enumerate(indices, 1):
                print(f"  {i}. {feature_names[idx]}: {importances[idx]:.4f}")
            
            return pd.DataFrame({
                'feature': [feature_names[i] for i in indices],
                'importance': importances[indices]
            })
        elif hasattr(model, 'coef_'):
            # Para Logistic Regression
            importances = np.abs(model.coef_[0])
            indices = np.argsort(importances)[::-1][:top_n]
            
            print(f"\nTop {top_n} Features Mais Importantes ({model_name}):")
            for i, idx in enumerate(indices, 1):
                print(f"  {i}. {feature_names[idx]}: {importances[idx]:.4f}")
            
            return pd.DataFrame({
                'feature': [feature_names[i] for i in indices],
                'importance': importances[indices]
            })
        else:
            print(f"Modelo {model_name} não suporta feature importance")
            return None
    
    def compare_models(self):
        """Compara todos os modelos treinados"""
        print("\n" + "="*60)
        print("COMPARAÇÃO DE MODELOS")
        print("="*60)
        
        comparison_df = pd.DataFrame({
            name: results['metrics']
            for name, results in self.results.items()
        }).T
        
        print("\n", comparison_df)
        
        # Identificar melhor modelo baseado em F1-Score
        best_model_name = comparison_df['f1'].idxmax()
        self.best_model_name = best_model_name
        self.best_model = self.models[best_model_name]
        
        print(f"\n🏆 Melhor Modelo: {best_model_name}")
        print(f"   F1-Score: {comparison_df.loc[best_model_name, 'f1']:.4f}")
        
        return comparison_df
    
    def save_models(self):
        """Salva todos os modelos treinados"""
        print("\n" + "="*60)
        print("SALVANDO MODELOS")
        print("="*60)
        
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        
        for name, model in self.models.items():
            model_path = MODELS_DIR / f"{name}_model.pkl"
            joblib.dump(model, model_path)
            print(f"✓ {name} salvo em: {model_path}")
        
        # Salvar melhor modelo separadamente
        best_model_path = MODELS_DIR / "best_model.pkl"
        joblib.dump(self.best_model, best_model_path)
        print(f"\n✓ Melhor modelo ({self.best_model_name}) salvo em: {best_model_path}")
        
        # Salvar metadados
        metadata = {
            'best_model': self.best_model_name,
            'training_date': datetime.now().isoformat(),
            'results': {
                name: {k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                       for k, v in results['metrics'].items()}
                for name, results in self.results.items()
            }
        }
        
        import json
        metadata_path = MODELS_DIR / "model_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"✓ Metadados salvos em: {metadata_path}")
    
    def train_all_models(self, use_smote=True, use_grid_search=False):
        """Pipeline completo de treinamento"""
        print("="*60)
        print("PIPELINE DE TREINAMENTO DE MODELOS")
        print("="*60)
        
        # 1. Carregar dados
        X_train, X_test, y_train, y_test = self.load_data()
        
        # Carregar nomes das features
        with open(PROCESSED_DATA_DIR / 'feature_names.txt', 'r') as f:
            feature_names = [line.strip() for line in f.readlines()]
        
        # 2. Aplicar SMOTE (opcional)
        if use_smote:
            X_train, y_train = self.apply_smote(X_train, y_train)
        
        # 3. Treinar modelos
        self.train_logistic_regression(X_train, y_train)
        self.train_random_forest(X_train, y_train, use_grid_search=use_grid_search)
        self.train_xgboost(X_train, y_train, use_grid_search=use_grid_search)
        
        # 4. Avaliar modelos
        for name, model in self.models.items():
            self.evaluate_model(model, X_test, y_test, name)
            self.get_feature_importance(model, feature_names, name)
        
        # 5. Comparar modelos
        comparison_df = self.compare_models()
        
        # 6. Salvar modelos
        self.save_models()
        
        print("\n" + "="*60)
        print("TREINAMENTO CONCLUÍDO COM SUCESSO!")
        print("="*60)
        
        return comparison_df


def main():
    """Função principal"""
    trainer = ModelTrainer()
    
    # Treinar todos os modelos
    # use_grid_search=False para treinar mais rápido (para demonstração)
    # use_grid_search=True para otimização completa (mais lento)
    comparison_df = trainer.train_all_models(use_smote=True, use_grid_search=False)
    
    print("\nPróximos passos:")
    print("1. Executar dashboard: streamlit run app/dashboard.py")
    print("2. Testar API: uvicorn src.api.main:app --reload")
    print("3. Fazer upload para Hugging Face Hub")


if __name__ == "__main__":
    main()

