"""
Integração com Hugging Face Hub para versionamento de modelos
"""
import os
import sys
import joblib
from pathlib import Path
from datetime import datetime

# Adicionar o diretório raiz ao path
sys.path.append(str(Path(__file__).parent.parent.parent))

from config.config import MODELS_DIR, HUGGINGFACE_TOKEN, HUGGINGFACE_REPO

class HuggingFaceModelHub:
    """Classe para integração com Hugging Face Hub"""
    
    def __init__(self, token: str = None, repo_id: str = None):
        """
        Inicializa o cliente do Hugging Face Hub
        
        Args:
            token: Token de autenticação do Hugging Face
            repo_id: ID do repositório (username/repo-name)
        """
        self.token = token or HUGGINGFACE_TOKEN
        self.repo_id = repo_id or HUGGINGFACE_REPO
        
        if not self.token:
            print("⚠️  HUGGINGFACE_TOKEN não configurado no .env")
            print("   Para usar esta funcionalidade, configure o token em .env")
            self.enabled = False
        else:
            self.enabled = True
            self._setup_client()
    
    def _setup_client(self):
        """Configura o cliente do Hugging Face"""
        try:
            from huggingface_hub import HfApi, login
            
            # Fazer login
            login(token=self.token, add_to_git_credential=True)
            
            # Criar API client
            self.api = HfApi()
            
            print("✓ Conectado ao Hugging Face Hub")
            print(f"  Repositório: {self.repo_id}")
            
        except ImportError:
            print("⚠️  huggingface_hub não instalado")
            print("   Execute: pip install huggingface-hub")
            self.enabled = False
        except Exception as e:
            print(f"❌ Erro ao conectar ao Hugging Face: {e}")
            self.enabled = False
    
    def create_model_card(self, model_name: str, metrics: dict) -> str:
        """
        Cria um Model Card para documentação
        
        Args:
            model_name: Nome do modelo
            metrics: Métricas do modelo
            
        Returns:
            Conteúdo do Model Card em Markdown
        """
        card_content = f"""---
license: apache-2.0
tags:
- hr-analytics
- employee-attrition
- classification
- scikit-learn
- xgboost
datasets:
- synthetic-employee-attrition
metrics:
- accuracy
- precision
- recall
- f1
- roc-auc
---

# HR Analytics - Employee Attrition Prediction Model

## Model Description

This model predicts employee attrition (turnover) based on various HR features including job satisfaction, work-life balance, salary, performance rating, and more.

**Model Type:** {model_name}
**Task:** Binary Classification
**Language:** Python
**Framework:** scikit-learn / XGBoost

## Intended Use

This model is designed for HR analytics and workforce planning. It helps identify employees at risk of leaving the organization, enabling proactive retention strategies.

### Primary Use Cases:
- Predicting employee turnover risk
- Identifying key factors influencing attrition
- Supporting strategic HR decision-making
- Workforce planning and retention programs

## Performance Metrics

| Metric | Score |
|--------|-------|
| Accuracy | {metrics.get('accuracy', 'N/A'):.4f} |
| Precision | {metrics.get('precision', 'N/A'):.4f} |
| Recall | {metrics.get('recall', 'N/A'):.4f} |
| F1-Score | {metrics.get('f1', 'N/A'):.4f} |
| ROC-AUC | {metrics.get('roc_auc', 'N/A'):.4f} |

## Training Data

**Dataset:** Synthetic Employee Attrition Dataset
**Size:** 10,000 samples
**Features:** 15 features including:
- Job Satisfaction
- Work-Life Balance
- Monthly Income
- Years at Company
- Number of Promotions
- Performance Rating
- And more...

**Target Variable:** Attrition (0 = Stayed, 1 = Left)

## Model Training

- **Preprocessing:** StandardScaler normalization
- **Class Balancing:** SMOTE (Synthetic Minority Over-sampling Technique)
- **Train/Test Split:** 80/20
- **Cross-Validation:** 5-fold CV

## Usage

```python
import joblib
from huggingface_hub import hf_hub_download

# Download model
model_path = hf_hub_download(
    repo_id="{self.repo_id}",
    filename="{model_name}_model.pkl"
)

# Load model
model = joblib.load(model_path)

# Make predictions
prediction = model.predict(X)
probability = model.predict_proba(X)[:, 1]
```

## Limitations

- This model is trained on synthetic data and may not generalize perfectly to real-world scenarios
- Performance may vary across different industries and organizational contexts
- Should be used as a decision support tool, not as the sole basis for HR decisions
- Regular retraining is recommended as workforce dynamics change

## Ethical Considerations

- **Privacy:** Ensure employee data is anonymized and handled according to GDPR/LGPD
- **Bias:** Monitor for potential biases in predictions across demographic groups
- **Transparency:** Communicate model usage to employees and stakeholders
- **Human Oversight:** Final decisions should involve human judgment

## Citation

```bibtex
@misc{{hr_analytics_model,
  author = {{HR Analytics System}},
  title = {{Employee Attrition Prediction Model}},
  year = {{2025}},
  publisher = {{Hugging Face}},
  howpublished = {{\\url{{{self.repo_id}}}}}
}}
```

## Model Card Authors

HR Analytics System Development Team

## Model Card Contact

For questions or feedback, please open an issue in the repository.

---

**Last Updated:** {datetime.now().strftime('%Y-%m-%d')}
"""
        return card_content
    
    def upload_model(self, model_path: Path, model_name: str, 
                    metrics: dict = None, commit_message: str = None):
        """
        Faz upload de um modelo para o Hugging Face Hub
        
        Args:
            model_path: Caminho do arquivo do modelo
            model_name: Nome do modelo
            metrics: Métricas do modelo
            commit_message: Mensagem do commit
        """
        if not self.enabled:
            print("⚠️  Integração com Hugging Face não habilitada")
            return
        
        try:
            from huggingface_hub import upload_file
            
            # Criar repositório se não existir
            try:
                self.api.create_repo(
                    repo_id=self.repo_id,
                    repo_type="model",
                    exist_ok=True
                )
                print(f"✓ Repositório criado/verificado: {self.repo_id}")
            except Exception as e:
                print(f"⚠️  Aviso ao criar repositório: {e}")
            
            # Upload do modelo
            print(f"\n📤 Fazendo upload de {model_name}...")
            
            upload_file(
                path_or_fileobj=str(model_path),
                path_in_repo=model_path.name,
                repo_id=self.repo_id,
                repo_type="model",
                commit_message=commit_message or f"Upload {model_name} - {datetime.now().strftime('%Y-%m-%d %H:%M')}"
            )
            
            print(f"✓ Modelo {model_name} enviado com sucesso!")
            
            # Criar e fazer upload do Model Card
            if metrics:
                print(f"\n📝 Criando Model Card...")
                card_content = self.create_model_card(model_name, metrics)
                
                # Salvar temporariamente
                card_path = MODELS_DIR / "README.md"
                with open(card_path, 'w') as f:
                    f.write(card_content)
                
                # Upload do Model Card
                upload_file(
                    path_or_fileobj=str(card_path),
                    path_in_repo="README.md",
                    repo_id=self.repo_id,
                    repo_type="model",
                    commit_message="Update Model Card"
                )
                
                print(f"✓ Model Card criado e enviado!")
            
            print(f"\n🎉 Upload concluído!")
            print(f"   Visualize em: https://huggingface.co/{self.repo_id}")
            
        except Exception as e:
            print(f"❌ Erro ao fazer upload: {e}")
    
    def upload_all_models(self):
        """Faz upload de todos os modelos treinados"""
        if not self.enabled:
            print("⚠️  Integração com Hugging Face não habilitada")
            print("\nPara habilitar:")
            print("1. Crie uma conta em https://huggingface.co")
            print("2. Gere um token em https://huggingface.co/settings/tokens")
            print("3. Configure HUGGINGFACE_TOKEN no arquivo .env")
            print("4. Configure HUGGINGFACE_REPO com seu username/repo-name")
            return
        
        print("="*60)
        print("UPLOAD DE MODELOS PARA HUGGING FACE HUB")
        print("="*60)
        
        # Carregar metadados
        import json
        metadata_path = MODELS_DIR / "model_metadata.json"
        
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        else:
            metadata = {'results': {}}
        
        # Upload de cada modelo
        for model_file in MODELS_DIR.glob("*_model.pkl"):
            model_name = model_file.stem.replace('_model', '')
            metrics = metadata.get('results', {}).get(model_name, {})
            
            self.upload_model(
                model_path=model_file,
                model_name=model_name,
                metrics=metrics
            )
        
        # Upload do melhor modelo
        best_model_path = MODELS_DIR / "best_model.pkl"
        if best_model_path.exists():
            best_model_name = metadata.get('best_model', 'best')
            best_metrics = metadata.get('results', {}).get(best_model_name, {})
            
            self.upload_model(
                model_path=best_model_path,
                model_name='best',
                metrics=best_metrics
            )
        
        print("\n" + "="*60)
        print("UPLOAD CONCLUÍDO!")
        print("="*60)
    
    def download_model(self, filename: str, local_dir: Path = None):
        """
        Baixa um modelo do Hugging Face Hub
        
        Args:
            filename: Nome do arquivo do modelo
            local_dir: Diretório local para salvar
        """
        if not self.enabled:
            print("⚠️  Integração com Hugging Face não habilitada")
            return None
        
        try:
            from huggingface_hub import hf_hub_download
            
            local_dir = local_dir or MODELS_DIR
            
            print(f"📥 Baixando {filename} do Hugging Face Hub...")
            
            model_path = hf_hub_download(
                repo_id=self.repo_id,
                filename=filename,
                local_dir=str(local_dir)
            )
            
            print(f"✓ Modelo baixado: {model_path}")
            return model_path
            
        except Exception as e:
            print(f"❌ Erro ao baixar modelo: {e}")
            return None


def main():
    """Função principal para demonstração"""
    print("Integração com Hugging Face Hub")
    print("="*60)
    
    # Criar cliente
    hf_hub = HuggingFaceModelHub()
    
    if hf_hub.enabled:
        # Upload de todos os modelos
        hf_hub.upload_all_models()
    else:
        print("\n⚠️  Configure o Hugging Face Hub para usar esta funcionalidade")
        print("\nInstruções:")
        print("1. Crie uma conta em https://huggingface.co")
        print("2. Gere um token em https://huggingface.co/settings/tokens")
        print("3. Adicione ao .env:")
        print("   HUGGINGFACE_TOKEN=seu_token_aqui")
        print("   HUGGINGFACE_REPO=seu_username/hr-analytics-models")


if __name__ == "__main__":
    main()

