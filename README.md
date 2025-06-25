# Credit Risk Prediction

A production-ready machine learning project for predicting credit risk using FastAPI, Docker, and CI/CD best practices.

## Credit Scoring Business Understanding

### Basel II & Interpretability
Under Basel II’s IRB approach, banks must quantify credit risk (Probability of Default, Loss Given Default, Exposure at Default) and hold capital proportional to that risk. This framework imposes strict validation and documentation requirements for any internal scoring model. In practice, regulators expect credit models to be transparent and well‐documented so that risk exposures are understandable. We therefore favor a scorecard-style model where each feature’s effect is clear, ensuring the model can be fully audited and justified to regulators.

### Proxy Default Label
Our data likely lacks a direct “default” flag (common in new lending streams), so we must define a proxy target. For example, we may label accounts as “bad” if they hit a threshold (e.g. >90 days delinquent) in a defined period. Creating such a proxy is necessary to train a PD model, but it introduces risk: if the proxy doesn’t align with true default, the model can misclassify borrowers. Misalignment may lead to false positives (denying credit to good customers) or false negatives (undetected losses), hurting profitability and customer relationships. In short, any bias or noise in the proxy definition propagates into our risk predictions, so we must monitor and validate that the proxy remains a reliable stand‐in for actual default events.

### Model Complexity Trade-offs
Simple models (e.g. logistic regression with weight-of-evidence coding) score well for interpretability. They produce a monotonic scorecard that lenders understand and can explain. Such models are easier to document and meet regulatory scrutiny. More complex models (e.g. Gradient Boosting Machines) often yield better predictive accuracy, especially on high-dimensional or non-linear data, but at the cost of opacity. In a regulated banking context, this trade-off means we typically sacrifice a bit of accuracy to maintain transparency and compliance. Complex models require additional explainability tools (SHAP/LIME) and extensive validation. In emerging markets, studies have found that even though ML models can outperform, they are rarely adopted at scale due to concerns about model interpretability, regulatory acceptance, and institutional readiness. Thus, we must balance performance against the need for a model that auditors and regulators can inspect and understand.

## Project Structure

```
├── .github/workflows/ci.yml   # CI/CD workflow
├── data/                      # Data folder (excluded from git)
│   ├── raw/                   # Raw data
│   └── processed/             # Processed data for training
├── notebooks/
│   └── 1.0-eda.ipynb          # Exploratory Data Analysis
├── src/
│   ├── __init__.py
│   ├── data_processing.py     # Feature engineering
│   ├── train.py               # Model training
│   ├── predict.py             # Inference
│   └── api/
│       ├── main.py            # FastAPI app
│       └── pydantic_models.py # Pydantic models
├── tests/
│   └── test_data_processing.py # Unit tests
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .gitignore
```

## Features
- Modular codebase for data processing, training, and inference
- REST API for predictions using FastAPI
- Dockerized for easy deployment
- GitHub Actions for CI/CD
- Unit tests for reliability

## Getting Started

### Prerequisites
- Python 3.10+
- Docker & Docker Compose (optional, for containerization)

### Installation
1. Clone the repository:
   ```bash
   git clone <repo-url>
   cd Credit-Risk-Prediction
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the API
- Locally:
  ```bash
  uvicorn src.api.main:app --reload
  ```
- With Docker:
  ```bash
  docker-compose up --build
  ```

### Data
- Place raw data in `data/raw/`.
- Processed data will be saved in `data/processed/`.

### Training
Edit and run `src/train.py` to train models on processed data.

### Testing
Run unit tests with:
```bash
pytest tests/
```

## Contributing
1. Create a new branch: `git checkout -b feature-name`
2. Commit your changes
3. Open a pull request

## License
MIT License

---

**Credit Risk Prediction** — built with FastAPI, Docker, and ❤️
