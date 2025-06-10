# StackOverflow Keywords Prediction API 🚀

Cette API prédit les tags (mots clés) d'une question StackOverflow.

## 📦 Fonctionnalité

- Classification multilabel
- Pipeline avec vectoriseur TF-IDF
- Modèle `LogisticRegression` avec OneVsRestClassifier
- API via FastAPI

## 🧪 Test

```bash
pytest tests/# stackoverflow-tag-predictor
