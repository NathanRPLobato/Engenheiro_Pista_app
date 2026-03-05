
# 🏎️ Engenheiro de Pista — AI Racing Strategy System

Sistema de análise de telemetria, modelagem de degradação de pneus, previsão de pace e simulação de estratégia de corrida inspirado em pipelines utilizados em engenharia de performance no motorsport.

O projeto implementa um pipeline completo de Machine Learning para corrida incluindo:

- ingestão de telemetria
- engenharia de features
- modelagem de degradação
- modelagem de risco e pace
- fingerprint comportamental de pilotagem
- recomendação de setup
- simulação de estratégias
- API e dashboard

O objetivo é criar um engenheiro de pista virtual baseado em dados.

---

# Arquitetura do Sistema

Telemetria → Data Ingestion → Feature Engineering → Model Training → Race Simulation → API → Dashboard

Modelos principais:

- Fingerprint Model (Autoencoder)
- Tire Degradation Model (TCN)
- Risk/Pace Model (LightGBM)
- Setup Surrogate Model

---

# Estrutura do Projeto

src/

api/
- main.py
- db.py
- schemas.py
- services/

data/
- ingest_raw_to_sqlite.py
- clean_sqlite_laps.py
- make_views.py

sim/
- simulate_race_fast.py
- simulate_race_strategy.py

train/

core/
- build_lap_features.py
- build_windows.py
- build_risk_pace_priors.py
- build_lap_degradation_dataset.py
- build_fingerprint_lap_table.py

fingerprint/
- autoencoder_fingerprint.py

degradation/
- train_degradation_tcn_multitask_dl.py

risk_pace/
- model_lgbm_risk_pace_by_track_gpu.py

setup/
- train_setup_surrogate_dl.py
- optimize_setup.py
- recommend_setup.py

webdash/
- index.html
- app.js
- styles.css

models/
champion/
- fingerprint_ae/
- degradation_tcn/
- risk_pace/
- risk_pace_by_track/

---

# Principais Modelos

## Fingerprint Model (Autoencoder)

Arquivo:
train/fingerprint/autoencoder_fingerprint.py

Objetivo:
Criar um vetor latente que representa o estilo de pilotagem.

Entradas típicas:

- throttle
- brake
- steering
- speed
- rpm
- lateral_g
- longitudinal_g

Saída:

driver_fingerprint_vector

Uso:

- caracterização de piloto
- comparação de estilo
- entrada para modelos de estratégia

---

## Tire Degradation Model (TCN)

Arquivo:
train/degradation/train_degradation_tcn_multitask_dl.py

Modelo:
Temporal Convolutional Network

Prediz:

- lap_time_delta
- tyre_degradation
- pace_loss

Entrada:

- histórico de voltas
- idade do pneu
- combustível
- temperatura da pista
- composto

Motivo de usar TCN em vez de LSTM:

- maior paralelização
- treinamento mais estável
- melhor captura de dependência temporal longa

---

## Risk Pace Model (LightGBM)

Arquivo:
train/risk_pace/model_lgbm_risk_pace_by_track_gpu.py

Prediz:

- lap_time esperado
- risco de estratégia
- risco de pace

Treinado por pista:

models/champion/risk_pace_by_track/

---

## Setup Surrogate Model

Arquivos:

train/setup/train_setup_surrogate_dl.py
train/setup/optimize_setup.py

Entrada:

- wing
- ride_height
- suspension
- differential
- camber
- toe

Saída:

- predicted_lap_time
- stability_score
- tyre_wear_impact

---

# Pipeline de Treinamento

1) Ingestão de Telemetria

python src/data/ingest_raw_to_sqlite.py

2) Limpeza dos dados

python src/data/clean_sqlite_laps.py

3) Criar features

python src/train/core/build_lap_features.py

4) Criar janelas temporais

python src/train/core/build_windows.py

5) Dataset de degradação

python src/train/core/build_lap_degradation_dataset.py

6) Treinar fingerprint

python src/train/fingerprint/autoencoder_fingerprint.py

7) Treinar degradação

python src/train/degradation/train_degradation_tcn_multitask_dl.py

8) Criar priors

python src/train/core/build_risk_pace_priors.py

9) Treinar risk pace

python src/train/risk_pace/model_lgbm_risk_pace_by_track_gpu.py

10) Treinar modelo de setup

python src/train/setup/train_setup_surrogate_dl.py

---

# Simulação de Corrida

Arquivo:

src/sim/simulate_race_strategy.py

Executar:

python src/sim/simulate_race_strategy.py

Modelo de combustível:

fuel(t+1) = fuel(t) - burn(t)

---

# API

API construída com FastAPI.

Arquivo:

src/api/main.py

Executar:

uvicorn src.api.main:app --reload

Endpoints:

/simulate
/predict_degradation
/recommend_setup
/fingerprint_driver

---

# Dashboard

Local:

src/webdash/

Abrir index.html no navegador.

---

# Requisitos

Python 3.10+

Instalar dependências:

pip install -r requirements.txt

Principais bibliotecas:

- tensorflow
- lightgbm
- pandas
- numpy
- scikit-learn
- fastapi
- uvicorn
- sqlite

---

# Hardware Recomendado

CPU: 8+ cores
RAM: 16GB
GPU: NVIDIA CUDA

---

# Métricas Utilizadas

- MAE lap time
- RMSE lap time
- R²

Arquivos de métricas:

models/champion/*/metrics.json

---

# Futuras Melhorias

- integração direta com FastF1
- modelo de safety car
- modelo de tráfego
- reinforcement learning para estratégia
- modelo térmico de pneus

---

# 📞 Contato
Nathan Rafael Pedroso Lobato ✉️ nathan.lobato@outlook.com.br

---

# 📄 Licença
Este projeto está sob a licença MIT — livre para uso, modificação e distribuição.
