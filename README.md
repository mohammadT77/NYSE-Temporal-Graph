# NYSE Graph: Stock Price Prediction Using Graph Neural Networks

This repository contains the implementation of a project focused on constructing graphs from NYSE stock data and predicting stock price changes using Graph Neural Networks (GNNs). The project integrates static and temporal graph construction with end-to-end learning for predictive modeling.

## Table of Contents

- [NYSE Graph: Stock Price Prediction Using Graph Neural Networks](#nyse-graph-stock-price-prediction-using-graph-neural-networks)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
  - [Motivation](#motivation)
  - [Data](#data)
  - [Methodology](#methodology)
    - [Static Graph Prediction](#static-graph-prediction)
    - [Temporal Graph Prediction](#temporal-graph-prediction)
    - [End-to-End Learning](#end-to-end-learning)
  - [Experiments](#experiments)
  - [Setup and Usage](#setup-and-usage)
  - [References](#references)
  - [Contribution](#contribution)

---

## Introduction

This project explores graph-based analysis of NYSE stock data for predicting price changes. By leveraging GNNs, the project aims to uncover interdependencies between stocks and dynamic market behaviors. Graphs are constructed based on daily stock price changes using similarity measures, and prediction is performed on both static and temporal graphs.

## Motivation

Key questions addressed include:
- How do price changes in one stock influence others in the same or different categories?
- Can future price changes be predicted using both a stock's historical data and data from other stocks?
- Can similarities in price behavior be used to define weighted edges between stocks?

## Data

The dataset, sourced from [Kaggle](https://www.kaggle.com/datasets/dgawlik/nyse), contains:
- Daily stock price records (2010–2016) with Open, Close, High, Low prices, and Volume.
- Over 851K records for 501 unique stock symbols.

New features such as `daily_change` were engineered for graph construction.

---

## Methodology

### Static Graph Prediction
- **Graph Construction:** A static graph is built using the Pearson Correlation Coefficient to measure price similarities.
- **Prediction:** A non-learnable static predictor aggregates weighted neighbor messages to estimate price changes.

### Temporal Graph Prediction
- **Graph Construction:** A series of daily graph snapshots forms a dynamic temporal graph. Edges are weighted using a custom logarithmic similarity function.
- **Prediction:** A Recurrent Graph Neural Network (RGNN) with an encoder-decoder architecture predicts node-level price changes.

### End-to-End Learning
- **GraphConstructor Module:** A trainable module replaces the hand-crafted similarity function with a multi-head attention mechanism.
- **Prediction Module:** Predictions are made using temporal graph snapshots and attention-based adjacency matrices.

---

## Experiments

- **Baselines:** Static and temporal predictors were compared using sector-based graphs as a reference.
- **Training:** Models were trained with varying data splits and thresholds to analyze performance.
- **Results:** Challenges such as data insufficiency and reliance on similarity functions highlighted areas for future improvement.

---

## Setup and Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/mohammadT77/NYSE-Temporal-Graph.git
   cd NYSE-Temporal-Graph
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Explore notebooks for static and temporal graph construction and prediction:
   - Static Graph: `static_pearson.ipynb`
   - Temporal Graph: `temporal_graph_construction.ipynb`, `temporal_graph_prediction.ipynb`

4. Run experiments and analyze results:
   ```bash
   python train.py
   ```

---

## References

1. [Attention Is All You Need](https://arxiv.org/abs/1706.03762)  
2. [Multi-Head Attention](https://paperswithcode.com/method/multi-head-attention)  
3. [Kaggle NYSE Dataset](https://www.kaggle.com/datasets/dgawlik/nyse)  

---

## Contribution

Developed by MohammadAmin Hajibagher Tehran (100%).

--- 

Let me know if you’d like to customize it further!