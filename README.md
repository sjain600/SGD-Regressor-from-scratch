# Neural Collaborative Filtering Recommender System

This project implements a deep learning-based recommender system using Neural Matrix Factorization (NeuMF). It combines classical collaborative filtering with neural networks to learn complex user–item interactions and generate personalized movie recommendations.

The system is deployed as an interactive application using Gradio and Hugging Face Spaces.

---

## Overview

Recommender systems are widely used in platforms like Netflix and Amazon to personalize user experience by predicting user preferences based on historical interactions.

This project focuses on:

- Collaborative filtering using latent factor models
- Extending matrix factorization using neural networks
- Building a production-ready recommendation interface

---

## Models Implemented

### 1. Matrix Factorization

Based on the foundational work from:

- Koren et al., Matrix Factorization Techniques for Recommender Systems: 

Key idea:
- Users and items are mapped to a shared latent space
- Predictions are computed using the dot product of embeddings


Limitations:
- Assumes linear interaction between user and item features
- Struggles to capture complex relationships

---

### 2. Neural Collaborative Filtering (NCF)

Based on:

- He et al., Neural Collaborative Filtering :contentReference

Key improvements:
- Replaces inner product with a neural network
- Learns non-linear user–item interactions
- Uses embeddings + multi-layer perceptron

---

### 3. Neural Matrix Factorization (NeuMF)

Final model used in this project.

Combines:
- Generalized Matrix Factorization (GMF)
- Multi-Layer Perceptron (MLP)

Advantages:
- Captures both linear and non-linear interactions
- Improves recommendation accuracy significantly over traditional MF

---

## Architecture

The model consists of:

- User embedding layer
- Item embedding layer
- GMF pathway (element-wise interaction)
- MLP pathway (deep neural network)
- Final prediction layer (sigmoid output)

---

## Dataset

- MovieLens dataset
- Contains user ratings and movie metadata
- Converted into implicit feedback for training

---

## Project Structure

```bash
Recommender-System/
│
├── app.py                 # Gradio app for deployment
├── neumf.pth              # Trained model weights
├── movies.csv             # Movie metadata
├── ratings.csv            # User-item interactions
├── movie_ids.pkl          # Encoded movie mapping
├── user_map.pkl           # Encoded user mapping
├── requirements.txt       # Dependencies
│
├── notebooks/             # Model development and experiments
│   ├── training.ipynb
│   └── evaluation.ipynb
│
├── src/                   # Modular code (recommended improvement)
│   ├── model.py
│   ├── data_loader.py
│   ├── inference.py
│   └── utils.py
│
└── README.md
