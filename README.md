# Text and Image-Based Movie Recommendation System

## Introduction
This project develops a text and image-based movie recommendation system that utilizes various embedding techniques, including Annoy (Approximate Nearest Neighbors), Bag of Words (BoW), and DistilBERT, to provide fast and accurate recommendations. The system is designed to be interacted with through a Gradio web interface, making it accessible for users to query and receive recommendations. The backend is built with Python and can be easily deployed using Docker, ensuring compatibility and ease of setup across different environments.


## Table of Contents
1. [Introduction](#introduction)
2. [Features](#features)
3. [Installation](#installation)
   - [Docker Setup](#docker-setup)
   - [Python Environment](#python-environment)
4. [Usage](#usage)
   - [Running the Application](#running-the-application)
   - [Interacting with the Web Interface](#interacting-with-the-web-interface)
5. [Dependencies](#dependencies)

## Features
- **Annoy Index for Fast Similarity Search**: Utilizes Annoy to quickly find the nearest neighbors in high-dimensional spaces, speeding up the recommendation process.
- **Multiple Embedding Techniques**: Supports Bag of Words and DistilBERT embeddings, allowing for flexible and effective text representation.
- **Gradio Web Interface**: Offers an easy-to-use web interface for users to input queries and receive text-based recommendations.
- **Docker Support**: Includes Docker and Docker Compose configurations for straightforward deployment and scalability.
- **API for Recommendation System**: A comprehensive API is designed to handle requests and integrate the various components of the system.

## Installation

### Docker Setup
The project includes Dockerfile configurations for both the Annoy index service and the Gradio app, as well as a docker-compose.yml file for easy orchestration of the services.

### Python Environment
To run the project outside Docker, Python 3.8 or later is required. Dependencies can be installed via the provided `requirements.txt` file.

## Usage

### Running the Application

Navigate to the Project :

```
cd Projet_AIF/
```

and run Docker:
```bash
docker-compose up
```

### Interacting with the Web Interface

Once the application is running, navigate to the Gradio web interface URL (http://localhost:7860/) displayed in the console to input queries and receive recommendations. The Gradio interface may take some time to become available; please allow a few moments for it to fully load.


## Dependencies
Dependencies are listed in the requirements.txt file, including Flask for the API, Gradio for the web interface, and Hugging Face Transformers for DistilBERT embeddings, among others.
