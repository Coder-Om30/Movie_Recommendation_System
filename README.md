# 🎬 Movie Recommendation System

A full-stack **Movie Recommendation Web App** built using **FastAPI**, **Streamlit**, and **TF-IDF Machine Learning**.
It allows users to search movies, view details, and get intelligent recommendations based on content similarity.

---

## 🚀 Features

* 🔍 Search movies using keywords (OMDB API)
* 🎯 Smart recommendations using **TF-IDF similarity**
* 🖼️ Movie posters and details (plot, genre, rating, etc.)
* 📊 Clean UI with responsive grid layout
* ⚡ Fast backend with optimized API calls
* 🔄 Real-time suggestions + autocomplete

---

## 🏗️ Tech Stack

### Frontend

* **Streamlit**
* Python
* Requests

### Backend

* **FastAPI**
* Python
* HTTPX (async API calls)
* Pydantic

### Machine Learning

* TF-IDF Vectorizer
* Cosine Similarity
* Pandas, NumPy, Scikit-learn

### APIs

* 🎥 OMDB API (movie data)

---

## 📂 Project Structure

```
📁 Movie-Recommendation-System
│
├── main.py          # FastAPI backend
├── app.py           # Streamlit frontend
├── movies.ipynb     # Model training / experimentation
├── df.pkl           # Movie dataset
├── tfidf.pkl        # TF-IDF vectorizer
├── tfidf_matrix.pkl # TF-IDF matrix
├── indices.pkl      # Title index mapping
├── .env             # API keys
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the repository

```bash
git clone https://github.com/your-username/movie-recommendation-system.git
cd movie-recommendation-system
```

### 2️⃣ Create virtual environment

```bash
python -m venv venv
source venv/bin/activate  # (Linux/Mac)
venv\Scripts\activate     # (Windows)
```

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Add OMDB API key

Create a `.env` file:

```env
OMDB_API_KEY=your_api_key_here
```

---

## ▶️ Running the Project

### Start Backend (FastAPI)

```bash
uvicorn main:app --reload
```

### Start Frontend (Streamlit)

```bash
streamlit run app.py
```

---

## 🔗 API Endpoints

### 🎥 Search Movies

```
GET /omdb/search?query=batman
```

### 📄 Movie Details

```
GET /movie/id/{imdb_id}
```

### 🎯 Recommendations

```
GET /movie/search?query=batman
```

### ❤️ Health Check

```
GET /health
```

---

## 🧠 How It Works

* Movie metadata is processed and converted into text features
* TF-IDF vectorizer transforms text into numerical vectors
* Cosine similarity is used to find similar movies
* Backend combines:

  * OMDB data (posters, details)
  * Local ML model (recommendations)

---

## 📸 UI Overview

* Home page → trending/popular movies
* Search → autocomplete + results grid
* Details page → movie info + recommendations

---

## 📌 Key Highlights

* 🔥 Hybrid system (API + ML model)
* ⚡ Fast caching using Streamlit
* 🧩 Clean modular architecture
* 🌐 Deployable (Render / Streamlit Cloud)

---

## 👨‍💻 Author

**Om Sahu**

---

## ⭐ Support

If you like this project, give it a ⭐ on GitHub!
