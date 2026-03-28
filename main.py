import ast
import os
import pickle
import re
from typing import Optional, List, Dict, Any, Tuple

import numpy as np
import pandas as pd
import httpx
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from dotenv import load_dotenv


# =========================
# ENV
# =========================
load_dotenv()
OMDB_API_KEY = os.getenv("OMDB_API_KEY")

OMDB_BASE = "https://www.omdbapi.com"

if not OMDB_API_KEY:
    # Don't crash import-time in production if you prefer; but for you better fail early:
    raise RuntimeError("OMDB_API_KEY missing. Put it in .env as OMDB_API_KEY=xxxx")


# =========================
# FASTAPI APP
# =========================
app = FastAPI(title="Movie Recommender API", version="3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # for local streamlit
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================
# PICKLE GLOBALS
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DF_PATH = os.path.join(BASE_DIR, "df.pkl")
INDICES_PATH = os.path.join(BASE_DIR, "indices.pkl")
TFIDF_MATRIX_PATH = os.path.join(BASE_DIR, "tfidf_matrix.pkl")
TFIDF_PATH = os.path.join(BASE_DIR, "tfidf.pkl")

df: Optional[pd.DataFrame] = None
indices_obj: Any = None
tfidf_matrix: Any = None
tfidf_obj: Any = None

TITLE_TO_IDX: Optional[Dict[str, int]] = None


# =========================
# MODELS
# =========================
class OMDBMovieCard(BaseModel):
    imdb_id: str
    title: str
    poster_url: Optional[str] = None
    year: Optional[str] = None
    imdb_rating: Optional[str] = None


class OMDBMovieDetails(BaseModel):
    imdb_id: str
    title: str
    plot: Optional[str] = None
    year: Optional[str] = None
    poster_url: Optional[str] = None
    genres: Optional[str] = None
    imdb_rating: Optional[str] = None
    actors: Optional[str] = None
    director: Optional[str] = None


class TFIDFRecItem(BaseModel):
    title: str
    score: float
    omdb: Optional[OMDBMovieCard] = None


class SearchBundleResponse(BaseModel):
    query: str
    movie_details: OMDBMovieDetails
    tfidf_recommendations: List[TFIDFRecItem]


# =========================
# UTILS
# =========================
def _norm_title(t: str) -> str:
    return str(t).strip().lower()


async def omdb_get(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Safe OMDB GET:
    - Network errors -> 502
    - OMDB API errors -> 502 with detail
    """
    q = dict(params)
    q["apikey"] = OMDB_API_KEY
    q["type"] = "movie"  # Only search for movies

    try:
        async with httpx.AsyncClient(timeout=20) as client:
            r = await client.get(f"{OMDB_BASE}", params=q)
    except httpx.RequestError as e:
        raise HTTPException(
            status_code=502,
            detail=f"OMDB request error: {type(e).__name__} | {repr(e)}",
        )

    if r.status_code != 200:
        raise HTTPException(
            status_code=502, detail=f"OMDB error {r.status_code}: {r.text}"
        )

    data = r.json()
    
    # Check for OMDB error response
    if data.get("Response") == "False":
        raise HTTPException(
            status_code=404, detail=f"OMDB error: {data.get('Error', 'Unknown error')}"
        )

    return data


async def omdb_cards_from_results(
    results: List[dict], limit: int = 20
) -> List[OMDBMovieCard]:
    """Convert OMDB search results to OMDBMovieCard list"""
    out: List[OMDBMovieCard] = []
    for m in (results or [])[:limit]:
        out.append(
            OMDBMovieCard(
                imdb_id=m.get("imdbID", ""),
                title=m.get("Title", ""),
                poster_url=m.get("Poster") if m.get("Poster") != "N/A" else None,
                year=m.get("Year"),
                imdb_rating=m.get("imdbRating") if m.get("imdbRating") != "N/A" else None,
            )
        )
    return out


async def omdb_movie_details(imdb_id: str) -> OMDBMovieDetails:
    """Fetch full movie details from OMDB by IMDb ID"""
    data = await omdb_get({"i": imdb_id})
    return OMDBMovieDetails(
        imdb_id=data.get("imdbID", imdb_id),
        title=data.get("Title", ""),
        plot=data.get("Plot"),
        year=data.get("Year"),
        poster_url=data.get("Poster") if data.get("Poster") != "N/A" else None,
        genres=data.get("Genre"),  # comma-separated string
        imdb_rating=data.get("imdbRating") if data.get("imdbRating") != "N/A" else None,
        actors=data.get("Actors"),
        director=data.get("Director"),
    )


async def omdb_search_movies(query: str, page: int = 1) -> Dict[str, Any]:
    """
    Search for movies using OMDB API.
    Returns raw OMDB response with 'Search' array.
    """
    return await omdb_get(
        {
            "s": query,
            "page": page,
        }
    )


async def omdb_search_first(query: str) -> Optional[dict]:
    """Get the first search result from OMDB"""
    try:
        data = await omdb_search_movies(query=query, page=1)
        results = data.get("Search", [])
        return results[0] if results else None
    except HTTPException:
        return None


# =========================
# TF-IDF Helpers
# =========================
def build_title_to_idx_map(indices: Any) -> Dict[str, int]:
    """
    indices.pkl can be:
    - dict(title -> index)
    - pandas Series (index=title, value=index)
    We normalize into TITLE_TO_IDX.
    """
    title_to_idx: Dict[str, int] = {}

    if isinstance(indices, dict):
        for k, v in indices.items():
            title_to_idx[_norm_title(k)] = int(v)
        return title_to_idx

    # pandas Series or similar mapping
    try:
        for k, v in indices.items():
            title_to_idx[_norm_title(k)] = int(v)
        return title_to_idx
    except Exception:
        # last resort: if it's a list-like etc.
        raise RuntimeError(
            "indices.pkl must be dict or pandas Series-like (with .items())"
        )


def get_local_idx_by_title(title: str) -> int:
    global TITLE_TO_IDX
    if TITLE_TO_IDX is None:
        raise HTTPException(status_code=500, detail="TF-IDF index map not initialized")
    key = _norm_title(title)
    if key in TITLE_TO_IDX:
        return int(TITLE_TO_IDX[key])
    raise HTTPException(
        status_code=404, detail=f"Title not found in local dataset: '{title}'"
    )


def tfidf_recommend_titles(
    query_title: str, top_n: int = 10
) -> List[Tuple[str, float]]:
    """
    Returns list of (title, score) from local df using cosine similarity on TF-IDF matrix.
    Safe against missing columns/rows.
    """
    global df, tfidf_matrix
    if df is None or tfidf_matrix is None:
        raise HTTPException(status_code=500, detail="TF-IDF resources not loaded")

    idx = get_local_idx_by_title(query_title)

    # query vector
    qv = tfidf_matrix[idx]
    scores = (tfidf_matrix @ qv.T).toarray().ravel()

    # sort descending
    order = np.argsort(-scores)

    out: List[Tuple[str, float]] = []
    for i in order:
        if int(i) == int(idx):
            continue
        try:
            title_i = str(df.iloc[int(i)]["title"])
        except Exception:
            continue
        out.append((title_i, float(scores[int(i)])))
        if len(out) >= top_n:
            break
    return out


def _text_clean(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def create_resources_from_csv() -> Tuple[pd.DataFrame, Any, Any, Any]:
    csv_path = os.path.join(BASE_DIR, "movies_metadata.csv")
    if not os.path.exists(csv_path):
        raise RuntimeError("movies_metadata.csv not found; cannot rebuild pickles")

    df_local = pd.read_csv(csv_path, low_memory=False)
    df_local = df_local.drop_duplicates().reset_index(drop=True)

    for col in ["title", "overview", "genres", "tagline"]:
        if col not in df_local.columns:
            raise RuntimeError(f"movies_metadata.csv missing required column: {col}")

    df_local = df_local[["title", "overview", "genres", "tagline", "vote_average", "popularity"]]
    df_local = df_local.dropna(subset=["title"]).reset_index(drop=True)
    df_local["overview"] = df_local["overview"].fillna("")
    df_local["tagline"] = df_local["tagline"].fillna("")

    def parse_genres(v):
        try:
            items = ast.literal_eval(v)
            return " ".join([i.get("name", "") for i in items if isinstance(i, dict) and i.get("name")])
        except Exception:
            return ""

    df_local["genres"] = df_local["genres"].fillna("").apply(parse_genres)
    df_local["tags"] = (df_local["overview"] + " " + df_local["genres"] + " " + df_local["tagline"]).apply(_text_clean)

    indices_local = pd.Series(df_local.index, index=df_local["title"]).drop_duplicates()

    tfidf_vectorizer = TfidfVectorizer(max_features=50000, ngram_range=(1, 2), stop_words="english")
    tfidf_matrix_local = tfidf_vectorizer.fit_transform(df_local["tags"])

    return df_local, indices_local, tfidf_matrix_local, tfidf_vectorizer


async def attach_omdb_card_by_title(title: str) -> Optional[OMDBMovieCard]:
    """
    Uses OMDB search by title to fetch poster for a local title.
    If not found, returns None (never crashes the endpoint).
    """
    try:
        m = await omdb_search_first(title)
        if not m:
            return None
        return OMDBMovieCard(
            imdb_id=m.get("imdbID", ""),
            title=m.get("Title", title),
            poster_url=m.get("Poster") if m.get("Poster") != "N/A" else None,
            year=m.get("Year"),
            imdb_rating=m.get("imdbRating") if m.get("imdbRating") != "N/A" else None,
        )
    except Exception:
        return None


# =========================
# STARTUP: LOAD PICKLES
# =========================
@app.on_event("startup")
def load_pickles():
    global df, indices_obj, tfidf_matrix, tfidf_obj, TITLE_TO_IDX

    try:
        # Load df
        with open(DF_PATH, "rb") as f:
            df = pickle.load(f)

        # Load indices
        with open(INDICES_PATH, "rb") as f:
            indices_obj = pickle.load(f)

        # Load TF-IDF matrix (usually scipy sparse)
        with open(TFIDF_MATRIX_PATH, "rb") as f:
            tfidf_matrix = pickle.load(f)

        # Load tfidf vectorizer (optional, not used directly here)
        with open(TFIDF_PATH, "rb") as f:
            tfidf_obj = pickle.load(f)

    except Exception as e:
        # If pickles fail (old pandas pickle compatibility issue), rebuild from CSV path
        print(f"Warning: Pickle loading failed ({e}); rebuilding from CSV...")
        df, indices_obj, tfidf_matrix, tfidf_obj = create_resources_from_csv()

        # Optionally persist regenerated pickles for faster startup next time
        try:
            with open(DF_PATH, "wb") as f:
                pickle.dump(df, f)
            with open(INDICES_PATH, "wb") as f:
                pickle.dump(indices_obj, f)
            with open(TFIDF_MATRIX_PATH, "wb") as f:
                pickle.dump(tfidf_matrix, f)
            with open(TFIDF_PATH, "wb") as f:
                pickle.dump(tfidf_obj, f)
        except Exception:
            pass

    # Build normalized map
    TITLE_TO_IDX = build_title_to_idx_map(indices_obj)

    # sanity
    if df is None or "title" not in df.columns:
        raise RuntimeError("df.pkl must contain a DataFrame with a 'title' column")


# =========================
# ROUTES
# =========================
@app.get("/health")
def health():
    return {"status": "ok"}


# ---------- OMDB KEYWORD SEARCH (MULTIPLE RESULTS) ----------
@app.get("/omdb/search")
async def omdb_search(
    query: str = Query(..., min_length=1),
    page: int = Query(1, ge=1, le=10),
):
    """
    Returns RAW OMDB response with 'Search' array.
    Streamlit will use it for:
      - dropdown suggestions
      - grid results
    """
    return await omdb_search_movies(query=query, page=page)


# ---------- MOVIE DETAILS (SAFE ROUTE) ----------
@app.get("/movie/id/{imdb_id}", response_model=OMDBMovieDetails)
async def movie_details_route(imdb_id: str):
    return await omdb_movie_details(imdb_id)


# ---------- TF-IDF ONLY (debug/useful) ----------
@app.get("/recommend/tfidf")
async def recommend_tfidf(
    title: str = Query(..., min_length=1),
    top_n: int = Query(10, ge=1, le=50),
):
    recs = tfidf_recommend_titles(title, top_n=top_n)
    return [{"title": t, "score": s} for t, s in recs]


# ---------- BUNDLE: Details + TF-IDF recs ----------
@app.get("/movie/search", response_model=SearchBundleResponse)
async def search_bundle(
    query: str = Query(..., min_length=1),
    tfidf_top_n: int = Query(12, ge=1, le=30),
):
    """
    This endpoint is for when you have a selected movie and want:
      - movie details (from OMDB)
      - TF-IDF recommendations (local) + posters (from OMDB)

    NOTE:
    - It selects the BEST match from OMDB for the given query.
    - If you want MULTIPLE matches, use /omdb/search
    """
    best = await omdb_search_first(query)
    if not best:
        raise HTTPException(
            status_code=404, detail=f"No OMDB movie found for query: {query}"
        )

    imdb_id = best["imdbID"]
    details = await omdb_movie_details(imdb_id)

    # 1) TF-IDF recommendations (never crash endpoint)
    tfidf_items: List[TFIDFRecItem] = []

    recs: List[Tuple[str, float]] = []
    try:
        # try local dataset by OMDB title
        recs = tfidf_recommend_titles(details.title, top_n=tfidf_top_n)
    except Exception:
        # fallback to user query
        try:
            recs = tfidf_recommend_titles(query, top_n=tfidf_top_n)
        except Exception:
            recs = []

    for title, score in recs:
        card = await attach_omdb_card_by_title(title)
        tfidf_items.append(TFIDFRecItem(title=title, score=score, omdb=card))

    return SearchBundleResponse(
        query=query,
        movie_details=details,
        tfidf_recommendations=tfidf_items,
    )