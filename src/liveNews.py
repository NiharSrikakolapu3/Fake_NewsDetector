import pandas as pd
from newsapi import NewsApiClient

# Your NewsAPI key
API_KEY = "151c0e1278ca4a69ab5a7468727d6d39"

# Initialize NewsAPI client
newsapi = NewsApiClient(api_key=API_KEY)

def fetch_political_headlines(query="biden OR trump OR election OR congress OR senate OR government OR 'white house'", 
                              language="en", 
                              page_size=20, 
                              max_pages=1):
    """
    Fetches recent political news articles based on the query.
    Returns a DataFrame with 'text' and 'source'.
    """
    all_articles = []

    try:
        for page in range(1, max_pages + 1):
            response = newsapi.get_everything(
                q=query,
                language=language,
                sort_by="publishedAt",
                page_size=page_size,
                page=page
            )

            articles = response.get("articles", [])
            if not articles:
                break  # Stop if no articles are returned

            for art in articles:
                title = art.get("title", "")
                description = art.get("description", "") or ""
                text = f"{title} {description}".strip()

                all_articles.append({
                    "text": text,
                    "source": art.get("source", {}).get("name", "Unknown")
                })

    except Exception as e:
        print("Error fetching news:", e)

    return pd.DataFrame(all_articles)
