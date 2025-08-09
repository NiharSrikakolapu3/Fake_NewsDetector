import pandas as pd
from newsapi import NewsApiClient

# This is just my News API KEY
API_KEY = "151c0e1278ca4a69ab5a7468727d6d39"

# Initialize NewsAPI client
newsapi=NewsApiClient(api_key=API_KEY)

# My Models were trained on lots of political news so I want to use the NewsAPi to establish a connection and get lots of political headlines
def fetch_political_headlines(query="biden OR trump OR election OR congress OR senate OR government OR 'white house'", 
                              language="en", 
                              page_size=20, 
                              max_pages=1):
    
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

            for article in articles:
                title = article.get("title", "")
                description = article.get("description", "") or ""
                text = f"{title} {description}".strip()

                all_articles.append({
                    "text": text,
                    "source": article.get("source", {}).get("name", "Unknown")
                })

    except Exception as e:
        print("Error fetching news:", e)

    return pd.DataFrame(all_articles)
