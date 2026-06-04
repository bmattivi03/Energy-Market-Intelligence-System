import openmeteo_requests
import requests_cache
import pandas as pd
from retry_requests import retry

def fetch_weather(start_date: str, end_date: str, locations: dict) -> pd.DataFrame:
    """
    start_date, end_date format: "YYYY-MM-DD"
    locations: dict of {name: (lat, lon)}
    """
    cache_session = requests_cache.CachedSession(".cache", expire_after=-1)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)

    all_locations_data = []

    for name, (lat, lon) in locations.items():
        print(f"Fetching weather for {name} ({lat}, {lon}) from {start_date} to {end_date}...")
        
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": start_date,
            "end_date": end_date,
            "hourly": ["temperature_2m", "wind_speed_10m", "wind_speed_100m", "shortwave_radiation"],
            "timezone": "UTC"
        }
        
        responses = openmeteo.weather_api("https://archive-api.open-meteo.com/v1/archive", params=params)
        response = responses[0]

        hourly = response.Hourly()
        df = pd.DataFrame({
            "datetime_utc": pd.date_range(
                start=pd.Timestamp(hourly.Time(), unit="s", tz="UTC"),
                end=pd.Timestamp(hourly.TimeEnd(), unit="s", tz="UTC"),
                freq=pd.Timedelta(seconds=hourly.Interval()),
                inclusive="left"
            ),
            "location": name,
            "temperature_2m": hourly.Variables(0).ValuesAsNumpy(),
            "wind_speed_10m": hourly.Variables(1).ValuesAsNumpy(),
            "wind_speed_100m": hourly.Variables(2).ValuesAsNumpy(),
            "shortwave_radiation": hourly.Variables(3).ValuesAsNumpy()
        })
        all_locations_data.append(df)

    final_df = pd.concat(all_locations_data, axis=0)
    return final_df.set_index(["datetime_utc", "location"]).sort_index()

if __name__ == "__main__":
    CITIES = {
        "Berlin": (52.52, 13.41),
        "Hamburg": (53.55, 9.99),
        "Munich": (48.13, 11.58),
        "Cologne": (50.93, 6.95),
        "Frankfurt": (50.11, 8.68)
    }
    weather_2024 = fetch_weather("2024-01-01", "2024-12-31", CITIES)
    if not weather_2024.empty:
        weather_2024.to_parquet("data/raw/weather_2024.parquet")
        print(f"Saved weather_2024.parquet with {len(weather_2024)} rows across {len(CITIES)} locations.")
