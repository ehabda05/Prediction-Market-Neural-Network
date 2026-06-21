import requests

BASE_URL = "https://external-api.kalshi.com/trade-api/v2"
SERIES_TICKER = "KXHIGHNY"
OUTPUT_FILE = "kalshi_output.txt"

# Get series information
url = f"{BASE_URL}/series/{SERIES_TICKER}"
response = requests.get(url)
series_data = response.json()

# Get all open markets for the KXHIGHNY series
markets_url = f"{BASE_URL}/markets?series_ticker={SERIES_TICKER}&status=open"
markets_response = requests.get(markets_url)
markets_data = markets_response.json()

with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
    # Series info
    file.write(f"Series Title: {series_data['series']['title']}\n")
    file.write(f"Frequency: {series_data['series']['frequency']}\n")
    file.write(f"Category: {series_data['series']['category']}\n")

    print(f"Series Title: {series_data['series']['title']}")
    print(f"Frequency: {series_data['series']['frequency']}")
    print(f"Category: {series_data['series']['category']}")

    # Market info
    file.write(f"\nActive markets in {SERIES_TICKER} series:\n")
    print(f"\nActive markets in {SERIES_TICKER} series:")

    for market in markets_data["markets"]:
        market_text = (
            f"- {market['ticker']}: {market['title']}\n"
            f"  Event: {market['event_ticker']}\n"
            f"  Yes Price: ${market['yes_bid_dollars']} | Volume: {market['volume_fp']}\n\n"
        )

        file.write(market_text)
        print(market_text, end="")

    # Event details
    if markets_data["markets"]:
        event_ticker = markets_data["markets"][0]["event_ticker"]
        event_url = f"{BASE_URL}/events/{event_ticker}"

        event_response = requests.get(event_url)
        event_data = event_response.json()

        event_text = (
            "Event Details:\n"
            f"Title: {event_data['event']['title']}\n"
            f"Category: {event_data['event']['category']}\n"
        )

        file.write(event_text)
        print(event_text)
    else:
        file.write("No open markets found.\n")
        print("No open markets found.")

print(f"\nData written to {OUTPUT_FILE}")