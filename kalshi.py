import csv
import requests

BASE_URL = "https://external-api.kalshi.com/trade-api/v2"
SERIES_TICKER = "KXHIGHNY"
OUTPUT_FILE = "kalshi_training_data.csv"


def safe_float(market, key, default=0.0):
    value = market.get(key, default)

    if value is None:
        return default

    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def normalize_volume(value):
    return min(value / 100000.0, 1.0)


def get_label(market):
    result = market.get("result")

    if isinstance(result, str):
        result = result.lower()

    if result == "yes":
        return 1.0

    if result == "no":
        return 0.0

    return None


def market_to_features(market):
    yes_bid = safe_float(market, "yes_bid_dollars")
    yes_ask = safe_float(market, "yes_ask_dollars")
    previous_yes_bid = safe_float(market, "previous_yes_bid_dollars")
    previous_yes_ask = safe_float(market, "previous_yes_ask_dollars")
    last_price = safe_float(market, "last_price_dollars")
    previous_price = safe_float(market, "previous_price_dollars")

    volume = normalize_volume(safe_float(market, "volume_fp"))
    volume_24h = normalize_volume(safe_float(market, "volume_24h_fp"))
    open_interest = normalize_volume(safe_float(market, "open_interest_fp"))
    liquidity = safe_float(market, "liquidity_dollars")

    mid_price = (yes_bid + yes_ask) / 2.0
    spread = yes_ask - yes_bid

    previous_mid_price = (previous_yes_bid + previous_yes_ask) / 2.0
    price_change = last_price - previous_price

    return [
        yes_bid,
        yes_ask,
        mid_price,
        spread,
        previous_mid_price,
        last_price,
        previous_price,
        price_change,
        volume,
        volume_24h,
        open_interest,
        liquidity,
    ]


def main():
    markets_url = (
        f"{BASE_URL}/markets"
        f"?series_ticker={SERIES_TICKER}"
        f"&status=settled"
    )

    response = requests.get(markets_url)
    response.raise_for_status()

    markets_data = response.json()
    markets = markets_data.get("markets", [])

    print(f"Downloaded {len(markets)} markets.")

    rows_written = 0
    skipped = 0

    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)

        writer.writerow([
            "yes_bid",
            "yes_ask",
            "mid_price",
            "spread",
            "previous_mid_price",
            "last_price",
            "previous_price",
            "price_change",
            "volume",
            "volume_24h",
            "open_interest",
            "liquidity",
            "label",
        ])

        for market in markets:
            label = get_label(market)

            if label is None:
                skipped += 1
                print(
                    "Skipping:",
                    market.get("ticker"),
                    "status=",
                    market.get("status"),
                    "result=",
                    market.get("result"),
                )
                continue

            features = market_to_features(market)
            writer.writerow(features + [label])
            rows_written += 1

    print(f"Wrote {rows_written} training rows to {OUTPUT_FILE}")
    print(f"Skipped {skipped} markets.")


if __name__ == "__main__":
    main()