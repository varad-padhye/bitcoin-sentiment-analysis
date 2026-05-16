# bitcoin-sentiment-analysis
# Market Sentiment and Risk Regime Analysis

This analysis evaluates the impact of psychological market regimes—ranging from **Extreme Fear** to **Extreme Greed**—on trading performance and risk-to-reward dynamics within the **Hyperliquid** ecosystem. The data confirms that market sentiment is not merely a psychological indicator but a fundamental driver of volatility and profitability distributions.

---

## Quantitative Risk Metrics Overview

The following metrics represent the performance boundaries and efficiency across different emotional regimes.

| Sentiment Regime | Max Profit ($) | Max Loss ($) | Volatility | Avg Return ($) | Sharpe Proxy |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Extreme Fear** | 2,020.00 | -1,430.89 | 76.73 | 1.89 | 0.025 |
| **Neutral** | 2,979.55 | -1,032.98 | 142.95 | 27.09 | 0.190 |
| **Fear** | 71,535.72 | -19,841.24 | 1,342.35 | 128.29 | 0.096 |
| **Greed** | 34,903.82 | -117,990.10 | 1,399.47 | 53.99 | 0.039 |
| **Extreme Greed** | 44,223.45 | -18,360.67 | 1,861.56 | 205.82 | 0.111 |

---

## Comparative Sentiment Analysis

### 1. The Efficiency of Neutrality
The **Neutral** regime emerges as the most efficient environment for capital growth. Despite having lower absolute returns compared to greed-driven phases, its exceptionally low volatility (142.95) results in the highest **Sharpe Proxy (0.190)**. This suggests that "boring" markets allow for the most predictable and reliable strategy execution.

### 2. The "Greed Trap" and Liquidation Risk
A critical anomaly is observed in the **Greed** regime. While *Extreme Greed* provides high average returns (205.82) through powerful momentum, standard *Greed* acts as a high-risk trap. It features the worst maximum loss in the dataset (**-$117,990.10**), likely due to over-leveraged positions being caught in sudden market corrections or "leverage flushes".

### 3. Fear as a Divergent Catalyst
* **Fear:** Characterized by high volatility and significant profit potential (Avg Return: 128.29). This regime favors "mean reversion" traders who capitalize on sharp oversold rallies.
* **Extreme Fear:** Represents a period of market paralysis. Both volatility and profit potential collapse, indicating a lack of liquidity and trading volume as participants exit the market.

---

## Statistical Validation: Mann-Whitney U Test

The Mann-Whitney U Test provides the mathematical foundation for these observations, comparing the distribution of returns between **Fear** and **Greed**.

* **U-Statistic:** 122,776,380.00
* **P-Value:** 0.000000
* **Verdict:** Statistical Significance Proven

> **Conclusion:** The P-value of zero indicates that the differences in trader performance across these regimes are statistically significant and not the result of random chance. This confirms that market psychology fundamentally alters the "rules of engagement," necessitating adaptive trading strategies.

---

## Strategic Implications

* **Risk Management:** Exposure should be significantly reduced during **Greed** regimes to avoid catastrophic liquidation tails.
* **Strategy Selection:** Momentum-based strategies excel in **Extreme Greed**, while mean-reversion strategies should be prioritized during standard **Fear**.
* **Predictive Modeling:** Sentiment data must be integrated as a primary feature in any machine learning or algorithmic execution model, as it holds high predictive power for market regime shifts.
