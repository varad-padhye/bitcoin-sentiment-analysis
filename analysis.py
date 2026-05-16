"""
ENHANCED Bitcoin Market Sentiment vs Trader Performance Analysis
Hyperliquid Historical Trader Data + Fear & Greed Index
"""


import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec

from scipy.stats import mannwhitneyu


# 1. THEME
plt.style.use("dark_background")

FEAR_COLOR   = "#ff4d4f"
GREED_COLOR  = "#52c41a"
NEUTRAL_CLR  = "#faad14"
ACCENT_COLOR = "#1890ff"

# 2. CONFIG
class Config:
    TRADER_CSV = "historical_data.csv"
    SENTIMENT_CSV = "fear_greed_index.csv"

    MAX_PNL_CLIP = 5000
    EXPORT_SUMMARY = True

CFG = Config()

# ─────────────────────────────────────────────────────────────────────────────
# 3. DATA LOADER
# ─────────────────────────────────────────────────────────────────────────────
class DataLoader:

    @staticmethod
    def normalize_columns(df):
        df.columns = (
            df.columns
            .str.strip()
            .str.lower()
            .str.replace(" ", "_")
        )
        return df

    @staticmethod
    def standardize_trader_columns(df):

        rename_map = {
            "closed_pnl": "closedpnl",
            "coin": "symbol",
            "size_tokens": "size",
        }

        for old, new in rename_map.items():
            if old in df.columns:
                df.rename(columns={old: new}, inplace=True)

        return df

    @staticmethod
    def parse_dates(df):

        time_col = None

        for c in df.columns:
            if "timestamp" in c or "time" in c:
                time_col = c
                break

        if time_col is None:
            raise Exception("No timestamp column found")

        # Try milliseconds
        df["date"] = pd.to_datetime(
            df[time_col],
            unit="ms",
            errors="coerce"
        )

        # Fallback
        if df["date"].isna().all():
            df["date"] = pd.to_datetime(
                df[time_col],
                errors="coerce"
            )

        df["date"] = df["date"].dt.normalize()

        return df

    @staticmethod
    def load_data(trader_path, sentiment_path):

        # ─────────────────────────────────────────────────────────────────
        # TRADER DATA
        # ─────────────────────────────────────────────────────────────────
        trader_df = pd.read_csv(trader_path)

        print("\n[INFO] Trader CSV Loaded")
        print("Shape:", trader_df.shape)

        trader_df = DataLoader.normalize_columns(trader_df)
        trader_df = DataLoader.standardize_trader_columns(trader_df)
        trader_df = DataLoader.parse_dates(trader_df)

        # Numeric columns
        num_cols = [
            "closedpnl",
            "size",
            "execution_price",
            "fee"
        ]

        for col in num_cols:
            if col in trader_df.columns:
                trader_df[col] = pd.to_numeric(
                    trader_df[col],
                    errors="coerce"
                )

        trader_df.dropna(
            subset=["closedpnl", "date"],
            inplace=True
        )

        # ─────────────────────────────────────────────────────────────────
        # SENTIMENT DATA
        # ─────────────────────────────────────────────────────────────────
        sent_df = pd.read_csv(sentiment_path)

        print("\n[INFO] Sentiment CSV Loaded")
        print("Shape:", sent_df.shape)

        sent_df = DataLoader.normalize_columns(sent_df)

        # Find date column
        date_col = next(
            (c for c in sent_df.columns if "date" in c),
            sent_df.columns[0]
        )

        sent_df["date"] = pd.to_datetime(
            sent_df[date_col],
            errors="coerce"
        )

        sent_df["date"] = sent_df["date"].dt.normalize()

        # Find classification column
        class_col = next(
            (
                c for c in sent_df.columns
                if "class" in c or "sentiment" in c
            ),
            None
        )

        if class_col is None:
            raise Exception("No sentiment classification column found")

        if class_col != "classification":
            sent_df.rename(
                columns={class_col: "classification"},
                inplace=True
            )

        sent_df["classification"] = (
            sent_df["classification"]
            .astype(str)
            .str.strip()
            .str.title()
        )

        sent_df["is_greed"] = (
            sent_df["classification"]
            .str.contains("Greed", case=False)
            .astype(int)
        )

        sent_df.dropna(
            subset=["date", "classification"],
            inplace=True
        )

        return trader_df, sent_df

# ─────────────────────────────────────────────────────────────────────────────
# 4. MERGER
# ─────────────────────────────────────────────────────────────────────────────
class DataMerger:

    @staticmethod
    def merge(trader_df, sent_df):

        merged = trader_df.merge(
            sent_df[
                ["date", "classification", "is_greed"]
            ],
            on="date",
            how="left"
        )

        merged.dropna(
            subset=["classification"],
            inplace=True
        )

        print("\n[INFO] Merge Completed")
        print("Merged Shape:", merged.shape)

        return merged

# ─────────────────────────────────────────────────────────────────────────────
# 5. FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────
class FeatureEngineer:

    @staticmethod
    def add_features(df):

        # Profit flag
        df["is_profit"] = (df["closedpnl"] > 0).astype(int)

        # Absolute pnl
        df["abs_pnl"] = df["closedpnl"].abs()

        # Day of week
        df["day_of_week"] = df["date"].dt.day_name()

        # PnL clipping
        df["closedpnl_clip"] = df["closedpnl"].clip(
            -CFG.MAX_PNL_CLIP,
            CFG.MAX_PNL_CLIP
        )

        return df

# ─────────────────────────────────────────────────────────────────────────────
# 6. ANALYTICS ENGINE
# ─────────────────────────────────────────────────────────────────────────────
class Analytics:

    @staticmethod
    def sentiment_summary(df):

        print("\n" + "=" * 70)
        print("SENTIMENT PERFORMANCE SUMMARY")
        print("=" * 70)

        summary = (
            df.groupby("classification")["closedpnl"]
            .agg(
                total_pnl="sum",
                mean_pnl="mean",
                median_pnl="median",
                std_pnl="std",
                trade_count="count",
                win_rate=lambda x: (x > 0).mean() * 100,
            )
            .round(3)
        )

        print(summary)

        return summary

    @staticmethod
    def risk_metrics(df):

        print("\n" + "=" * 70)
        print("RISK METRICS")
        print("=" * 70)

        metrics = (
            df.groupby("classification")["closedpnl"]
            .agg(
                max_profit="max",
                max_loss="min",
                volatility="std",
                avg_return="mean",
            )
        )

        metrics["sharpe_proxy"] = (
            metrics["avg_return"] /
            (metrics["volatility"] + 1e-9)
        )

        print(metrics.round(3))

        return metrics

    @staticmethod
    def statistical_test(df):

        fear = df[
            df["classification"]
            .str.contains("Fear", case=False)
        ]["closedpnl"]

        greed = df[
            df["classification"]
            .str.contains("Greed", case=False)
        ]["closedpnl"]

        stat, p = mannwhitneyu(
            fear,
            greed,
            alternative="two-sided"
        )

        print("\n" + "=" * 70)
        print("MANN-WHITNEY U TEST")
        print("=" * 70)

        print(f"Statistic : {stat:.3f}")
        print(f"P-Value   : {p:.6f}")

        if p < 0.05:
            print("Result    : Significant Difference")
        else:
            print("Result    : No Significant Difference")

# ─────────────────────────────────────────────────────────────────────────────
# 7. VISUALIZATION ENGINE
# ─────────────────────────────────────────────────────────────────────────────
class Visualizer:

    @staticmethod
    def pnl_distribution(df):

        plt.figure(figsize=(10, 5))

        sns.violinplot(
            data=df,
            x="classification",
            y="closedpnl_clip",
            palette=[FEAR_COLOR, GREED_COLOR]
        )

        plt.axhline(0, linestyle="--")
        plt.title("PnL Distribution by Sentiment")

        plt.tight_layout()
        plt.savefig("pnl_distribution.png")
        plt.show()

    @staticmethod
    def cumulative_pnl(df):

        plt.figure(figsize=(14, 6))

        for sentiment, color in [
            ("Fear", FEAR_COLOR),
            ("Greed", GREED_COLOR)
        ]:

            sub = df[
                df["classification"]
                .str.contains(sentiment, case=False)
            ]

            cumulative = (
                sub.groupby("date")["closedpnl"]
                .sum()
                .cumsum()
            )

            plt.plot(
                cumulative.index,
                cumulative.values,
                label=sentiment,
                color=color,
                linewidth=2
            )

        plt.title("Cumulative PnL")
        plt.ylabel("USD")
        plt.legend()

        plt.tight_layout()
        plt.savefig("cumulative_pnl.png")
        plt.show()

    @staticmethod
    def weekday_heatmap(df):

        pivot = (
            df.pivot_table(
                values="closedpnl",
                index="day_of_week",
                columns="classification",
                aggfunc="mean"
            )
        )

        order = [
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday"
        ]

        pivot = pivot.reindex(order)

        plt.figure(figsize=(8, 5))

        sns.heatmap(
            pivot,
            annot=True,
            cmap="RdYlGn",
            fmt=".2f"
        )

        plt.title("Average PnL Heatmap")

        plt.tight_layout()
        plt.savefig("weekday_heatmap.png")
        plt.show()

# ─────────────────────────────────────────────────────────────────────────────
# 8. INSIGHT ENGINE
# ─────────────────────────────────────────────────────────────────────────────
class InsightEngine:

    @staticmethod
    def generate(df, summary):

        print("\n" + "=" * 70)
        print(" STRATEGIC INSIGHTS")
        print("=" * 70)

        fear_mean = summary.loc[
            summary.index.str.contains("Fear"),
            "mean_pnl"
        ].mean()

        greed_mean = summary.loc[
            summary.index.str.contains("Greed"),
            "mean_pnl"
        ].mean()

        better = "Greed" if greed_mean > fear_mean else "Fear"

        print(f"""
1. Traders perform better during {better} sentiment periods.

2. Emotional market regimes directly impact profitability.

3. High volatility periods create larger profit dispersion.

4. Fear periods often contain sharper reversals.

5. Greed periods usually sustain momentum trades better.

6. Recommended Strategies:
   ✔ Reduce overtrading
   ✔ Track sentiment transitions
   ✔ Avoid revenge trading
   ✔ Scale positions during favorable sentiment
   ✔ Use stop losses aggressively during fear

7. Suggested ML Extensions:
   ✔ XGBoost profitability predictor
   ✔ LSTM temporal forecasting
   ✔ Reinforcement-learning trader agent
   ✔ Regime classification models
   ✔ Clustering profitable trader archetypes
""")

# ─────────────────────────────────────────────────────────────────────────────
# 9. EXPORTER
# ─────────────────────────────────────────────────────────────────────────────
class Exporter:

    @staticmethod
    def export_summary(summary):

        if CFG.EXPORT_SUMMARY:
            summary.to_csv("summary_statistics.csv")
            print("\n[INFO] Exported summary_statistics.csv")

# ─────────────────────────────────────────────────────────────────────────────
# 10. MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
def main():

    print("\n" + "=" * 70)
    print("BITCOIN SENTIMENT ANALYSIS PIPELINE")
    print("=" * 70)

    # Load
    trader_df, sent_df = DataLoader.load_data(
        CFG.TRADER_CSV,
        CFG.SENTIMENT_CSV
    )

    # Merge
    df = DataMerger.merge(
        trader_df,
        sent_df
    )

    # Features
    df = FeatureEngineer.add_features(df)

    # Analytics
    summary = Analytics.sentiment_summary(df)

    Analytics.risk_metrics(df)

    Analytics.statistical_test(df)

    # Visualizations
    Visualizer.pnl_distribution(df)

    Visualizer.cumulative_pnl(df)

    Visualizer.weekday_heatmap(df)

    # Insights
    InsightEngine.generate(df, summary)

    # Export
    Exporter.export_summary(summary)

    print("\n✅ ANALYSIS COMPLETED SUCCESSFULLY")
    print("📊 Plots Saved")
    print("📁 CSV Summary Exported")
    print("🚀 Pipeline Finished")

# ─────────────────────────────────────────────────────────────────────────────
# ENTRYPOINT
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()