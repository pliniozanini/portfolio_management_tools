import pandas as pd
import numpy as np
import logging
import re

logger = logging.getLogger(__name__)


class ScenarioDefinitions:
    """
    Stores predefined historical stress scenarios with their start and end dates.
    Use register() to add custom scenarios at runtime.
    """

    _SCENARIOS = {
        "COVID_19": {"start_date": "2020-02-19", "end_date": "2020-04-01"},
        "JOESLEY_DAY": {"start_date": "2017-05-17", "end_date": "2017-05-19"},
        "SUBPRIME_CRASH": {"start_date": "2008-09-01", "end_date": "2008-11-30"},
    }

    @classmethod
    def get(cls, name: str) -> dict:
        """Return scenario dict for a given name. Raises KeyError if not found."""
        if name not in cls._SCENARIOS:
            raise KeyError(
                f"Scenario '{name}' not found. "
                f"Available: {list(cls._SCENARIOS.keys())}"
            )
        return cls._SCENARIOS[name]

    @classmethod
    def list(cls) -> list:
        """Return list of all registered scenario names."""
        return list(cls._SCENARIOS.keys())

    @classmethod
    def register(cls, name: str, start_date: str, end_date: str) -> None:
        """
        Register a custom stress scenario.

        Parameters
        ----------
        name : str
            Unique scenario identifier (e.g. 'CUSTOM_CRASH').
        start_date : str
            Start date in 'YYYY-MM-DD' format.
        end_date : str
            End date in 'YYYY-MM-DD' format.
        """
        if name in cls._SCENARIOS:
            logger.warning("Overwriting existing scenario '%s'", name)
        cls._SCENARIOS[name] = {"start_date": start_date, "end_date": end_date}
        logger.info("Registered scenario '%s' (%s → %s)", name, start_date, end_date)


# ---------------------------------------------------------------------------
# B3 underlying lookup table
# ---------------------------------------------------------------------------
# Maps the first 4 chars of an option ticker to the underlying stock ticker.
# Covers the most liquid B3 names that trade options.
_B3_UNDERLYING_MAP = {
    "PETR": "PETR4",
    "VALE": "VALE3",
    "BBAS": "BBAS3",
    "ITUB": "ITUB4",
    "BBDC": "BBDC4",
    "B3SA": "B3SA3",
    "MGLU": "MGLU3",
    "WEGE": "WEGE3",
    "RENT": "RENT3",
    "ABEV": "ABEV3",
    "SUZB": "SUZB3",
    "JBSS": "JBSS3",
    "HAPV": "HAPV3",
    "COGN": "COGN3",
    "CYRE": "CYRE3",
    "USIM": "USIM5",
    "CSNA": "CSNA3",
    "GGBR": "GGBR4",
    "BRKM": "BRKM5",
    "LREN": "LREN3",
    "CIEL": "CIEL3",
    "VIVT": "VIVT3",
    "EMBR": "EMBR3",
    "AZUL": "AZUL4",
    "GOLL": "GOLL4",
    "IRBR": "IRBR3",
    "CVCB": "CVCB3",
    "PRIO": "PRIO3",
    "BOVA": "BOVA11",
    "RADL": "RADL3",
    "TOTS": "TOTS3",
    "RAIL": "RAIL3",
    "KLBN": "KLBN11",
    "SANB": "SANB11",
    "TAEE": "TAEE11",
    "ENGI": "ENGI11",
    "CMIG": "CMIG4",
    "ELET": "ELET3",
    "NTCO": "NTCO3",
    "SBSP": "SBSP3",
}

# B3 call months: A=Jan … L=Dec  |  Put months: M=Jan … X=Dec
_CALL_MONTHS = set("ABCDEFGHIJKL")
_PUT_MONTHS = set("MNOPQRSTUVWX")


class PortfolioParser:
    """
    Parses a current portfolio snapshot to isolate Renda Variável positions
    and apply option-specific logic (calls vs puts).

    Expected input:
        A dictionary mapping ticker to weight (e.g. {'PETR4': 0.4})
        OR a DataFrame with 'ticker' and 'weight' columns.
    """

    @staticmethod
    def parse(portfolio):
        """
        Split portfolio into equities, calls, and puts.

        Returns
        -------
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
            (equities, calls, puts)  — puts include an 'underlying' column.
        """
        import re

        # Normalize input to DataFrame
        if isinstance(portfolio, dict):
            df = pd.DataFrame(list(portfolio.items()), columns=["ticker", "weight"])
        elif isinstance(portfolio, pd.DataFrame):
            if "ticker" in portfolio.columns and "weight" in portfolio.columns:
                df = portfolio.copy()
            elif len(portfolio.columns) == 1:
                df = portfolio.copy().reset_index()
                df.columns = ["ticker", "weight"]
            else:
                raise ValueError("DataFrame must have 'ticker' and 'weight' columns")
        else:
            raise TypeError("portfolio must be a dictionary or a DataFrame")

        # Automatically identify asset classes
        df["asset_class"] = df["ticker"].apply(PortfolioParser._identify_asset_class)

        # 1. Isolate equities
        equities = df[df["asset_class"] == "Equity"].copy()

        # 2. Isolate calls
        calls = df[df["asset_class"] == "Call"].copy()

        # 3. Isolate puts
        puts = df[df["asset_class"] == "Put"].copy()

        # 4. Map puts to their underlying asset
        if not puts.empty:
            puts["underlying"] = puts["ticker"].apply(
                PortfolioParser._extract_underlying
            )

        return equities, calls, puts

    @staticmethod
    def _identify_asset_class(ticker: str) -> str:
        """
        Determine if a B3 ticker is Equity, Call, or Put.
        """
        ticker = str(ticker).strip().upper()
        if ticker.startswith("^"):
            return "Equity"
        
        # B3 Options: 4 letters + 1 letter (A-X) + numbers
        import re
        if re.match(r"^[A-Z]{4}[A-X]\d+$", ticker):
            month_char = ticker[4]
            if month_char in _CALL_MONTHS:
                return "Call"
            elif month_char in _PUT_MONTHS:
                return "Put"
        
        return "Equity"

    @staticmethod
    def _extract_underlying(ticker: str) -> str:
        """
        Map an option ticker to its underlying stock ticker.

        Uses a hardcoded lookup table for the most common B3 option names.
        Falls back to first_4_chars + '4' if not found.
        """
        prefix = str(ticker).strip().upper()[:4]
        underlying = _B3_UNDERLYING_MAP.get(prefix, prefix + "4")
        return underlying


class HistoricalStressEngine:
    """
    Applies a historical stress scenario to a parsed portfolio and computes
    BRL-denominated P&L impact plus risk statistics.
    """

    def __init__(self):
        pass

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------
    def run_scenario(
        self,
        parsed_portfolio,
        scenario_name: str = "COVID_19",
        proxy_ticker: str = "^BVSP",
        total_portfolio_value: float = None,
        historical_prices: pd.DataFrame = None,
    ) -> dict:
        """
        Execute a stress scenario on the parsed portfolio.

        Parameters
        ----------
        parsed_portfolio : tuple
            (equities, calls, puts) as returned by PortfolioParser.parse().
        scenario_name : str
            Key in ScenarioDefinitions (built-in or custom-registered).
        proxy_ticker : str
            Fallback index ticker for assets without data in the period.
        total_portfolio_value : float, optional
            Total portfolio NAV in BRL. Used to compute impact as percentage.
            If None, impact_pct is omitted from the result.
        historical_prices : pd.DataFrame, optional
            User-supplied price matrix (index=dates, columns=tickers,
            values=adjusted close). When provided, yfinance is NOT called
            for historical data (current prices for puts may still be fetched).

        Returns
        -------
        dict
            scenario, total_impact_brl, total_impact_pct, details,
            summary_stats, max_drawdown_stats, max_rundown_stats
        """
        equities, calls, puts = parsed_portfolio
        dates = ScenarioDefinitions.get(scenario_name)

        # ---- Collect tickers needed ----
        equity_tickers = equities["ticker"].tolist() if not equities.empty else []
        put_underlyings = puts["underlying"].tolist() if not puts.empty else []
        all_raw_tickers = list(set(equity_tickers + put_underlyings))

        # ---- Fetch / receive historical prices ----
        if historical_prices is not None:
            logger.info("Using user-supplied historical prices")
            prices = historical_prices.copy()
            # Ensure proxy is available in user data (column name must match)
        else:
            yf_tickers = [self._to_yf_ticker(t) for t in all_raw_tickers]
            yf_proxy = proxy_ticker  # index tickers like ^BVSP don't need .SA
            yf_tickers_with_proxy = list(set(yf_tickers + [yf_proxy]))
            prices = self._fetch_prices(
                yf_tickers_with_proxy, dates["start_date"], dates["end_date"]
            )

        # ---- Scenario returns (first → last) ----
        if prices.empty:
            raise ValueError(
                "No price data available for the scenario window. "
                "Check tickers and date range."
            )
        scenario_returns = prices.iloc[-1] / prices.iloc[0] - 1

        # ---- Daily returns for risk stats ----
        daily_returns = prices.pct_change().dropna()

        # ---- Resolve proxy return ----
        proxy_col = self._resolve_proxy_col(proxy_ticker, prices.columns)
        proxy_return = scenario_returns[proxy_col] if proxy_col else 0.0

        total_impact_pct = 0.0
        impact_details = []
        weighted_daily_impacts = pd.Series(0.0, index=daily_returns.index)

        # ---- 1. Equities ----
        for _, row in equities.iterrows():
            ticker = row["ticker"]
            col = self._find_col(ticker, prices.columns)
            asset_ret = scenario_returns[col] if col else proxy_return
            weight = row["weight"]

            impact_pct = weight * asset_ret
            total_impact_pct += impact_pct
            
            impact_details.append(
                {"asset": ticker, "type": "Equity", "impact_pct": impact_pct}
            )
            # daily contribution
            daily_col = col if col else proxy_col
            if daily_col and daily_col in daily_returns.columns:
                weighted_daily_impacts += weight * daily_returns[daily_col]

        # ---- 2. Calls (100% loss of current value) ----
        for _, row in calls.iterrows():
            weight = row["weight"]
            # Long call (weight > 0): lose the premium (-weight)
            # Short call (weight < 0): keep the premium (+abs(weight))
            impact_pct = -abs(weight) if weight >= 0 else abs(weight)
            total_impact_pct += impact_pct
            impact_details.append(
                {"asset": row["ticker"], "type": "Call", "impact_pct": impact_pct}
            )

        # ---- 3. Puts (mapped to underlying) ----
        for _, row in puts.iterrows():
            put_ticker = row["ticker"]
            underlying = row["underlying"]
            col = self._find_col(underlying, prices.columns)
            asset_ret = scenario_returns[col] if col else proxy_return

            weight = row["weight"]
            # Need current prices to establish leverage (how much underlying exposure per $1 of put)
            put_price = self._get_current_price(put_ticker)
            underlying_price = self._get_current_price(underlying)
            
            if put_price > 0:
                leverage = underlying_price / put_price
            else:
                logger.warning("Put price for '%s' is 0 or unavailable. Assuming leverage of 1.0", put_ticker)
                leverage = 1.0

            # Long put (weight > 0) profits when underlying drops
            exposure_pct = weight * leverage
            impact_pct = exposure_pct * (-asset_ret)
            
            total_impact_pct += impact_pct
            impact_details.append(
                {"asset": put_ticker, "type": "Put", "impact_pct": impact_pct}
            )
            # daily contribution
            daily_col = col if col else proxy_col
            if daily_col and daily_col in daily_returns.columns:
                weighted_daily_impacts += exposure_pct * (-daily_returns[daily_col])

        # ---- Build result ----
        result = {
            "scenario": scenario_name,
            "total_impact_pct": total_impact_pct,
            "details": impact_details,
        }

        if total_portfolio_value is not None and total_portfolio_value != 0:
            result["total_impact_brl"] = total_impact_pct * total_portfolio_value
            for detail in impact_details:
                detail["impact_brl"] = detail["impact_pct"] * total_portfolio_value

        # ---- Risk stats on weighted daily impacts ----
        # The weighted impacts are already in portfolio percentage terms
        portfolio_daily_rets = weighted_daily_impacts

        result.update(self._compute_risk_stats(portfolio_daily_rets))

        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _to_yf_ticker(ticker: str) -> str:
        """Append .SA suffix for B3 tickers if not already present."""
        t = ticker.strip()
        if t.startswith("^") or t.upper().endswith(".SA"):
            return t
        return t + ".SA"

    @staticmethod
    def _resolve_proxy_col(proxy_ticker: str, columns) -> str | None:
        """Find the proxy ticker column in the DataFrame."""
        if proxy_ticker in columns:
            return proxy_ticker
        # yfinance may return with .SA
        alt = proxy_ticker + ".SA"
        if alt in columns:
            return alt
        return None

    @staticmethod
    def _find_col(ticker: str, columns) -> str | None:
        """Find a ticker in DataFrame columns, trying raw and .SA variants."""
        if ticker in columns:
            return ticker
        alt = ticker + ".SA"
        if alt in columns:
            return alt
        # Also try without .SA in case user-supplied data uses raw tickers
        if ticker.upper().endswith(".SA"):
            raw = ticker[:-3]
            if raw in columns:
                return raw
        return None

    @staticmethod
    def _fetch_prices(tickers: list, start: str, end: str) -> pd.DataFrame:
        """Fetch adjusted close prices via yfinance."""
        try:
            import yfinance as yf
        except ImportError:
            raise ImportError(
                "yfinance is required for automatic price fetching. "
                "Install it with: pip install yfinance"
            )

        logger.info(
            "Fetching prices for %d tickers from %s to %s", len(tickers), start, end
        )
        data = yf.download(tickers, start=start, end=end, auto_adjust=True)

        if data.empty:
            return pd.DataFrame()

        # yf.download returns MultiIndex columns when > 1 ticker
        if isinstance(data.columns, pd.MultiIndex):
            # Use 'Close' since auto_adjust=True makes Close = Adj Close
            prices = data["Close"]
        else:
            prices = data[["Close"]].rename(columns={"Close": tickers[0]})

        return prices

    @staticmethod
    def _get_current_price(ticker: str) -> float:
        """Get the current market price for a ticker via yfinance."""
        try:
            import yfinance as yf
        except ImportError:
            raise ImportError(
                "yfinance is required for current price lookup. "
                "Install it with: pip install yfinance"
            )

        yf_ticker = ticker if ticker.endswith(".SA") or ticker.startswith("^") else ticker + ".SA"
        try:
            info = yf.Ticker(yf_ticker).info
            price = info.get("currentPrice") or info.get("regularMarketPrice") or info.get("previousClose", 0)
            return float(price)
        except Exception as e:
            logger.warning("Could not fetch current price for '%s': %s. Using 0.", yf_ticker, e)
            return 0.0

    @staticmethod
    def _compute_risk_stats(returns: pd.Series) -> dict:
        """
        Compute summary_stats, max_drawdown_stats, and max_rundown_stats
        using functions from risk_metrics.
        """
        from . import risk_metrics as rm

        stats = {}

        if returns.empty or len(returns) < 2:
            logger.warning("Not enough data points for risk stats")
            stats["summary_stats"] = None
            stats["max_drawdown_stats"] = None
            stats["max_rundown_stats"] = None
            return stats

        # summary_stats expects a DataFrame
        returns_df = returns.to_frame(name="scenario")
        try:
            stats["summary_stats"] = rm.summary_stats(returns_df)
        except Exception as e:
            logger.warning("summary_stats failed: %s", e)
            stats["summary_stats"] = None

        try:
            dd = rm.drawdown(returns)
            stats["max_drawdown_stats"] = rm.calculate_max_drawdown_stats(dd)
        except Exception as e:
            logger.warning("max_drawdown_stats failed: %s", e)
            stats["max_drawdown_stats"] = None

        try:
            stats["max_rundown_stats"] = rm.calculate_max_rundown_stats(returns)
        except Exception as e:
            logger.warning("max_rundown_stats failed: %s", e)
            stats["max_rundown_stats"] = None

        return stats
