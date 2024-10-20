# phoenix_v1
#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
# coding: utf-8

#!/usr/bin/env python
# coding: utf-8

# portfolio_analysis_app.py

# portfolio_analysis_app.py

import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import datetime

# Configure the page
st.set_page_config(layout="wide")

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

# Function to fetch historical data from Yahoo Finance
def fetch_yfinance_ohlcv(symbol, start_date, end_date=None):
    try:
        df = yf.download(symbol, start=start_date, end=end_date)
        if df.empty:
            st.error(f"No data found for ticker {symbol}.")
            return None
        adj_close = df['Adj Close']
        adj_close.index = pd.to_datetime(adj_close.index)
        adj_close.name = symbol  # Set the name of the Series to the ticker symbol
        return adj_close
    except Exception as e:
        st.error(f"Error fetching data for {symbol}: {e}")
        return None

# Function to compute cumulative returns with rebalancing
def compute_rebalanced_portfolio(returns, weights, rebalancing_dates):
    portfolio_values = pd.Series(index=returns.index)
    portfolio_values.iloc[0] = 1  # Start with initial portfolio value of 1
    current_weights = pd.Series(weights, index=returns.columns)

    for i in range(1, len(returns)):
        date = returns.index[i]
        daily_returns = returns.iloc[i]

        # Compute portfolio return
        portfolio_return = (current_weights * daily_returns).sum()

        # Update portfolio value
        portfolio_values.iloc[i] = portfolio_values.iloc[i - 1] * (1 + portfolio_return)

        # Update weights based on price movement
        current_weights = current_weights * (1 + daily_returns)
        # Normalize weights
        current_weights = current_weights / current_weights.sum()

        # If it's a rebalancing date, reset weights
        if date in rebalancing_dates:
            current_weights = pd.Series(weights, index=returns.columns)

    return portfolio_values

# Function to perform Monte Carlo simulation with multiple cash flows
def monte_carlo_simulation(returns_series, num_simulations, time_horizon_days, initial_investment, cash_flows):
    last_price = initial_investment  # Starting from the initial investment amount
    results = np.zeros((time_horizon_days, num_simulations))

    # Convert daily returns to log returns
    log_returns = np.log(1 + returns_series)

    # Calculate mean and standard deviation of log returns
    daily_mean = log_returns.mean()
    daily_std = log_returns.std()

    # Prepare cash flow schedule
    cash_flow_schedule = np.zeros(time_horizon_days)
    for cf in cash_flows:
        amount = cf['amount']
        frequency = cf['frequency']
        start_day = cf['start_day']
        end_day = cf['end_day']
        if frequency == 'Daily':
            interval = 1
        elif frequency == 'Weekly':
            interval = 5  # Approximate number of trading days in a week
        elif frequency == 'Monthly':
            interval = 21  # Approximate number of trading days in a month
        elif frequency == 'Quarterly':
            interval = 63  # Approximate number of trading days in a quarter
        elif frequency == 'Annually':
            interval = 252  # Number of trading days in a year

        for day in range(start_day, min(end_day, time_horizon_days), interval):
            if day < time_horizon_days:
                cash_flow_schedule[day] += amount

    for sim in range(num_simulations):
        prices = [last_price]
        for day in range(time_horizon_days):
            # Simulate daily return using GBM
            drift = daily_mean - 0.5 * daily_std ** 2
            shock = daily_std * np.random.normal()
            # Update price using the GBM formula
            new_price = prices[-1] * np.exp(drift + shock)

            # Apply cash flow if scheduled
            new_price += cash_flow_schedule[day]

            # Ensure portfolio value doesn't go negative
            new_price = max(new_price, 0)

            prices.append(new_price)
        results[:, sim] = prices[1:]  # Exclude the initial price

    return results

# Define the bear market periods
bear_market_periods = {
    'Dot-com Bubble Burst (2000-03-24 to 2002-10-09)': {'start_date': '2000-03-24', 'end_date': '2002-10-09'},
    'Global Financial Crisis (2007-10-09 to 2009-03-09)': {'start_date': '2007-10-09', 'end_date': '2009-03-09'},
    'COVID-19 Crash (2020-02-19 to 2020-03-23)': {'start_date': '2020-02-19', 'end_date': '2020-03-23'},
    '2022 Market Downturn (2022-01-03 to 2022-10-12)': {'start_date': '2022-01-03', 'end_date': '2022-10-12'}
}

# Streamlit app starts here
st.title("Portfolio Analysis with Custom Backtesting Periods and Monte Carlo Simulation")

# Initialize session state variables
if 'backtesting_run' not in st.session_state:
    st.session_state.backtesting_run = False
if 'simulation_run' not in st.session_state:
    st.session_state.simulation_run = False

# Sidebar inputs
st.sidebar.header("Portfolio Configuration")

# Input tickers
tickers_input = st.sidebar.text_input("Enter tickers separated by commas (e.g., AAPL,MSFT,GOOGL)", value="AAPL,MSFT,GOOGL")
tickers = [ticker.strip().upper() for ticker in tickers_input.split(',') if ticker.strip()]

if not tickers:
    st.error("Please enter at least one ticker.")
    st.stop()

# Input weights
weights = []
total_weight = 0

st.sidebar.write("Enter percentage allocation for each ticker:")
for ticker in tickers:
    weight = st.sidebar.number_input(f"Allocation for {ticker} (%)", min_value=0.0, max_value=100.0, value=0.0, step=0.1)
    weights.append(weight / 100)

total_weight = sum(weights)
if abs(total_weight - 1.0) > 1e-6:
    st.error(f"The total allocation is {total_weight*100:.2f}%, which does not equal 100%. Please adjust the allocations.")
    st.stop()

weights_series = pd.Series(weights, index=tickers)

# Backtesting period selection (Custom Period shown first)
st.sidebar.header("Backtesting Period")
period_option = st.sidebar.selectbox("Choose Backtesting Period:", ["Custom Period", "Predefined Bear Market Periods"])

if period_option == "Custom Period":
    start_date = st.sidebar.date_input("Start Date", value=datetime.date(2000, 1, 1))
    end_date = st.sidebar.date_input("End Date", value=datetime.date.today())
    if start_date >= end_date:
        st.error("Start date must be before end date.")
        st.stop()
    selected_periods = [("Custom Period", {'start_date': start_date.strftime("%Y-%m-%d"), 'end_date': end_date.strftime("%Y-%m-%d")})]
else:
    # Select bear market periods
    st.sidebar.header("Bear Market Periods")
    selected_periods_names = st.sidebar.multiselect(
        "Select bear market periods to analyze:",
        list(bear_market_periods.keys()),
        default=list(bear_market_periods.keys())
    )
    if not selected_periods_names:
        st.error("Please select at least one bear market period.")
        st.stop()
    selected_periods = [(name, bear_market_periods[name]) for name in selected_periods_names]

# Add a button to run the backtesting
if st.button("Run Backtesting"):
    st.session_state.backtesting_run = True

if st.session_state.backtesting_run:
    # Main content
    for period_name, period in selected_periods:
        st.header(f"Analyzing Period: {period_name}")

        # Fetch data for the tickers
        portfolios = {}
        missing_tickers = []
        for ticker in tickers:
            series = fetch_yfinance_ohlcv(ticker, period['start_date'], period['end_date'])
            if series is not None:
                portfolios[ticker] = series
            else:
                missing_tickers.append(ticker)

        if missing_tickers:
            st.warning(f"The following tickers are missing data in this period: {', '.join(missing_tickers)}")
            st.info("Portfolio data does not exist for this period due to missing data.")
            continue  # Skip this period

        # Merge all portfolios on common dates
        merged_data = pd.DataFrame(portfolios).dropna()

        # Ensure we have enough data
        if merged_data.empty:
            st.info("No overlapping data for the selected tickers in this period.")
            st.info("Portfolio data does not exist for this period.")
            continue

        # Calculate daily returns for each ticker
        returns = merged_data.pct_change().fillna(0)

        # Define rebalancing dates
        monthly_rebalance_dates = returns.resample('M').last().index
        yearly_rebalance_dates = returns.resample('Y').last().index

        # Compute portfolios
        non_rebalanced_portfolio = compute_rebalanced_portfolio(returns, weights_series.values, [])
        monthly_portfolio = compute_rebalanced_portfolio(returns, weights_series.values, monthly_rebalance_dates)
        yearly_portfolio = compute_rebalanced_portfolio(returns, weights_series.values, yearly_rebalance_dates)

        # Calculate cumulative returns
        non_rebalanced_return = non_rebalanced_portfolio.iloc[-1] / non_rebalanced_portfolio.iloc[0] - 1
        monthly_rebalanced_return = monthly_portfolio.iloc[-1] / monthly_portfolio.iloc[0] - 1
        yearly_rebalanced_return = yearly_portfolio.iloc[-1] / yearly_portfolio.iloc[0] - 1

        # Display the returns
        st.subheader("Cumulative Returns:")
        col1, col2, col3 = st.columns(3)
        col1.metric("Non-Rebalanced Portfolio", f"{non_rebalanced_return * 100:.2f}%")
        col2.metric("Monthly Rebalanced Portfolio", f"{monthly_rebalanced_return * 100:.2f}%")
        col3.metric("Yearly Rebalanced Portfolio", f"{yearly_rebalanced_return * 100:.2f}%")

        # Plot the cumulative returns for each portfolio
        st.subheader("Cumulative Returns Chart:")
        plt.figure(figsize=(14, 8))

        # Plot individual asset returns
        for ticker in tickers:
            plt.plot(merged_data.index, merged_data[ticker] / merged_data[ticker].iloc[0] - 1, label=f'{ticker} Return')

        # Plot the non-rebalanced (buy-and-hold) portfolio return
        plt.plot(non_rebalanced_portfolio.index, non_rebalanced_portfolio / non_rebalanced_portfolio.iloc[0] - 1,
                 label='Weighted Portfolio Return', linewidth=2, linestyle='--', color='black')

        # Plot the monthly rebalanced portfolio
        plt.plot(monthly_portfolio.index, monthly_portfolio / monthly_portfolio.iloc[0] - 1,
                 label='Monthly Rebalanced Portfolio', linewidth=2, linestyle='--', color='blue')

        # Plot the yearly rebalanced portfolio
        plt.plot(yearly_portfolio.index, yearly_portfolio / yearly_portfolio.iloc[0] - 1,
                 label='Yearly Rebalanced Portfolio', linewidth=2, linestyle='--', color='red')

        plt.title(f'Cumulative Returns during {period_name}')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Return')
        plt.legend()
        plt.grid()
        st.pyplot(plt.gcf())
        plt.clf()

        # Check if the period is a custom period or bear market period
        if period_option == "Custom Period":
            # Additional analyses
            st.subheader("Additional Analyses")

            # Create DataFrame for cumulative returns (Wealth Index)
            cumulative_returns = pd.DataFrame(index=merged_data.index)
            for ticker in tickers:
                cumulative_returns[ticker] = merged_data[ticker] / merged_data[ticker].iloc[0]

            cumulative_returns['Weighted Portfolio Return'] = non_rebalanced_portfolio / non_rebalanced_portfolio.iloc[0]
            cumulative_returns['Monthly Rebalanced Portfolio'] = monthly_portfolio / monthly_portfolio.iloc[0]
            cumulative_returns['Yearly Rebalanced Portfolio'] = yearly_portfolio / yearly_portfolio.iloc[0]

            # Create DataFrame for daily returns
            daily_returns = returns.copy()
            daily_returns['Weighted Portfolio Return'] = non_rebalanced_portfolio.pct_change().fillna(0)
            daily_returns['Monthly Rebalanced Portfolio'] = monthly_portfolio.pct_change().fillna(0)
            daily_returns['Yearly Rebalanced Portfolio'] = yearly_portfolio.pct_change().fillna(0)

            # Compute monthly and annual returns
            monthly_returns = daily_returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
            annual_returns = daily_returns.resample('Y').apply(lambda x: (1 + x).prod() - 1)

            # Performance Metrics
            st.subheader("Performance Metrics")

            # List of portfolios
            columns = tickers + ['Weighted Portfolio Return', 'Monthly Rebalanced Portfolio', 'Yearly Rebalanced Portfolio']

            # Compute total number of years
            start_date = daily_returns.index[0]
            end_date = daily_returns.index[-1]
            total_days = (end_date - start_date).days
            total_years = total_days / 365.25  # Using 365.25 to account for leap years

            # Initialize DataFrame to store metrics
            metrics = pd.DataFrame(index=['CAGR', 'Annualized Sharpe Ratio', 'Annualized Sortino Ratio',
                                          'Maximum Drawdown', 'Mean Monthly Return', 'Median Monthly Return',
                                          'Std Monthly Return', 'Annualized Std Dev'], columns=columns)

            # Assuming Risk-Free Rate is zero
            risk_free_rate = 0

            # Compute drawdowns
            wealth_index = cumulative_returns
            previous_peaks = wealth_index.cummax()
            drawdowns = (wealth_index - previous_peaks) / previous_peaks

            for col in columns:
                # CAGR
                ending_value = cumulative_returns[col].iloc[-1]
                beginning_value = cumulative_returns[col].iloc[0]
                cagr = (ending_value / beginning_value) ** (1 / total_years) - 1
                metrics.loc['CAGR', col] = cagr

                # Daily returns
                mean_daily_return = daily_returns[col].mean()
                std_daily_return = daily_returns[col].std()
                annualized_return = ((1 + mean_daily_return) ** 252) - 1  # Annualized return
                annualized_std = std_daily_return * np.sqrt(252)

                # Sharpe Ratio
                sharpe_ratio = (annualized_return - risk_free_rate) / annualized_std
                metrics.loc['Annualized Sharpe Ratio', col] = sharpe_ratio

                # Sortino Ratio
                downside_returns = daily_returns[col][daily_returns[col] < 0]
                downside_std = downside_returns.std() * np.sqrt(252)
                sortino_ratio = (annualized_return - risk_free_rate) / downside_std if downside_std != 0 else np.nan
                metrics.loc['Annualized Sortino Ratio', col] = sortino_ratio

                # Maximum Drawdown
                drawdown_series = drawdowns[col]
                mdd = drawdown_series.min()
                metrics.loc['Maximum Drawdown', col] = mdd

                # Monthly returns
                mean_monthly_return = monthly_returns[col].mean()
                median_monthly_return = monthly_returns[col].median()
                std_monthly_return = monthly_returns[col].std()
                metrics.loc['Mean Monthly Return', col] = mean_monthly_return
                metrics.loc['Median Monthly Return', col] = median_monthly_return
                metrics.loc['Std Monthly Return', col] = std_monthly_return

                # Annualized Std Dev (from daily returns)
                metrics.loc['Annualized Std Dev', col] = annualized_std

            st.dataframe(metrics.style.format("{:.2%}"))

            # Correlation Matrices
            st.subheader("Correlation Matrices")

            # Correlation matrix of daily returns
            daily_corr = daily_returns[columns].corr()
            st.write("Correlation Matrix of Daily Returns:")
            st.dataframe(daily_corr.style.background_gradient(cmap='coolwarm'))

            # Correlation matrix of monthly returns
            monthly_corr = monthly_returns[columns].corr()
            st.write("Correlation Matrix of Monthly Returns:")
            st.dataframe(monthly_corr.style.background_gradient(cmap='coolwarm'))

            # Histograms of Monthly Returns
            st.subheader("Histograms of Monthly Returns")
            for column in monthly_returns.columns:
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.hist(monthly_returns[column], bins=20, edgecolor='black')
                ax.set_title(f'Histogram of Monthly Returns - {column}')
                ax.set_xlabel('Monthly Return')
                ax.set_ylabel('Frequency')
                ax.grid(True)
                st.pyplot(fig)
                plt.clf()

            # Histograms of Annual Returns
            st.subheader("Histograms of Annual Returns")
            for column in annual_returns.columns:
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.hist(annual_returns[column], bins=10, edgecolor='black')
                ax.set_title(f'Histogram of Annual Returns - {column}')
                ax.set_xlabel('Annual Return')
                ax.set_ylabel('Frequency')
                ax.grid(True)
                st.pyplot(fig)
                plt.clf()

            # Drawdown Charts
            st.subheader("Drawdown Charts")
            for column in drawdowns.columns:
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.plot(drawdowns[column], label=f'Drawdown - {column}')
                ax.set_title(f'Drawdown Chart - {column}')
                ax.set_xlabel('Date')
                ax.set_ylabel('Drawdown')
                ax.legend()
                ax.grid(True)
                st.pyplot(fig)
                plt.clf()

            # Rolling Returns
            st.subheader("Rolling Returns")
            rolling_windows = [12, 36, 60]  # in months
            for window in rolling_windows:
                rolling_returns = (1 + monthly_returns).rolling(window).apply(np.prod, raw=True) - 1
                for column in rolling_returns.columns:
                    fig, ax = plt.subplots(figsize=(12, 6))
                    ax.plot(rolling_returns[column], label=f'{window}-Month Rolling Return - {column}')
                    ax.set_title(f'{window}-Month Rolling Return - {column}')
                    ax.set_xlabel('Date')
                    ax.set_ylabel('Rolling Return')
                    ax.legend()
                    ax.grid(True)
                    st.pyplot(fig)
                    plt.clf()

            # Monte Carlo Simulation
            st.header("Monte Carlo Simulation")
            st.write("Simulate future portfolio values based on historical returns and investment strategies.")

            # Input initial investment
            initial_investment = st.number_input("Enter the initial investment amount ($):",
                                                 min_value=0.0, value=10000.0, step=1000.0)

            # Cash flow inputs
            st.subheader("Cash Flow Details")
            num_cash_flows = st.number_input("Number of different cash flows to add:", min_value=0, max_value=10, value=0, step=1)

            cash_flows = []
            if num_cash_flows > 0:
                for i in range(num_cash_flows):
                    st.write(f"**Cash Flow {i+1}:**")
                    amount = st.number_input(f"Amount to invest (positive) or withdraw (negative) for Cash Flow {i+1} ($):",
                                             value=0.0, step=100.0, format="%.2f", key=f"cf_amount_{i}")
                    frequency = st.selectbox(f"Frequency of Cash Flow {i+1}:", ['Daily', 'Weekly', 'Monthly', 'Quarterly', 'Annually'],
                                             index=2, key=f"cf_frequency_{i}")
                    start_year = st.number_input(f"Start year for Cash Flow {i+1} (e.g., 0 for immediate start):",
                                                 min_value=0.0, value=0.0, step=0.1, key=f"cf_start_{i}")
                    end_year = st.number_input(f"End year for Cash Flow {i+1}:",
                                               min_value=start_year, value=start_year+1, step=0.1, key=f"cf_end_{i}")
                    cash_flows.append({
                        'amount': amount,
                        'frequency': frequency,
                        'start_day': int(start_year * 252),  # Convert years to trading days
                        'end_day': int(end_year * 252)       # Convert years to trading days
                    })

            if num_cash_flows >= 0:
                # Time horizon and number of simulations
                time_horizon_years = st.number_input("Enter the investment time horizon in years:",
                                                     min_value=1.0, value=10.0, step=1.0)
                num_simulations = st.number_input("Enter the number of simulations to run:",
                                                  min_value=100, max_value=10000, value=1000, step=100)
                time_horizon_days = int(time_horizon_years * 252)  # Convert years to trading days

                # Select portfolio to simulate
                portfolios_to_simulate = {
                    'Weighted Portfolio Return': non_rebalanced_portfolio.pct_change().dropna(),
                    'Monthly Rebalanced Portfolio': monthly_portfolio.pct_change().dropna(),
                    'Yearly Rebalanced Portfolio': yearly_portfolio.pct_change().dropna()
                }

                selected_portfolio_name = st.selectbox("Select a portfolio for simulation:", list(portfolios_to_simulate.keys()))
                returns_series = portfolios_to_simulate[selected_portfolio_name]

                # Run Monte Carlo Simulation only when button is clicked
                if st.button("Run Monte Carlo Simulation"):
                    st.session_state.simulation_run = True

                if st.session_state.simulation_run:
                    st.write(f"Performing Monte Carlo simulation for **{selected_portfolio_name}**...")
                    simulation_results = monte_carlo_simulation(
                        returns_series,
                        int(num_simulations),
                        time_horizon_days,
                        initial_investment,
                        cash_flows
                    )

                    # Calculate the mean and percentiles
                    ending_values = simulation_results[-1, :]
                    mean_ending_value = np.mean(ending_values)
                    median_ending_value = np.median(ending_values)
                    percentile_5 = np.percentile(ending_values, 5)
                    percentile_95 = np.percentile(ending_values, 95)

                    st.write(f"**Mean ending value after {time_horizon_years} years:** ${mean_ending_value:,.2f}")
                    st.write(f"**Median ending value after {time_horizon_years} years:** ${median_ending_value:,.2f}")
                    st.write(f"**5th percentile:** ${percentile_5:,.2f}")
                    st.write(f"**95th percentile:** ${percentile_95:,.2f}")

                    # Calculate success rate
                    # A simulation is successful if the portfolio value never hits zero or below during the entire period
                    successes = np.sum(np.all(simulation_results > 0, axis=0))
                    success_rate = successes / num_simulations * 100

                    st.write(f"**Success Rate:** {success_rate:.2f}% of simulations never hit zero or below during the period.")

                    # Plot the distribution of ending values
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.hist(ending_values, bins=50, edgecolor='black')
                    ax.set_title(f'Distribution of Ending Portfolio Values - {selected_portfolio_name}')
                    ax.set_xlabel('Ending Portfolio Value ($)')
                    ax.set_ylabel('Frequency')
                    ax.grid(True)
                    st.pyplot(fig)
                    plt.clf()

                    # Plot sample simulated paths
                    fig, ax = plt.subplots(figsize=(12, 6))
                    num_paths_to_plot = min(100, int(num_simulations))
                    ax.plot(simulation_results[:, :num_paths_to_plot])
                    ax.set_title(f'Sample Simulated Portfolio Paths - {selected_portfolio_name}')
                    ax.set_xlabel('Days')
                    ax.set_ylabel('Portfolio Value ($)')
                    ax.grid(True)
                    st.pyplot(fig)
                    plt.clf()
        else:
            # If it's a bear market period, do not perform additional analyses
            st.write("Bear Market Period Analysis: Only cumulative returns and chart are displayed for bear market periods.")
