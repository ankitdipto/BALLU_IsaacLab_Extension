import streamlit as st
import pandas as pd
import time
import os

# --- Configuration ---
CSV_FILE_PATH = "aggregate_actions.csv"  # Path to your CSV file

# Actual column names expected in the CSV header for the actions to be plotted
CSV_COLUMN_ACTION_0 = 'action_0'
CSV_COLUMN_ACTION_1 = 'action_1'
# List of essential columns that must be present in the CSV for plotting
ESSENTIAL_ACTION_COLUMNS = [CSV_COLUMN_ACTION_0, CSV_COLUMN_ACTION_1]

# Display names for the plotted lines in the chart legend
PLOT_DISPLAY_NAME_LEFT = f"Action Left ({CSV_COLUMN_ACTION_0})"
PLOT_DISPLAY_NAME_RIGHT = f"Action Right ({CSV_COLUMN_ACTION_1})"

# How often (in seconds) to check for new data and refresh the chart
REFRESH_INTERVAL_SECONDS = 0.5
# Maximum number of data points to display in the chart (to keep it responsive)
MAX_DATAPOINTS_DISPLAY = 200

# --- Streamlit Page Setup ---
st.set_page_config(
    page_title="Real-time Action Visualizer (CSV with Headers)",
    layout="wide",
    initial_sidebar_state="collapsed"
)
st.title("🤖 Real-time BALLU Policy Actions Viewer")
st.caption(f"Continuously monitoring: `{os.path.abspath(CSV_FILE_PATH)}` (expecting header row with '{CSV_COLUMN_ACTION_0}' and '{CSV_COLUMN_ACTION_1}')")

# --- Placeholders for dynamic content ---
chart_placeholder = st.empty()
status_placeholder = st.empty()
latest_data_placeholder = st.empty()

# --- Main Application Logic ---

# Store the last modification time and size to optimize file reading.
last_mod_time = 0
last_file_size = 0

while True:
    try:
        # Check if the file exists before attempting to get modification time or size.
        if not os.path.exists(CSV_FILE_PATH):
            status_placeholder.warning(
                f"'{CSV_FILE_PATH}' not found. Waiting for your RL algorithm to create and populate it (with headers)..."
            )
            time.sleep(REFRESH_INTERVAL_SECONDS * 2) # Wait longer if file not found
            continue

        current_mod_time = os.path.getmtime(CSV_FILE_PATH)
        current_file_size = os.path.getsize(CSV_FILE_PATH)

        # Only re-process if the file has changed.
        if current_mod_time > last_mod_time or current_file_size != last_file_size:
            last_mod_time = current_mod_time
            last_file_size = current_file_size

            # Read the CSV file, expecting headers from the first row.
            try:
                data_df = pd.read_csv(
                    CSV_FILE_PATH,
                    header=0,  # Use the first row as header
                    skip_blank_lines=True,
                    on_bad_lines='warn'
                )
            except pd.errors.EmptyDataError:
                status_placeholder.info("🔄 CSV file is empty or only contains headers. Waiting for data rows...")
                time.sleep(REFRESH_INTERVAL_SECONDS)
                continue
            except Exception as e: # Catch other potential pandas read errors
                if "No columns to parse from file" in str(e): # Specific error if only header exists but no data
                     status_placeholder.info("🔄 CSV file likely only contains headers. Waiting for data rows...")
                else:
                    status_placeholder.error(f"⚠️ Error reading CSV: {e}. Retrying...")
                time.sleep(REFRESH_INTERVAL_SECONDS * 2) # Wait longer on read error
                continue

            if not data_df.empty:
                # Check if the essential action columns are present in the DataFrame.
                missing_cols = [col for col in ESSENTIAL_ACTION_COLUMNS if col not in data_df.columns]
                if missing_cols:
                    status_placeholder.warning(
                        f"⚠️ CSV is missing expected action columns for plotting: {', '.join(missing_cols)}. "
                        f"Found columns: {', '.join(data_df.columns)}. "
                        "Please ensure your CSV header includes these columns."
                    )
                    time.sleep(REFRESH_INTERVAL_SECONDS * 2)
                    continue
                
                # Prepare DataFrame for plotting.
                plot_df_selected = pd.DataFrame()
                for col in ESSENTIAL_ACTION_COLUMNS:
                    # Convert action columns to numeric, coercing errors to NaN.
                    plot_df_selected[col] = pd.to_numeric(data_df[col], errors='coerce')
                
                # Include other columns if they exist (e.g., 'iteration', 'rollout_step') for potential display
                # For now, we only strictly need action columns for plotting
                # If you want to use 'iteration' or 'rollout_step' as x-axis, modifications would be needed here.

                # Drop rows where any of the essential action columns became NaN after conversion.
                plot_df_cleaned = plot_df_selected.dropna(subset=ESSENTIAL_ACTION_COLUMNS)

                if not plot_df_cleaned.empty:
                    # Limit the number of data points for display.
                    if len(plot_df_cleaned) > MAX_DATAPOINTS_DISPLAY:
                        display_df_actions = plot_df_cleaned[ESSENTIAL_ACTION_COLUMNS].tail(MAX_DATAPOINTS_DISPLAY).reset_index(drop=True)
                    else:
                        display_df_actions = plot_df_cleaned[ESSENTIAL_ACTION_COLUMNS].reset_index(drop=True)

                    # Create a DataFrame for the chart with custom column names for the legend.
                    chart_data = pd.DataFrame({
                        PLOT_DISPLAY_NAME_LEFT: display_df_actions[CSV_COLUMN_ACTION_0],
                        PLOT_DISPLAY_NAME_RIGHT: display_df_actions[CSV_COLUMN_ACTION_1]
                    })

                    # Update the line chart.
                    chart_placeholder.line_chart(chart_data)
                    
                    status_message = (
                        f"📊 Displaying last {len(display_df_actions)} of {len(plot_df_cleaned)} valid data points. "
                        f"(Total data rows read: {len(data_df)}). "
                        f"Updated: {time.strftime('%Y-%m-%d %H:%M:%S')}"
                    )
                    status_placeholder.success(status_message)
                    
                    # Optionally display the latest few raw data points (including other columns if present).
                    with latest_data_placeholder.container():
                        st.markdown("---")
                        st.markdown("### Latest Raw Data Points (from CSV):")
                        # Show tail of original data_df to include 'iteration', 'rollout_step' if they exist
                        display_raw_df = data_df.dropna(subset=ESSENTIAL_ACTION_COLUMNS) # ensure we show rows that had valid actions
                        if len(display_raw_df) > MAX_DATAPOINTS_DISPLAY:
                            st.dataframe(display_raw_df.tail(), use_container_width=True, height=200)
                        else:
                            st.dataframe(display_raw_df, use_container_width=True, height=200)
                else:
                    status_placeholder.info(
                        "ℹ️ CSV contains data, but no valid numeric rows for the action columns after cleaning. "
                        "Waiting for valid data..."
                    )
            else: # data_df was empty after read_csv (e.g. only header row, no data rows)
                status_placeholder.info(
                    "ℹ️ CSV file has no data rows (it might only contain the header). "
                    "Waiting for data..."
                )
        else:
            # File hasn't changed, do nothing this cycle.
            pass

    except FileNotFoundError:
        # This case is primarily handled by the os.path.exists check at the beginning of the loop.
        status_placeholder.error(
            f"🔴 File not found: '{CSV_FILE_PATH}'. "
            "Please ensure the RL script is creating this file with headers."
        )
        time.sleep(REFRESH_INTERVAL_SECONDS * 3) # Wait longer
    except Exception as e:
        status_placeholder.error(f"An unexpected error occurred: {e}")
        # For debugging, you might want to print the full traceback:
        # import traceback
        # st.error(traceback.format_exc())
        time.sleep(REFRESH_INTERVAL_SECONDS * 3) # Wait longer on unexpected errors

    # Wait for the defined interval before the next check.
    time.sleep(REFRESH_INTERVAL_SECONDS)
