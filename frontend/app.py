import streamlit as st
import requests
import pandas as pd
import io


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="SuperKart Sales Forecast",
    page_icon="🛒",
    layout="wide"
)


# ============================================================
# API CONFIGURATION
# ============================================================

API_BASE_URL = "http://127.0.0.1:5050"


SINGLE_API_URL = (
    f"{API_BASE_URL}/v1/salesforecast"
)

BATCH_API_URL = (
    f"{API_BASE_URL}/v1/salesforecastbatch"
)


# ============================================================
# PAGE TITLE
# ============================================================

st.title("🛒 SuperKart Sales Forecast")
st.write(
    "Predict product-store sales using the trained "
    "Random Forest machine learning model."
)

st.divider()


# ============================================================
# TABS
# ============================================================

single_tab, batch_tab = st.tabs(
    [
        "📊 Single Prediction",
        "📁 Batch Prediction"
    ]
)


# ============================================================
# SINGLE PREDICTION
# ============================================================

with single_tab:

    st.subheader("Single Product Sales Prediction")

    col1, col2 = st.columns(2)

    # --------------------------------------------------------
    # Product Information
    # --------------------------------------------------------

    with col1:

        product_weight = st.number_input(
            "Product Weight",
            min_value=0.0,
            value=12.66,
            step=0.01
        )

        product_sugar_content = st.selectbox(
            "Product Sugar Content",
            [
                "Low Sugar",
                "Regular",
                "No Sugar"
            ]
        )

        product_allocated_area = st.number_input(
            "Product Allocated Area",
            min_value=0.0,
            value=0.027,
            step=0.001,
            format="%.3f"
        )

        product_mrp = st.number_input(
            "Product MRP",
            min_value=0.0,
            value=117.08,
            step=0.01
        )

        product_id_char = st.selectbox(
            "Product ID Category",
            [
                "FD",
                "DR",
                "NC"
            ]
        )

    # --------------------------------------------------------
    # Store Information
    # --------------------------------------------------------

    with col2:

        store_size = st.selectbox(
            "Store Size",
            [
                "Small",
                "Medium",
                "High"
            ]
        )

        store_location_city_type = st.selectbox(
            "Store Location City Type",
            [
                "Tier 1",
                "Tier 2",
                "Tier 3"
            ]
        )

        store_type = st.selectbox(
            "Store Type",
            [
                "Supermarket Type1",
                "Supermarket Type2",
                "Supermarket Type3",
                "Departmental Store",
                "Food Mart"
            ]
        )

        store_age_years = st.number_input(
            "Store Age (Years)",
            min_value=0,
            value=16,
            step=1
        )

        product_type_category = st.selectbox(
            "Product Type Category",
            [
                "Perishables",
                "Non Perishables"
            ]
        )

    st.divider()

    # --------------------------------------------------------
    # Predict Button
    # --------------------------------------------------------

    if st.button(
        "🔮 Predict Sales",
        type="primary",
        use_container_width=True
    ):

        payload = {

            "Product_Weight":
                product_weight,

            "Product_Sugar_Content":
                product_sugar_content,

            "Product_Allocated_Area":
                product_allocated_area,

            "Product_MRP":
                product_mrp,

            "Store_Size":
                store_size,

            "Store_Location_City_Type":
                store_location_city_type,

            "Store_Type":
                store_type,

            "Product_Id_char":
                product_id_char,

            "Store_Age_Years":
                store_age_years,

            "Product_Type_Category":
                product_type_category
        }

        try:

            response = requests.post(
                SINGLE_API_URL,
                json=payload,
                timeout=30
            )

            if response.status_code == 200:

                result = response.json()

                prediction = result[
                    "Predicted Sales forecast (in dollars)"
                ]

                st.success(
                    "Sales prediction generated successfully!"
                )

                st.metric(
                    "Predicted Sales",
                    f"${prediction:,.2f}"
                )

            else:

                st.error(
                    f"API Error: {response.status_code}"
                )

                try:
                    st.json(response.json())
                except:
                    st.write(response.text)

        except requests.exceptions.ConnectionError:

            st.error(
                "Unable to connect to the Sales Forecast API. "
                "Please make sure the Flask backend is running."
            )

        except Exception as e:

            st.error(
                f"An unexpected error occurred: {str(e)}"
            )


# ============================================================
# BATCH PREDICTION
# ============================================================

with batch_tab:

    st.subheader("Batch Sales Prediction")

    st.write(
        "Upload a CSV file containing multiple product-store "
        "records to generate sales forecasts."
    )

    uploaded_file = st.file_uploader(
        "Upload Batch CSV",
        type=["csv"]
    )

    if uploaded_file is not None:

        # Read file for preview
        batch_df = pd.read_csv(uploaded_file)

        st.write("### Uploaded Data")

        st.dataframe(
            batch_df,
            use_container_width=True
        )

        st.write(
            f"**Number of records:** {len(batch_df)}"
        )

        st.divider()

        if st.button(
            "🔮 Generate Batch Predictions",
            type="primary",
            use_container_width=True
        ):

            # Reset file pointer
            uploaded_file.seek(0)

            try:

                response = requests.post(
                    BATCH_API_URL,
                    files={
                        "file": (
                            uploaded_file.name,
                            uploaded_file,
                            "text/csv"
                        )
                    },
                    timeout=120
                )

                if response.status_code == 200:

                    result = response.json()

                    predictions = result[
                        "predictions"
                    ]

                    # Convert API response to DataFrame
                    prediction_df = pd.DataFrame(
                        predictions
                    )

                    st.success(
                        f"Successfully generated "
                        f"{len(predictions)} predictions!"
                    )

                    st.write("### Prediction Results")

                    st.dataframe(
                        prediction_df,
                        use_container_width=True
                    )

                    # ------------------------------------
                    # Download Results
                    # ------------------------------------

                    csv_data = prediction_df.to_csv(
                        index=False
                    )

                    st.download_button(
                        label="⬇️ Download Predictions CSV",
                        data=csv_data,
                        file_name="SuperKart_Sales_Predictions.csv",
                        mime="text/csv",
                        use_container_width=True
                    )

                else:

                    st.error(
                        f"API Error: {response.status_code}"
                    )

                    try:
                        st.json(response.json())
                    except:
                        st.write(response.text)

            except requests.exceptions.ConnectionError:

                st.error(
                    "Unable to connect to the Sales Forecast API. "
                    "Please make sure the Flask backend is running."
                )

            except Exception as e:

                st.error(
                    f"An unexpected error occurred: {str(e)}"
                )
