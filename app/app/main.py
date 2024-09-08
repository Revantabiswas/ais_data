from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
from tensorflow.keras.models import load_model
import joblib
import pickle
from tensorflow.keras.losses import MeanSquaredError
from io import StringIO

app = FastAPI()

custom_objects = {'mse': MeanSquaredError()}
# Load models
dbscan_model = pickle.load(open(r'D:\Projects\AIS-Anamoly_detector\new_3\app\models\dbscan_model.pkl', 'rb'))
isolation_forest_model = joblib.load(r'D:\Projects\AIS-Anamoly_detector\new_3\app\models\isolation_forest_model.joblib')
autoencoder_model = load_model(r'D:\Projects\AIS-Anamoly_detector\new_3\app\models\autoencoder_model.h5', custom_objects=custom_objects)

@app.get("/", response_class=HTMLResponse)
def index():
    return """
    <h1>AIS Data Anomaly Detection</h1>
    <form action="/upload/" enctype="multipart/form-data" method="post">
        <input name="file" type="file">
        <button type="submit">Upload CSV</button>
    </form>
    """

@app.post("/upload/", response_class=HTMLResponse)
async def upload_csv(file: UploadFile = File(...)):
    content = await file.read()
    ais_data = pd.read_csv(StringIO(content.decode("utf-8")))
    
    mmsi_numbers = ais_data['MMSI'].unique().tolist()
    
    # Create a dropdown menu for selecting MMSI numbers
    options = "".join([f'<option value="{mmsi}">{mmsi}</option>' for mmsi in mmsi_numbers])
    
    return f"""
    <h1>Select MMSI Number</h1>
    <form action="/process/" method="post">
        <select name="mmsi">
            {options}
        </select>
        <button type="submit">Process MMSI</button>
    </form>
    """

@app.post("/process/", response_class=HTMLResponse)
async def process_mmsi(mmsi: int = Form(...)):
    # Load the AIS data
    ais_data = pd.read_csv(r'D:\Projects\AIS-Anamoly_detector\new_3\app\data/ais_data.csv')
    
    # Filter data for the selected MMSI number
    vessel_data = ais_data[ais_data['MMSI'] == mmsi]
    
    if vessel_data.empty:
        return f"<h1>No data found for MMSI: {mmsi}</h1>"
    
    # Feature Engineering
    vessel_data['speed_change'] = vessel_data['SOG'].diff()
    vessel_data['acceleration'] = vessel_data['speed_change'].diff()
    vessel_data['turning_rate'] = vessel_data['Heading'].diff()
    vessel_data['sog_anomaly'] = vessel_data['SOG'] > (vessel_data['SOG'].mean() + 2 * vessel_data['SOG'].std())
    vessel_data['draught_change'] = vessel_data['Draft'].diff()

    # Apply anomaly detection models
    features = vessel_data[['SOG', 'COG', 'LAT', 'LON']].values

    # DBSCAN
    vessel_data['anomaly_dbscan'] = dbscan_model.fit_predict(features) == -1

    # Isolation Forest
    vessel_data['anomaly_isolation_forest'] = isolation_forest_model.predict(features) == -1

    # Autoencoder
    features_norm = (features - features.mean(axis=0)) / features.std(axis=0)
    reconstructions = autoencoder_model.predict(features_norm)
    mse = np.mean(np.power(features_norm - reconstructions, 2), axis=1)
    vessel_data['anomaly_autoencoder'] = mse > np.percentile(mse, 95)

    # Kalman Filter
    with open(r'D:\Projects\AIS-Anamoly_detector\new_3\app\models\SOG_kalman_filter.pkl', 'rb') as f:
        kalman_filter = pickle.load(f)
    state_means, _ = kalman_filter.em(vessel_data['SOG'].values).filter(vessel_data['SOG'].values)
    vessel_data['SOG_kalman'] = state_means
    vessel_data['anomaly_kalman'] = np.abs(vessel_data['SOG'] - state_means) > 2 * vessel_data['SOG'].std()

    # Combine anomalies
    vessel_data['combined_anomaly'] = vessel_data[['anomaly_dbscan', 'anomaly_isolation_forest', 'anomaly_autoencoder', 'anomaly_kalman']].sum(axis=1) > 2

    # Save and display results
    vessel_data.to_csv('data/vessel_data_with_anomalies.csv', index=False)
    
    return f"<h1>Anomalies Processed for MMSI: {mmsi}</h1><p>Total anomalies detected: {vessel_data['combined_anomaly'].sum()}</p>"

@app.get("/results/", response_class=HTMLResponse)
def display_results():
    vessel_data = pd.read_csv('data/vessel_data_with_anomalies.csv')
    return vessel_data.to_html()
