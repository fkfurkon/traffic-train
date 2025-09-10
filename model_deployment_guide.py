"""
Traffic Forecasting Model Deployment Guide
การนำโมเดลการพยากรณ์การจราจรไปใช้งานจริง

วิธีการใช้งาน:
1. บันทึกโมเดลที่ได้ฝึกแล้ว
2. สร้างฟังก์ชันสำหรับการพยากรณ์
3. สร้าง API หรือสคริปต์สำหรับการใช้งาน
"""

import pandas as pd
import numpy as np
import pickle
import joblib
from datetime import datetime, timedelta
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class TrafficForecastingSystem:
    """
    ระบบพยากรณ์การจราจรที่รวมโมเดลทั้ง 3 แบบเข้าด้วยกัน
    """
    
    def __init__(self):
        self.xgb_model = None
        self.lstm_model = None
        self.sarima_model = None
        self.scaler = None
        self.feature_columns = None
        self.is_trained = False
        
    def save_models(self, xgb_model, lstm_model, scaler, feature_columns, 
                   sarima_model=None, save_dir="./models/"):
        """
        บันทึกโมเดลทั้งหมดลงในไฟล์
        
        Parameters:
        - xgb_model: โมเดล XGBoost ที่ฝึกแล้ว
        - lstm_model: โมเดล LSTM ที่ฝึกแล้ว
        - scaler: StandardScaler สำหรับ LSTM
        - feature_columns: รายชื่อ features
        - sarima_model: โมเดล SARIMA (optional)
        - save_dir: โฟลเดอร์สำหรับบันทึกโมเดล
        """
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        # บันทึก XGBoost
        joblib.dump(xgb_model, f"{save_dir}xgb_model.pkl")
        
        # บันทึก LSTM
        lstm_model.save(f"{save_dir}lstm_model.h5")
        
        # บันทึก Scaler
        joblib.dump(scaler, f"{save_dir}scaler.pkl")
        
        # บันทึก feature columns
        with open(f"{save_dir}feature_columns.pkl", 'wb') as f:
            pickle.dump(feature_columns, f)
            
        # บันทึก SARIMA (ถ้ามี)
        if sarima_model:
            joblib.dump(sarima_model, f"{save_dir}sarima_model.pkl")
            
        print(f"✅ บันทึกโมเดลทั้งหมดแล้วที่ {save_dir}")
        
    def load_models(self, load_dir="./models/"):
        """
        โหลดโมเดลทั้งหมดจากไฟล์
        """
        try:
            # โหลด XGBoost
            self.xgb_model = joblib.load(f"{load_dir}xgb_model.pkl")
            
            # โหลด LSTM
            self.lstm_model = tf.keras.models.load_model(f"{load_dir}lstm_model.h5")
            
            # โหลด Scaler
            self.scaler = joblib.load(f"{load_dir}scaler.pkl")
            
            # โหลด feature columns
            with open(f"{load_dir}feature_columns.pkl", 'rb') as f:
                self.feature_columns = pickle.load(f)
                
            # โหลด SARIMA (ถ้ามี)
            try:
                self.sarima_model = joblib.load(f"{load_dir}sarima_model.pkl")
            except:
                print("⚠️ ไม่พบโมเดล SARIMA")
                
            self.is_trained = True
            print(f"✅ โหลดโมเดลทั้งหมดแล้วจาก {load_dir}")
            
        except Exception as e:
            print(f"❌ ไม่สามารถโหลดโมเดลได้: {str(e)}")
            
    def prepare_features(self, timestamp, vehicle_count, lag_values=None):
        """
        เตรียมข้อมูล features สำหรับการพยากรณ์
        
        Parameters:
        - timestamp: วันที่และเวลา (datetime object)
        - vehicle_count: จำนวนรถณขณะปัจจุบัน
        - lag_values: ค่า lag [lag_1, lag_2, lag_3] (optional)
        
        Returns:
        - DataFrame พร้อม features ทั้งหมด
        """
        # สร้าง features พื้นฐาน
        features = {
            'vehicle_count': vehicle_count,
            'hour': timestamp.hour,
        }
        
        # เพิ่ม lag values
        if lag_values:
            features['lag_1'] = lag_values[0]
            features['lag_2'] = lag_values[1] if len(lag_values) > 1 else vehicle_count
            features['lag_3'] = lag_values[2] if len(lag_values) > 2 else vehicle_count
        else:
            # ใช้ค่าปัจจุบันแทน lag (สำหรับการทดสอบ)
            features['lag_1'] = vehicle_count
            features['lag_2'] = vehicle_count
            features['lag_3'] = vehicle_count
            
        # เพิ่ม day of week features
        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        for i, day in enumerate(day_names):
            features[f'day_{day}'] = 1 if timestamp.weekday() == i else 0
            
        return pd.DataFrame([features])
    
    def predict_xgboost(self, features_df):
        """
        พยากรณ์ด้วย XGBoost
        """
        if not self.is_trained or self.xgb_model is None:
            raise ValueError("โมเดล XGBoost ยังไม่ได้โหลด")
            
        # จัดเรียง columns ให้ตรงกับที่ใช้ฝึก
        features_aligned = features_df.reindex(columns=self.feature_columns, fill_value=0)
        
        prediction = self.xgb_model.predict(features_aligned)[0]
        return prediction
    
    def predict_lstm(self, sequence_data):
        """
        พยากรณ์ด้วย LSTM
        
        Parameters:
        - sequence_data: array ของ vehicle_count ย้อนหลัง 24 จุด (2 ชั่วโมง)
        """
        if not self.is_trained or self.lstm_model is None:
            raise ValueError("โมเดล LSTM ยังไม่ได้โหลด")
            
        # Scale ข้อมูล
        sequence_scaled = self.scaler.transform(np.array(sequence_data).reshape(-1, 1))
        
        # Reshape สำหรับ LSTM
        sequence_reshaped = sequence_scaled.reshape(1, len(sequence_data), 1)
        
        # พยากรณ์
        prediction_scaled = self.lstm_model.predict(sequence_reshaped, verbose=0)
        
        # Inverse transform
        prediction = self.scaler.inverse_transform(prediction_scaled)[0][0]
        
        return prediction
    
    def predict_ensemble(self, timestamp, vehicle_count, sequence_data=None, lag_values=None, weights=None):
        """
        พยากรณ์แบบรวม (Ensemble) จากโมเดลทั้งหมด
        
        Parameters:
        - timestamp: วันที่และเวลา
        - vehicle_count: จำนวนรถปัจจุบัน
        - sequence_data: ข้อมูลย้อนหลัง 24 จุดสำหรับ LSTM
        - lag_values: ค่า lag สำหรับ XGBoost
        - weights: น้ำหนักสำหรับแต่ละโมเดล [xgb_weight, lstm_weight]
        
        Returns:
        - การพยากรณ์แบบรวม
        """
        if weights is None:
            weights = [0.5, 0.5]  # น้ำหนักเท่ากัน
            
        predictions = []
        model_names = []
        
        # XGBoost prediction
        try:
            features_df = self.prepare_features(timestamp, vehicle_count, lag_values)
            xgb_pred = self.predict_xgboost(features_df)
            predictions.append(xgb_pred)
            model_names.append("XGBoost")
        except Exception as e:
            print(f"⚠️ XGBoost prediction failed: {e}")
            
        # LSTM prediction
        if sequence_data and len(sequence_data) >= 24:
            try:
                lstm_pred = self.predict_lstm(sequence_data[-24:])  # ใช้ 24 จุดล่าสุด
                predictions.append(lstm_pred)
                model_names.append("LSTM")
            except Exception as e:
                print(f"⚠️ LSTM prediction failed: {e}")
        
        if not predictions:
            raise ValueError("ไม่สามารถพยากรณ์ได้จากโมเดลใดๆ")
            
        # คำนวณค่าเฉลี่ยถ่วงน้ำหนัก
        if len(predictions) == 2:
            ensemble_pred = predictions[0] * weights[0] + predictions[1] * weights[1]
        else:
            ensemble_pred = predictions[0]  # ใช้โมเดลเดียวที่ทำงานได้
            
        return {
            'ensemble_prediction': ensemble_pred,
            'individual_predictions': dict(zip(model_names, predictions))
        }

def create_prediction_pipeline():
    """
    สร้าง pipeline สำหรับการพยากรณ์แบบอัตโนมัติ
    """
    
    def predict_next_hour(current_time, current_vehicle_count, historical_data=None):
        """
        พยากรณ์การจราจรชั่วโมงถัดไป
        
        Parameters:
        - current_time: เวลาปัจจุบัน (datetime)
        - current_vehicle_count: จำนวนรถปัจจุบัน
        - historical_data: ข้อมูลย้อนหลัง (list of vehicle counts)
        
        Returns:
        - การพยากรณ์สำหรับชั่วโมงถัดไป
        """
        
        # โหลดระบบพยากรณ์
        forecasting_system = TrafficForecastingSystem()
        forecasting_system.load_models()
        
        if not forecasting_system.is_trained:
            return {"error": "ไม่สามารถโหลดโมเดลได้"}
        
        # เตรียม lag values
        lag_values = None
        if historical_data and len(historical_data) >= 3:
            lag_values = historical_data[-3:]  # ใช้ 3 ค่าล่าสุด
        
        try:
            # พยากรณ์แบบรวม
            result = forecasting_system.predict_ensemble(
                timestamp=current_time,
                vehicle_count=current_vehicle_count,
                sequence_data=historical_data,
                lag_values=lag_values
            )
            
            # เพิ่มข้อมูลเวลา
            result['prediction_time'] = current_time
            result['target_time'] = current_time + timedelta(hours=1)
            result['current_vehicle_count'] = current_vehicle_count
            
            return result
            
        except Exception as e:
            return {"error": f"เกิดข้อผิดพลาดในการพยากรณ์: {str(e)}"}
    
    return predict_next_hour

# ตัวอย่างการใช้งาน
if __name__ == "__main__":
    
    print("🚗 Traffic Forecasting System - Model Deployment Guide")
    print("=" * 60)
    
    # ตัวอย่างการใช้งาน
    print("\n📝 ตัวอย่างการใช้งาน:")
    print("1. บันทึกโมเดล:")
    print("""
    # หลังจากฝึกโมเดลแล้ว
    system = TrafficForecastingSystem()
    system.save_models(
        xgb_model=xgb_model,
        lstm_model=lstm_model, 
        scaler=scaler,
        feature_columns=feature_cols
    )
    """)
    
    print("\n2. โหลดโมเดลและพยากรณ์:")
    print("""
    # โหลดโมเดลและพยากรณ์
    predict_func = create_prediction_pipeline()
    
    # ข้อมูลตัวอย่าง
    current_time = datetime.now()
    current_count = 45  # จำนวนรถปัจจุบัน
    historical_data = [40, 42, 44, 43, 45]  # ข้อมูลย้อนหลัง
    
    # พยากรณ์
    result = predict_func(current_time, current_count, historical_data)
    print(f"การพยากรณ์: {result}")
    """)
    
    print("\n🔧 การปรับแต่งขั้นสูง:")
    print("- ปรับน้ำหนักของแต่ละโมเดลในการรวมผล")
    print("- เพิ่มการตรวจสอบคุณภาพข้อมูล")
    print("- สร้าง API endpoint สำหรับเรียกใช้")
    print("- เพิ่มการบันทึก log และ monitoring")
