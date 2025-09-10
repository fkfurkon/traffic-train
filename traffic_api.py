"""
Simple API for Traffic Forecasting
API อย่างง่ายสำหรับการพยากรณ์การจราจร

รันด้วยคำสั่ง: python traffic_api.py
"""

from flask import Flask, request, jsonify
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import json
from model_deployment_guide import TrafficForecastingSystem, create_prediction_pipeline

app = Flask(__name__)

# โหลดระบบพยากรณ์
forecasting_system = TrafficForecastingSystem()
predict_func = create_prediction_pipeline()

@app.route('/')
def home():
    """หน้าแรกของ API"""
    return """
    <h1>🚗 Traffic Forecasting API</h1>
    <h2>Available Endpoints:</h2>
    <ul>
        <li><b>POST /predict</b> - พยากรณ์การจราจร</li>
        <li><b>GET /status</b> - ตรวจสอบสถานะระบบ</li>
        <li><b>GET /health</b> - ตรวจสอบสุขภาพระบบ</li>
    </ul>
    
    <h3>Example Usage:</h3>
    <pre>
    curl -X POST http://localhost:5000/predict \
      -H "Content-Type: application/json" \
      -d '{
        "current_vehicle_count": 45,
        "historical_data": [40, 42, 44, 43, 45],
        "timestamp": "2025-09-10T14:30:00"
      }'
    </pre>
    """

@app.route('/predict', methods=['POST'])
def predict():
    """
    พยากรณ์การจราจรชั่วโมงถัดไป
    
    Expected JSON input:
    {
        "current_vehicle_count": 45,
        "historical_data": [40, 42, 44, 43, 45],  # optional
        "timestamp": "2025-09-10T14:30:00"  # optional, default to now
    }
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "ไม่พบข้อมูล JSON"}), 400
            
        # ตรวจสอบข้อมูลที่จำเป็น
        if 'current_vehicle_count' not in data:
            return jsonify({"error": "ต้องระบุ current_vehicle_count"}), 400
            
        current_count = data['current_vehicle_count']
        historical_data = data.get('historical_data', [])
        
        # ตรวจสอบและแปลงเวลา
        if 'timestamp' in data:
            try:
                current_time = datetime.fromisoformat(data['timestamp'].replace('Z', '+00:00'))
            except:
                return jsonify({"error": "รูปแบบเวลาไม่ถูกต้อง ใช้ ISO format"}), 400
        else:
            current_time = datetime.now()
            
        # พยากรณ์
        result = predict_func(current_time, current_count, historical_data)
        
        # ตรวจสอบผลลัพธ์
        if 'error' in result:
            return jsonify(result), 500
            
        # ปรับรูปแบบผลลัพธ์
        response = {
            "success": True,
            "prediction": {
                "ensemble_prediction": float(result['ensemble_prediction']),
                "individual_predictions": {
                    model: float(pred) 
                    for model, pred in result['individual_predictions'].items()
                },
                "confidence": "high" if len(result['individual_predictions']) > 1 else "medium"
            },
            "input": {
                "current_time": current_time.isoformat(),
                "target_time": (current_time + timedelta(hours=1)).isoformat(),
                "current_vehicle_count": current_count,
                "historical_data_points": len(historical_data)
            },
            "metadata": {
                "models_used": list(result['individual_predictions'].keys()),
                "prediction_method": "ensemble" if len(result['individual_predictions']) > 1 else "single_model"
            }
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"เกิดข้อผิดพลาด: {str(e)}"
        }), 500

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """
    พยากรณ์หลายจุดพร้อมกัน
    
    Expected JSON input:
    {
        "predictions": [
            {
                "current_vehicle_count": 45,
                "historical_data": [40, 42, 44, 43, 45],
                "timestamp": "2025-09-10T14:30:00"
            },
            ...
        ]
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'predictions' not in data:
            return jsonify({"error": "ต้องระบุ array ของ predictions"}), 400
            
        results = []
        
        for i, pred_data in enumerate(data['predictions']):
            try:
                current_count = pred_data['current_vehicle_count']
                historical_data = pred_data.get('historical_data', [])
                
                if 'timestamp' in pred_data:
                    current_time = datetime.fromisoformat(pred_data['timestamp'].replace('Z', '+00:00'))
                else:
                    current_time = datetime.now()
                    
                result = predict_func(current_time, current_count, historical_data)
                
                if 'error' not in result:
                    results.append({
                        "index": i,
                        "success": True,
                        "prediction": float(result['ensemble_prediction']),
                        "timestamp": current_time.isoformat(),
                        "target_time": (current_time + timedelta(hours=1)).isoformat()
                    })
                else:
                    results.append({
                        "index": i,
                        "success": False,
                        "error": result['error']
                    })
                    
            except Exception as e:
                results.append({
                    "index": i,
                    "success": False,
                    "error": str(e)
                })
                
        return jsonify({
            "success": True,
            "results": results,
            "total_predictions": len(results),
            "successful_predictions": sum(1 for r in results if r['success'])
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"เกิดข้อผิดพลาด: {str(e)}"
        }), 500

@app.route('/status', methods=['GET'])
def status():
    """ตรวจสอบสถานะระบบ"""
    try:
        # ลองโหลดโมเดล
        temp_system = TrafficForecastingSystem()
        temp_system.load_models()
        
        model_status = {
            "xgb_loaded": temp_system.xgb_model is not None,
            "lstm_loaded": temp_system.lstm_model is not None,
            "sarima_loaded": temp_system.sarima_model is not None,
            "scaler_loaded": temp_system.scaler is not None,
            "is_trained": temp_system.is_trained
        }
        
        return jsonify({
            "status": "healthy" if temp_system.is_trained else "models_not_loaded",
            "models": model_status,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "ok",
        "service": "Traffic Forecasting API",
        "timestamp": datetime.now().isoformat()
    })

if __name__ == '__main__':
    print("🚗 เริ่มต้น Traffic Forecasting API...")
    print("📍 API จะทำงานที่: http://localhost:5000")
    print("📚 ดู documentation ที่: http://localhost:5000")
    print("\n⚠️  หมาย​เหตุ: ต้องมีโมเดลที่ฝึกแล้วในโฟลเดอร์ ./models/")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
