# 🚗 Traffic Forecasting System
# ระบบพยากรณ์การจราจร

โปรเจกต์นี้เปรียบเทียบประสิทธิภาพของโมเดลพยากรณ์การจราจร 3 แบบ และให้เครื่องมือสำหรับนำไปใช้งานจริง

## 📋 โมเดลที่เปรียบเทียบ

1. **XGBoost** - Gradient Boosting Model
   - ✅ เร็ว และใช้ทรัพยากรน้อย
   - ✅ ตีความผลได้ง่าย (Feature Importance)
   - ✅ เหมาะสำหรับ Real-time prediction

2. **SARIMA** - Seasonal Autoregressive Integrated Moving Average
   - ✅ เข้าใจรูปแบบตามฤดูกาลได้ดี
   - ✅ เหมาะสำหรับการวิเคราะห์แนวโน้ม
   - ⚠️ ใช้เวลาในการฝึกมาก

3. **LSTM** - Long Short-Term Memory Neural Network
   - ✅ จับรูปแบบซับซ้อนได้ดี
   - ✅ เหมาะสำหรับข้อมูลที่มีความสัมพันธ์ระยะยาว
   - ⚠️ ต้องการข้อมูลมาก และใช้เวลาฝึกนาน

## 🗂️ ไฟล์ในโปรเจกต์

```
📁 Traffic Forecasting Project/
├── 📊 traffic_dataset1.csv                    # ข้อมูล traffic
├── 📓 traffic_forecasting_comparison.ipynb    # โน้ตบุ๊กหลัก
├── 🔧 model_deployment_guide.py               # คลาสสำหรับใช้งานโมเดล
├── 🌐 traffic_api.py                         # Flask API
├── 💾 save_models.py                         # สคริปต์บันทึกโมเดล
├── 📝 usage_example.py                       # ตัวอย่างการใช้งาน
├── 📦 requirements_deployment.txt            # ไลบรารีที่จำเป็น
├── 📋 requirements.txt                       # ไลบรารีสำหรับ notebook
└── 📁 models/                               # โฟลเดอร์โมเดลที่บันทึก
    ├── xgb_model.pkl
    ├── lstm_model.h5
    ├── scaler.pkl
    ├── feature_columns.pkl
    ├── sarima_model.pkl (optional)
    └── metadata.json
```

## 🚀 วิธีการใช้งาน

### 1. การฝึกโมเดล
```bash
# รัน Jupyter notebook
jupyter notebook traffic_forecasting_comparison.ipynb

# หรือใช้ VS Code notebook extension
```

### 2. การบันทึกโมเดล
หลังจากฝึกโมเดลในโน้ตบุ๊กแล้ว รัน cell สุดท้ายเพื่อบันทึกโมเดล หรือใช้:
```python
from save_models import save_trained_models
save_trained_models(xgb_model, lstm_model, scaler, feature_cols)
```

### 3. การใช้งานแบบง่าย
```bash
python usage_example.py
```

### 4. การใช้งานผ่าน API
```bash
# ติดตั้งไลบรารีที่จำเป็น
pip install -r requirements_deployment.txt

# รัน API server
python traffic_api.py

# ทดสอบ API
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "current_vehicle_count": 45,
    "historical_data": [40, 42, 44, 43, 45],
    "timestamp": "2025-09-10T14:30:00"
  }'
```

### 5. การใช้งานในโค้ด
```python
from model_deployment_guide import create_prediction_pipeline
from datetime import datetime

# สร้างฟังก์ชันพยากรณ์
predict_func = create_prediction_pipeline()

# พยากรณ์
result = predict_func(
    datetime.now(), 
    45,  # จำนวนรถปัจจุบัน
    [40, 42, 44, 43, 45]  # ข้อมูลย้อนหลัง
)

print(f"พยากรณ์: {result['ensemble_prediction']:.1f} คัน")
```

## 📊 API Endpoints

### POST /predict
พยากรณ์การจราจรจุดเดียว

**Request:**
```json
{
  "current_vehicle_count": 45,
  "historical_data": [40, 42, 44, 43, 45],
  "timestamp": "2025-09-10T14:30:00"
}
```

**Response:**
```json
{
  "success": true,
  "prediction": {
    "ensemble_prediction": 47.5,
    "individual_predictions": {
      "XGBoost": 46.2,
      "LSTM": 48.8
    },
    "confidence": "high"
  },
  "input": {
    "current_time": "2025-09-10T14:30:00",
    "target_time": "2025-09-10T15:30:00",
    "current_vehicle_count": 45
  }
}
```

### POST /batch_predict
พยากรณ์การจราจรหลายจุด

### GET /status
ตรวจสอบสถานะโมเดล

### GET /health
Health check

## 🔧 การปรับแต่ง

### ปรับน้ำหนักโมเดล
```python
# ในการเรียก predict_ensemble
result = system.predict_ensemble(
    timestamp=datetime.now(),
    vehicle_count=45,
    sequence_data=historical_data,
    weights=[0.7, 0.3]  # XGBoost 70%, LSTM 30%
)
```

### เพิ่มโมเดลใหม่
1. เพิ่มโมเดลใน `TrafficForecastingSystem` class
2. เพิ่มฟังก์ชัน predict สำหรับโมเดลใหม่
3. อัพเดต `predict_ensemble` method

## 📈 ประสิทธิภาพโมเดล

| โมเดล | RMSE | MAE | R² | ความเร็ว | ทรัพยากร |
|-------|------|-----|----|---------|---------| 
| XGBoost | 🟢 ต่ำ | 🟢 ต่ำ | 🟢 สูง | 🟢 เร็ว | 🟢 น้อย |
| SARIMA | 🟡 ปานกลาง | 🟡 ปานกลาง | 🟡 ปานกลาง | 🔴 ช้า | 🟡 ปานกลาง |
| LSTM | 🟢 ต่ำ | 🟢 ต่ำ | 🟢 สูง | 🔴 ช้า | 🔴 มาก |

*ผลลัพธ์อาจแตกต่างกันขึ้นอยู่กับข้อมูล*

## 🎯 คำแนะนำการใช้งาน

### สำหรับ Real-time Application
- ใช้ **XGBoost** สำหรับความเร็ว
- ใช้ **Ensemble** ถ้าต้องการความแม่นยำสูง

### สำหรับ Batch Processing
- ใช้ **LSTM** สำหรับความแม่นยำสูงสุด
- ใช้ **SARIMA** สำหรับการวิเคราะห์แนวโน้ม

### สำหรับ Mobile/Edge Device
- ใช้ **XGBoost** เท่านั้น
- ลดจำนวน features ถ้าจำเป็น

## 🐛 การแก้ปัญหา

### โมเดลโหลดไม่ได้
```bash
# ตรวจสอบไฟล์โมเดล
ls -la models/

# ฝึกโมเดลใหม่
jupyter notebook traffic_forecasting_comparison.ipynb
```

### API ไม่ทำงาน
```bash
# ตรวจสอบ dependencies
pip install -r requirements_deployment.txt

# ตรวจสอบ port
netstat -an | grep 5000
```

### การพยากรณ์ไม่แม่นยำ
- ตรวจสอบคุณภาพข้อมูล input
- เพิ่มข้อมูล historical_data
- ปรับ weights ในการรวมโมเดล

## 📞 การติดต่อ

หากมีคำถามหรือต้องการความช่วยเหลือ สามารถ:
- เปิด Issue ใน GitHub repository
- ติดต่อผู้พัฒนา

## 📄 License

MIT License - ใช้งานได้อย่างอิสระ

---

### 🔄 อัพเดทล่าสุด
- เพิ่ม API endpoint สำหรับ batch prediction
- ปรับปรุงการจัดการข้อผิดพลาด
- เพิ่มตัวอย่างการใช้งานจริง

**Happy Traffic Forecasting! 🚗📊**
