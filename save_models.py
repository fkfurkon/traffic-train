"""
Save Models Script
สคริปต์สำหรับบันทึกโมเดลจากโน้ตบุ๊กเพื่อนำไปใช้งาน

รันไฟล์นี้หลังจากฝึกโมเดลในโน้ตบุ๊กเสร็จแล้ว
"""

import os
import sys
import joblib
import pickle
import pandas as pd
import numpy as np
from datetime import datetime

def save_trained_models(xgb_model, lstm_model, scaler, feature_columns, 
                       sarima_model=None, save_dir="./models/"):
    """
    บันทึกโมเดลที่ฝึกแล้วทั้งหมด
    
    วิธีใช้:
    1. รันโน้ตบุ๊กจนจบ
    2. เรียกฟังก์ชันนี้ด้วยโมเดลที่ได้
    """
    
    print("💾 กำลังบันทึกโมเดลทั้งหมด...")
    
    # สร้างโฟลเดอร์
    os.makedirs(save_dir, exist_ok=True)
    
    try:
        # บันทึก XGBoost
        joblib.dump(xgb_model, f"{save_dir}xgb_model.pkl")
        print("✅ บันทึก XGBoost model แล้ว")
        
        # บันทึก LSTM
        lstm_model.save(f"{save_dir}lstm_model.h5")
        print("✅ บันทึก LSTM model แล้ว")
        
        # บันทึก Scaler
        joblib.dump(scaler, f"{save_dir}scaler.pkl")
        print("✅ บันทึก Scaler แล้ว")
        
        # บันทึก feature columns
        with open(f"{save_dir}feature_columns.pkl", 'wb') as f:
            pickle.dump(feature_columns, f)
        print("✅ บันทึก Feature columns แล้ว")
        
        # บันทึก SARIMA (ถ้ามี)
        if sarima_model:
            joblib.dump(sarima_model, f"{save_dir}sarima_model.pkl")
            print("✅ บันทึก SARIMA model แล้ว")
        
        # บันทึกข้อมูล metadata
        metadata = {
            'saved_at': datetime.now().isoformat(),
            'models_saved': ['xgb', 'lstm', 'scaler'],
            'feature_count': len(feature_columns),
            'feature_columns': feature_columns
        }
        
        if sarima_model:
            metadata['models_saved'].append('sarima')
            
        with open(f"{save_dir}metadata.json", 'w') as f:
            import json
            json.dump(metadata, f, indent=2)
        print("✅ บันทึก Metadata แล้ว")
        
        print(f"\n🎉 บันทึกโมเดลทั้งหมดสำเร็จที่ {save_dir}")
        print(f"📁 ไฟล์ที่บันทึก:")
        for file in os.listdir(save_dir):
            print(f"   - {file}")
            
    except Exception as e:
        print(f"❌ เกิดข้อผิดพลาดในการบันทึก: {str(e)}")

def create_model_save_cell():
    """
    สร้าง code cell สำหรับเพิ่มในโน้ตบุ๊ก
    """
    code = '''
# บันทึกโมเดลทั้งหมดเพื่อนำไปใช้งาน
from save_models import save_trained_models

# บันทึกโมเดล (รันหลังจากฝึกโมเดลทั้งหมดแล้ว)
save_trained_models(
    xgb_model=xgb_model,
    lstm_model=lstm_model,
    scaler=scaler,
    feature_columns=feature_cols,
    sarima_model=sarima_fitted if sarima_success else None
)

print("🎯 โมเดลพร้อมใช้งานแล้ว!")
print("🔧 ขั้นตอนถัดไป:")
print("1. รัน: python traffic_api.py (สำหรับ API)")
print("2. หรือใช้ model_deployment_guide.py สำหรับใช้งานโดยตรง")
'''
    
    return code

# ตัวอย่างการใช้งานทดสอบ
def test_saved_models(models_dir="./models/"):
    """
    ทดสอบโมเดลที่บันทึกไว้
    """
    print("🧪 ทดสอบโมเดลที่บันทึกไว้...")
    
    from model_deployment_guide import TrafficForecastingSystem
    from datetime import datetime
    
    # โหลดระบบ
    system = TrafficForecastingSystem()
    system.load_models(models_dir)
    
    if not system.is_trained:
        print("❌ ไม่สามารถโหลดโมเดลได้")
        return False
    
    # ทดสอบการพยากรณ์
    test_time = datetime.now()
    test_vehicle_count = 45
    test_historical = [40, 42, 44, 43, 45, 47, 46, 45]
    
    try:
        # ทดสอบ XGBoost
        features = system.prepare_features(test_time, test_vehicle_count, [43, 45, 47])
        xgb_pred = system.predict_xgboost(features)
        print(f"✅ XGBoost prediction: {xgb_pred:.2f}")
        
        # ทดสอบ LSTM (ต้องมีข้อมูลอย่างน้อย 24 จุด)
        if len(test_historical) >= 24:
            lstm_pred = system.predict_lstm(test_historical[-24:])
            print(f"✅ LSTM prediction: {lstm_pred:.2f}")
        else:
            # เติมข้อมูลให้ครบ 24 จุด
            extended_historical = test_historical + [test_vehicle_count] * (24 - len(test_historical))
            lstm_pred = system.predict_lstm(extended_historical)
            print(f"✅ LSTM prediction (extended data): {lstm_pred:.2f}")
        
        # ทดสอบ ensemble
        result = system.predict_ensemble(
            timestamp=test_time,
            vehicle_count=test_vehicle_count,
            sequence_data=test_historical,
            lag_values=[43, 45, 47]
        )
        
        print(f"✅ Ensemble prediction: {result['ensemble_prediction']:.2f}")
        print(f"   Individual models: {result['individual_predictions']}")
        
        print("\n🎉 การทดสอบสำเร็จทั้งหมด!")
        return True
        
    except Exception as e:
        print(f"❌ การทดสอบล้มเหลว: {str(e)}")
        return False

if __name__ == "__main__":
    print("📝 Model Save Script")
    print("=" * 50)
    
    print("\n🔧 วิธีใช้งาน:")
    print("1. ฝึกโมเดลในโน้ตบุ๊กจนเสร็จ")
    print("2. เพิ่ม code cell ในโน้ตบุ๊ก:")
    print("\n" + create_model_save_cell())
    
    # ตรวจสอบว่ามีโมเดลบันทึกไว้หรือไม่
    if os.path.exists("./models/"):
        print("\n🔍 พบโฟลเดอร์ models/ อยู่แล้ว")
        print("📋 ไฟล์ในโฟลเดอร์:")
        for file in os.listdir("./models/"):
            print(f"   - {file}")
            
        # ทดสอบโมเดล
        test_choice = input("\n❓ ต้องการทดสอบโมเดลหรือไม่? (y/n): ")
        if test_choice.lower() == 'y':
            test_saved_models()
    else:
        print("\n⚠️  ยังไม่พบโฟลเดอร์ models/")
        print("💡 กรุณาฝึกโมเดลในโน้ตบุ๊กก่อน")
        
    print("\n📚 ขั้นตอนถัดไป:")
    print("1. รัน: python traffic_api.py (สำหรับ API)")
    print("2. หรือใช้ในโค้ดโดยตรง:")
    print("""
from model_deployment_guide import create_prediction_pipeline
predict_func = create_prediction_pipeline()
result = predict_func(datetime.now(), 45, [40, 42, 44])
print(result)
    """)
