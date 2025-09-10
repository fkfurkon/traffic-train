"""
Simple Usage Example
ตัวอย่างการใช้งานโมเดลแบบง่าย

สำหรับผู้ที่ต้องการใช้โมเดลโดยไม่ผ่าน API
"""

from datetime import datetime, timedelta
import pandas as pd
import numpy as np

def simple_traffic_prediction():
    """
    ฟังก์ชันพยากรณ์การจราจรแบบง่าย
    """
    
    print("🚗 ระบบพยากรณ์การจราจรอย่างง่าย")
    print("=" * 50)
    
    try:
        # โหลดระบบพยากรณ์
        from model_deployment_guide import create_prediction_pipeline
        predict_func = create_prediction_pipeline()
        
        # ข้อมูลตัวอย่าง
        current_time = datetime.now()
        current_vehicle_count = 45
        
        # ข้อมูลย้อนหลัง (สมมติ)
        historical_data = [
            40, 42, 38, 44, 46, 43, 41, 45, 47, 44,
            42, 40, 43, 45, 44, 46, 48, 45, 43, 41,
            44, 46, 45, 47, 43, 45  # 26 จุดข้อมูล
        ]
        
        print(f"📅 เวลาปัจจุบัน: {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"🚙 จำนวนรถปัจจุบัน: {current_vehicle_count}")
        print(f"📊 ข้อมูลย้อนหลัง: {len(historical_data)} จุด")
        
        # พยากรณ์
        result = predict_func(current_time, current_vehicle_count, historical_data)
        
        if 'error' in result:
            print(f"❌ เกิดข้อผิดพลาด: {result['error']}")
            return
            
        # แสดงผลลัพธ์
        print(f"\n🔮 การพยากรณ์สำหรับ {(current_time + timedelta(hours=1)).strftime('%H:%M:%S')}:")
        print(f"📈 จำนวนรถที่คาดการณ์: {result['ensemble_prediction']:.1f} คัน")
        
        print(f"\n📋 รายละเอียดจากแต่ละโมเดล:")
        for model_name, prediction in result['individual_predictions'].items():
            print(f"   • {model_name}: {prediction:.1f} คัน")
            
        # คำแนะนำ
        prediction_value = result['ensemble_prediction']
        if prediction_value > current_vehicle_count * 1.2:
            status = "🔴 การจราจรจะแออัดขึ้น"
            advice = "แนะนำหลีกเลี่ยงเส้นทางนี้"
        elif prediction_value < current_vehicle_count * 0.8:
            status = "🟢 การจราจรจะโล่งขึ้น"
            advice = "เวลาที่ดีสำหรับเดินทาง"
        else:
            status = "🟡 การจราจรคงที่"
            advice = "สภาพการจราจรไม่เปลี่ยนมาก"
            
        print(f"\n{status}")
        print(f"💡 คำแนะนำ: {advice}")
        
    except ImportError:
        print("❌ ไม่พบโมเดลหรือไลบรารีที่จำเป็น")
        print("💡 กรุณาตรวจสอบว่าได้ฝึกและบันทึกโมเดลแล้ว")
    except Exception as e:
        print(f"❌ เกิดข้อผิดพลาด: {str(e)}")

def batch_prediction_example():
    """
    ตัวอย่างการพยากรณ์หลายจุดเวลา
    """
    
    print("\n🔄 ตัวอย่างการพยากรณ์หลายจุดเวลา")
    print("=" * 50)
    
    try:
        from model_deployment_guide import create_prediction_pipeline
        predict_func = create_prediction_pipeline()
        
        # ข้อมูลสำหรับหลายจุดเวลา
        base_time = datetime.now()
        test_data = [
            {"time": base_time, "count": 45, "historical": [40, 42, 44, 43, 45] * 5},
            {"time": base_time + timedelta(hours=1), "count": 48, "historical": [42, 44, 46, 45, 48] * 5},
            {"time": base_time + timedelta(hours=2), "count": 52, "historical": [44, 46, 48, 50, 52] * 5},
            {"time": base_time + timedelta(hours=3), "count": 38, "historical": [42, 40, 38, 36, 38] * 5},
        ]
        
        results = []
        
        print("⏱️  กำลังพยากรณ์...")
        for i, data in enumerate(test_data):
            result = predict_func(data["time"], data["count"], data["historical"])
            
            if 'error' not in result:
                results.append({
                    "time": data["time"].strftime("%H:%M"),
                    "current": data["count"],
                    "predicted": result['ensemble_prediction'],
                    "change": result['ensemble_prediction'] - data["count"]
                })
            
        # แสดงผลลัพธ์
        print(f"\n📊 ผลการพยากรณ์:")
        print(f"{'เวลา':>6} | {'ปัจจุบัน':>8} | {'พยากรณ์':>8} | {'เปลี่ยนแปลง':>10} | {'แนวโน้ม':>8}")
        print("-" * 50)
        
        for result in results:
            trend = "📈" if result["change"] > 5 else "📉" if result["change"] < -5 else "➡️"
            print(f"{result['time']:>6} | {result['current']:>8.0f} | {result['predicted']:>8.1f} | {result['change']:>+10.1f} | {trend:>8}")
            
    except Exception as e:
        print(f"❌ เกิดข้อผิดพลาด: {str(e)}")

def load_and_analyze_dataset():
    """
    วิเคราะห์ข้อมูลจริงจาก dataset
    """
    
    print("\n📈 วิเคราะห์ข้อมูลจาก dataset จริง")
    print("=" * 50)
    
    try:
        # โหลดข้อมูล
        df = pd.read_csv('traffic_dataset1.csv')
        df['timestamp'] = pd.to_datetime(df['timestamp'], format='%d/%m/%Y %H:%M')
        
        # วิเคราะห์เบื้องต้น
        print(f"📊 ข้อมูลทั้งหมด: {len(df):,} จุด")
        print(f"⏰ ช่วงเวลา: {df['timestamp'].min()} ถึง {df['timestamp'].max()}")
        print(f"🚗 จำนวนรถเฉลี่ย: {df['vehicle_count'].mean():.1f} คัน")
        print(f"📈 จำนวนรถสูงสุด: {df['vehicle_count'].max()} คัน")
        print(f"📉 จำนวนรถต่ำสุด: {df['vehicle_count'].min()} คัน")
        
        # แนวโน้มตามช่วงเวลา
        df['hour'] = df['timestamp'].dt.hour
        hourly_avg = df.groupby('hour')['vehicle_count'].mean()
        
        print(f"\n⏰ ช่วงเวลาที่มีการจราจรมากที่สุด:")
        top_hours = hourly_avg.nlargest(3)
        for hour, count in top_hours.items():
            print(f"   🕒 {hour:02d}:00 - เฉลี่ย {count:.1f} คัน")
            
        print(f"\n⏰ ช่วงเวลาที่มีการจราจรน้อยที่สุด:")
        low_hours = hourly_avg.nsmallest(3)
        for hour, count in low_hours.items():
            print(f"   🕒 {hour:02d}:00 - เฉลี่ย {count:.1f} คัน")
            
        # ทดสอบพยากรณ์กับข้อมูลจริง
        from model_deployment_guide import create_prediction_pipeline
        predict_func = create_prediction_pipeline()
        
        # เลือกข้อมูลตัวอย่าง 5 จุดสุดท้าย
        sample_data = df.tail(30)  # เอา 30 จุดสุดท้าย
        
        print(f"\n🔮 ทดสอบพยากรณ์กับข้อมูลจริง (5 จุดสุดท้าย):")
        
        for i in range(5):
            idx = -(5-i)  # นับจากท้าย
            row = sample_data.iloc[idx]
            historical = sample_data['vehicle_count'].iloc[max(0, idx-24):idx].tolist()
            
            if len(historical) >= 5:  # ต้องมีข้อมูลพอ
                result = predict_func(
                    row['timestamp'], 
                    row['vehicle_count'], 
                    historical
                )
                
                if 'error' not in result:
                    actual = row['target_next_1h'] if 'target_next_1h' in row else "N/A"
                    predicted = result['ensemble_prediction']
                    
                    print(f"   📅 {row['timestamp'].strftime('%m/%d %H:%M')}: "
                          f"จริง={row['vehicle_count']:.0f}, "
                          f"พยากรณ์={predicted:.1f}"
                          f"{f', เป้าหมาย={actual:.0f}' if actual != 'N/A' else ''}")
                    
    except FileNotFoundError:
        print("❌ ไม่พบไฟล์ traffic_dataset1.csv")
        print("💡 กรุณาตรวจสอบว่าไฟล์อยู่ในโฟลเดอร์เดียวกัน")
    except Exception as e:
        print(f"❌ เกิดข้อผิดพลาด: {str(e)}")

if __name__ == "__main__":
    
    print("🚗 Traffic Forecasting - Usage Examples")
    print("การใช้งานโมเดลพยากรณ์การจราจร")
    print("=" * 60)
    
    # ตัวอย่างที่ 1: การพยากรณ์อย่างง่าย
    simple_traffic_prediction()
    
    # ตัวอย่างที่ 2: การพยากรณ์หลายจุดเวลา
    batch_prediction_example()
    
    # ตัวอย่างที่ 3: วิเคราะห์ข้อมูลจริง
    load_and_analyze_dataset()
    
    print(f"\n🎯 สรุปวิธีการใช้งาน:")
    print("1. 🔧 สำหรับการใช้งานโดยตรง: รันไฟล์นี้")
    print("2. 🌐 สำหรับ API: รัน python traffic_api.py")
    print("3. 🔬 สำหรับการพัฒนา: ใช้ model_deployment_guide.py")
    print("4. 💾 บันทึกโมเดล: ใช้ save_models.py")
    
    print(f"\n📚 ไฟล์ที่เกี่ยวข้อง:")
    import os
    for file in ['traffic_dataset1.csv', 'traffic_forecasting_comparison.ipynb', 
                 'model_deployment_guide.py', 'traffic_api.py', 'save_models.py']:
        status = "✅" if os.path.exists(file) else "❌"
        print(f"   {status} {file}")
        
    if os.path.exists('./models/'):
        print(f"\n📁 โมเดลที่บันทึกไว้:")
        for file in os.listdir('./models/'):
            print(f"   ✅ {file}")
    else:
        print(f"\n⚠️  ยังไม่มีโมเดลที่บันทึกไว้")
        print("💡 กรุณาฝึกโมเดลในโน้ตบุ๊กก่อน")
