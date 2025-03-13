#kütüphanelerin import edilmesi
from binance.client import Client
from dotenv import load_dotenv
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

#.env verilerini yüklüyoruz
load_dotenv()

#API anahtarlarını alıyoruz
api_key = os.getenv("API_KEY")
secret_key = os.getenv("SECRET_KEY")

#binance client'ı başlatıyoruz

client = Client(api_key,secret_key)

#para birimi/coin'i ve çekmek istediğimiz verinin zaman aralığını belirliyoruz

symbol = "USDTTRY"
interval = client.KLINE_INTERVAL_4HOUR
start_time = "1 month ago UTC"


#veriyi çekiyoruz

klines = client.get_historical_klines(symbol,interval,start_time)

#veriyi pandas dataframe'ine dönüştürüyoruz

df = pd.DataFrame(klines, columns=["Open Time", "Open", "High", "Low", "Close", "Volume",
    "Close Time", "Quote Asset Volume", "Number of Trades",
    "Taker Buy Base Asset Volume", "Taker Buy Quote Asset Volume", "Ignore"])

#veri kontrolü
print(df)


#zaman damgalarını tarihe çeviriyoruz

df["Open Time"] = pd.to_datetime(df["Open Time"], unit="ms", utc=True).dt.tz_convert("Europe/Istanbul")


#Model eğitimi için bağımlı ve bağımsız değişkenlerin belirlenmesi

features = ["Open", "High", "Low", "Volume"]

X = df[features].values.astype(float)
Y = df["Close"].values.astype(float)


#PolynomialFeatures ile polinom oluşturuyoruz

poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

#Polinomiyal regresyon modeli eğitimi

model = LinearRegression()
model.fit(X_poly,Y)


# Kısa vadeli tahmin (örneğin, 4 saatlik) için son günün bağımsız değişkenlerinin değerlerini kopyalıyoruz
# bağımlı değişkeni tahminleyebilmek için
prediction_periods = 1  # 4 saatlik tahmin
last_features = np.mean(X[-48:,:], axis=0)  # Son günün özelliklerini alıyoruz
future_features = np.tile(last_features, (prediction_periods, 1))
future_features_poly = poly.transform(future_features)
future_predictions = model.predict(future_features_poly)


# Sonuçları görselleştiriyoruz
days = df["Open Time"]
future_times = pd.date_range(start=df["Open Time"].iloc[-1] + pd.Timedelta(hours=4), periods=prediction_periods, freq='4h', tz="Europe/Istanbul")

plt.figure(figsize=(18, 12))
plt.scatter(days, Y, color='blue', label='Gerçek Kapanış Fiyatları')
plt.plot(days, model.predict(X_poly), color='red', label='Polinom Regresyon (Eğitim)')
plt.plot(future_times, future_predictions, color='green', label=f'Tahmin ({prediction_periods * 4} saat)')
plt.xlabel('Tarih/Saat')
plt.ylabel('Kapanış Fiyatı (USDT)')
plt.title(f'Multiple Polynomial Regression ile USDT/TRY Fiyat Tahmini ({prediction_periods * 4} saat)')
plt.legend()
plt.xticks(rotation=45)
plt.show()

# Tahmin sonuçlarını yazdırıyoruz
print(f"Sonraki {prediction_periods * 4} saat için tahminler:")
for i, (time, price) in enumerate(zip(future_times, future_predictions), start=1):
    print(f"Zaman {time}: {price:.10f} TRY")
