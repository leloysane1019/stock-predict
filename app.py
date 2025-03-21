from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import yfinance as yf
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os
import uvicorn

app = FastAPI()

# Jinja2の設定（テンプレートフォルダを指定）
templates = Jinja2Templates(directory="templates")

# staticフォルダをマウント（favicon.icoのエラー防止）
app.mount("/static", StaticFiles(directory="static"), name="static")

# 学習済みモデルの読み込み
model_path = os.path.join(os.path.dirname(__file__), "stock_prediction_model_ja.keras")
import os



model_path = os.path.join(os.path.dirname(__file__), "stock_prediction_model_ja.keras")

model = tf.keras.models.load_model(model_path)

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

from datetime import datetime

@app.post("/predict/", response_class=HTMLResponse)
async def predict(request: Request, code: str = Form(...)):
    try:
        # 今日の日付を取得
        today = datetime.today().strftime('%Y-%m-%d')

        # 株価データの取得（最新データまで）
        data = yf.download(code, start='2010-01-01', end=today)

        # データが取得できなかった場合のエラーハンドリング
        if data.empty:
            return templates.TemplateResponse("index.html", {"request": request, "error": "株価データが取得できませんでした"})

        data = data[['Close']]

        # データの前処理
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data)

        # 直近60日間のデータを用いて予測
        X_test = []
        X_test.append(scaled_data[-60:, 0])
        X_test = np.array(X_test)
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

        # 予測
        predicted_price = model.predict(X_test)

        # 予測結果を逆正規化
        predicted_price = scaler.inverse_transform(predicted_price)

        return templates.TemplateResponse("index.html", {"request": request, "predicted_price": predicted_price[0][0]})

    except Exception as e:
        return templates.TemplateResponse("index.html", {"request": request, "error": f"エラーが発生しました: {str(e)}"})