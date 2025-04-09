# pipeline_laptop_price.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# 1. Загрузка данных
def load_data(filepath):
    df = pd.read_csv(filepath)
    return df

# 2. Создание пайплайна
def create_pipeline():
    # Определяем числовые и категориальные признаки
    numeric_features = ['Processor_Speed', 'RAM_Size', 'Storage_Capacity', 
                       'Screen_Size', 'Weight']
    categorical_features = ['Brand']
    
    # Создаем трансформеры для предобработки
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore'))
    ])
    
    # Объединяем трансформеры в ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    # Создаем финальный пайплайн с моделью
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
    ])
    
    return pipeline

# 3. Основная функция для обучения и оценки
def train_and_evaluate(filepath):
    # Загрузка данных
    df = load_data(filepath)
    
    # Разделение на признаки и целевую переменную
    X = df.drop('Price', axis=1)
    y = df['Price']
    
    # Разделение на тренировочную и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Создание и обучение пайплайна
    pipeline = create_pipeline()
    pipeline.fit(X_train, y_train)
    
    # Предсказание
    y_pred = pipeline.predict(X_test)
    
    # Оценка модели
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f'Mean Squared Error: {mse:.2f}')
    print(f'R2 Score: {r2:.2f}')
    
    # Сохранение модели
    joblib.dump(pipeline, 'laptop_price_model.pkl')
    
    return pipeline

# 4. Функция для предсказания на новых данных
def predict_price(pipeline, new_data):
    predictions = pipeline.predict(new_data)
    return predictions

if __name__ == "__main__":
    # Запуск обучения
    filepath = "Laptop_price.csv"
    trained_pipeline = train_and_evaluate(filepath)
    
    # Пример предсказания для новых данных
    sample_data = pd.DataFrame({
        'Brand': ['Asus'],
        'Processor_Speed': [3.5],
        'RAM_Size': [16],
        'Storage_Capacity': [512],
        'Screen_Size': [13.5],
        'Weight': [2.5]
    })
    
    prediction = predict_price(trained_pipeline, sample_data)
    print(f"Predicted Price: ${prediction[0]:.2f}")