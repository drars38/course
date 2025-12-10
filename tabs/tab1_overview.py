"""
Вкладка 1: Обзор данных
"""
import streamlit as st
import pandas as pd
import numpy as np


def render_overview_tab(df, numeric_cols, categorical_cols):
    """Отображает вкладку обзора данных"""
    # Устанавливаем флаг активной вкладки для изоляции
    st.session_state.current_active_tab = 0
    
    # Обновляем статус прогресс-бара
    if 'status_text' in st.session_state:
        st.session_state.status_text.text("📋 Обработка вкладки: Обзор данных")
    
    st.header("1. Обзор структуры данных")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Число строк", df.shape[0])
    with col2:
        st.metric("Число столбцов", df.shape[1])
    with col3:
        st.metric("Общее значений", df.size)
    with col4:
        st.metric("Пропусков", df.isnull().sum().sum())
    
    st.subheader("Информация о данных")
    st.dataframe(pd.DataFrame({
        'Тип данных': df.dtypes.astype(str),
        'Пропущено': df.isnull().sum(),
        'Процент пропусков': (df.isnull().sum() / len(df) * 100).round(2),
        'Уникальных значений': [df[col].nunique() for col in df.columns]
    }), use_container_width=True)
    
    st.subheader("Первые строки")
    st.dataframe(df.head(10), use_container_width=True)
    
    st.subheader("Последние строки")
    st.dataframe(df.tail(10), use_container_width=True)
    
    if numeric_cols:
        st.subheader("Базовая статистика (числовые признаки)")
        st.dataframe(df[numeric_cols].describe(), use_container_width=True)
    
    if categorical_cols:
        st.subheader("Уникальные значения (категориальные признаки)")
        for col in categorical_cols[:5]:  # Показываем первые 5
            st.write(f"**{col}**: {df[col].nunique()} уникальных значений")
            st.write(df[col].value_counts().head(10))
