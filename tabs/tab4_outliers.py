"""
Вкладка 4: Выбросы
"""
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils import sample_data_for_plotting


def render_outliers_tab(df, numeric_cols, max_plot_points, use_sampling):
    """Отображает вкладку анализа выбросов"""
    # Устанавливаем флаг активной вкладки для изоляции
    st.session_state.current_active_tab = 3
    
    # Обновляем статус прогресс-бара
    if 'status_text' in st.session_state:
        st.session_state.status_text.text("🔍 Обработка вкладки: Выбросы")
    
    st.header("4. Выявление выбросов")
    
    if numeric_cols:
        selected_outlier_col = st.selectbox("Выберите признак для анализа выбросов", numeric_cols, key="outlier")
        
        if selected_outlier_col:
            from utils import compute_outliers
            
            # Используем кэшированную функцию
            Q1, Q3, IQR, lower_bound, upper_bound, outliers = compute_outliers(df, selected_outlier_col)
            outliers_count = len(outliers)
            outliers_percent = (outliers_count / len(df)) * 100
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Q1", f"{Q1:.2f}")
            with col2:
                st.metric("Q3", f"{Q3:.2f}")
            with col3:
                st.metric("IQR", f"{IQR:.2f}")
            with col4:
                st.metric("Выбросов", f"{outliers_count} ({outliers_percent:.2f}%)")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Boxplot с выбросами
                with st.spinner("Построение boxplot..."):
                    fig, ax = plt.subplots(figsize=(8, 5))  # Уменьшаем размер
                    sns.boxplot(y=df[selected_outlier_col], ax=ax, color='lightblue')
                    ax.axhline(lower_bound, color='red', linestyle='--', alpha=0.5, label=f'Нижняя: {lower_bound:.2f}')
                    ax.axhline(upper_bound, color='red', linestyle='--', alpha=0.5, label=f'Верхняя: {upper_bound:.2f}')
                    ax.set_title(f'Выбросы в {selected_outlier_col}', fontsize=10, fontweight='bold')
                    ax.set_ylabel('Значение', fontsize=9)
                    ax.legend(fontsize=8)
                    ax.grid(alpha=0.3, axis='y')
                    plt.tight_layout()
                    st.pyplot(fig, use_container_width=True)
                    plt.close(fig)
            
            with col2:
                # Scatterplot (если есть другой числовой признак)
                if len(numeric_cols) > 1:
                    other_col = st.selectbox("Выберите второй признак", 
                                             [c for c in numeric_cols if c != selected_outlier_col],
                                             key="scatter")
                    
                    # Выбор режима отображения
                    display_mode = st.radio(
                        "Что отображать на графике:",
                        ["Все вместе", "Только нормальные значения", "Только выбросы"],
                        key="scatter_display_mode",
                        horizontal=True
                    )
                    
                    with st.spinner("Построение scatter plot..."):
                        fig, ax = plt.subplots(figsize=(8, 5))  # Уменьшаем размер
                        
                        # Определяем выбросы для выбранного признака
                        is_outlier = (df[selected_outlier_col] < lower_bound) | (df[selected_outlier_col] > upper_bound)
                        
                        # Разделяем данные на нормальные и выбросы
                        normal_data = df[~is_outlier]
                        outlier_data = df[is_outlier]
                        
                        # Отображаем данные в зависимости от выбранного режима
                        if display_mode in ["Все вместе", "Только нормальные значения"]:
                            if len(normal_data) > 0:
                                normal_plot = sample_data_for_plotting(normal_data[[other_col, selected_outlier_col]], 
                                                                      max_plot_points, use_sampling)
                                ax.scatter(normal_plot[other_col], normal_plot[selected_outlier_col], 
                                          color='blue', alpha=0.5, s=15, label=f'Нормальные ({len(normal_data)})')  # Уменьшаем размер точек
                        
                        if display_mode in ["Все вместе", "Только выбросы"]:
                            # Показываем ВСЕ выбросы (не применяем выборку к выбросам, чтобы не потерять важную информацию)
                            if len(outlier_data) > 0:
                                ax.scatter(outlier_data[other_col], outlier_data[selected_outlier_col], 
                                          color='red', s=40, alpha=0.7, label=f'Выбросы ({len(outlier_data)})')  # Уменьшаем размер точек
                        
                        ax.set_xlabel(other_col, fontsize=9)
                        ax.set_ylabel(selected_outlier_col, fontsize=9)
                        ax.set_title(f'Scatterplot: {other_col} vs {selected_outlier_col}', 
                                    fontsize=10, fontweight='bold')
                        ax.legend(fontsize=8)
                        ax.grid(alpha=0.3)
                        plt.tight_layout()
                        st.pyplot(fig, use_container_width=True)
                        plt.close(fig)
            
            if outliers_count > 0:
                st.subheader("Обнаруженные выбросы")
                st.dataframe(outliers[[selected_outlier_col] + [c for c in df.columns if c != selected_outlier_col]], 
                            use_container_width=True)
