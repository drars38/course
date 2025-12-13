"""
Вкладка 5: Корреляции
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from utils import compute_correlation_matrix


def render_correlations_tab(df, numeric_cols, categorical_cols):
    """Отображает вкладку анализа корреляций"""
    # Устанавливаем флаг активной вкладки для изоляции
    st.session_state.current_active_tab = 4
    
    # Обновляем статус прогресс-бара
    if 'status_text' in st.session_state:
        st.session_state.status_text.text("🔗 Обработка вкладки: Корреляции")
    
    st.header("5. Анализ корреляций и взаимосвязей")
    
    if len(numeric_cols) > 1:
        # Корреляционная матрица (используем кэшированную функцию)
        st.subheader("5.1. Корреляционная матрица")
        with st.spinner("Вычисление корреляций..."):
            correlation_matrix = compute_correlation_matrix(df, numeric_cols)
        
        if correlation_matrix is not None:
            with st.spinner("Построение тепловой карты..."):
                fig, ax = plt.subplots(figsize=(10, 8))  # Уменьшаем размер
                sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                           center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.8}, 
                           ax=ax, annot_kws={'size': 8})  # Уменьшаем размер аннотаций
                ax.set_title('Корреляционная матрица числовых признаков', fontsize=12, fontweight='bold')
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
        
        # Сильные корреляции
        st.subheader("Сильные корреляции (|r| > 0.5)")
        strong_corrs = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr_val = correlation_matrix.iloc[i, j]
                if abs(corr_val) > 0.5:
                    strong_corrs.append({
                        'Признак 1': correlation_matrix.columns[i],
                        'Признак 2': correlation_matrix.columns[j],
                        'Корреляция': f"{corr_val:.3f}"
                    })
        
        if strong_corrs:
            st.dataframe(pd.DataFrame(strong_corrs), use_container_width=True)
        else:
            st.info("Сильных корреляций (|r| > 0.5) не обнаружено")
        
        # Анализ мультиколлинеарности (VIF)
        st.subheader("5.2. Анализ мультиколлинеарности (VIF)")
        st.markdown("""
        **VIF (Variance Inflation Factor)** — показатель мультиколлинеарности. 
        - VIF < 5: слабая мультиколлинеарность
        - 5 ≤ VIF < 10: умеренная мультиколлинеарность
        - VIF ≥ 10: сильная мультиколлинеарность (требует внимания)
        """)
        
        if len(numeric_cols) >= 2:
            with st.spinner("Вычисление VIF..."):
                try:
                    # Подготавливаем данные для VIF (убираем пропуски)
                    df_vif = df[numeric_cols].dropna()
                    
                    if len(df_vif) > len(numeric_cols):
                        # Добавляем константу для регрессии
                        X = add_constant(df_vif)
                        
                        # Вычисляем VIF для каждого признака
                        vif_data = []
                        for i, col in enumerate(numeric_cols):
                            try:
                                vif = variance_inflation_factor(X.values, i + 1)  # +1 из-за константы
                                vif_data.append({
                                    'Признак': col,
                                    'VIF': f"{vif:.2f}",
                                    'Оценка': 'Сильная' if vif >= 10 else ('Умеренная' if vif >= 5 else 'Слабая')
                                })
                            except:
                                vif_data.append({
                                    'Признак': col,
                                    'VIF': 'N/A',
                                    'Оценка': 'Ошибка вычисления'
                                })
                        
                        vif_df = pd.DataFrame(vif_data)
                        st.dataframe(vif_df, use_container_width=True)
                        
                        # Визуализация VIF
                        fig, ax = plt.subplots(figsize=(10, 6))
                        vif_values = [float(v['VIF']) if v['VIF'] != 'N/A' else 0 for v in vif_data]
                        colors = ['red' if v >= 10 else ('orange' if v >= 5 else 'green') for v in vif_values]
                        
                        bars = ax.barh(vif_df['Признак'], vif_values, color=colors, alpha=0.7)
                        ax.axvline(x=5, color='orange', linestyle='--', label='Порог умеренной мультиколлинеарности (VIF=5)')
                        ax.axvline(x=10, color='red', linestyle='--', label='Порог сильной мультиколлинеарности (VIF=10)')
                        ax.set_xlabel('VIF (Variance Inflation Factor)', fontsize=10)
                        ax.set_title('Анализ мультиколлинеарности (VIF)', fontsize=12, fontweight='bold')
                        ax.legend(fontsize=8)
                        ax.grid(alpha=0.3, axis='x')
                        plt.tight_layout()
                        st.pyplot(fig, use_container_width=True)
                        plt.close(fig)
                        
                        # Предупреждения
                        high_vif = [v for v in vif_data if v['VIF'] != 'N/A' and float(v['VIF']) >= 10]
                        if high_vif:
                            st.warning(f"⚠️ Обнаружена сильная мультиколлинеарность у признаков: {', '.join([v['Признак'] for v in high_vif])}")
                        elif any(v['VIF'] != 'N/A' and 5 <= float(v['VIF']) < 10 for v in vif_data):
                            st.info("ℹ️ Обнаружена умеренная мультиколлинеарность. Рекомендуется проверить признаки с VIF ≥ 5.")
                    else:
                        st.warning("Недостаточно данных для вычисления VIF (слишком много пропусков)")
                except Exception as e:
                    st.error(f"Ошибка при вычислении VIF: {str(e)}")
        else:
            st.info("Для анализа VIF требуется минимум 2 числовых признака")
        
        # Сравнение распределений по категориям
        if categorical_cols:
            st.subheader("5.3. Сравнение распределений по категориям")
            
            group_col = st.selectbox("Выберите категориальный признак для группировки", categorical_cols, key="group")
            num_col = st.selectbox("Выберите числовой признак", numeric_cols, key="num_group")
            
            if group_col and num_col:
                # Ограничиваем количество групп
                top_groups = df[group_col].value_counts().head(10).index
                df_filtered = df[df[group_col].isin(top_groups)]
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Гистограммы по группам
                    with st.spinner("Построение гистограмм..."):
                        fig, ax = plt.subplots(figsize=(8, 5))  # Уменьшаем размер
                        for group_val in top_groups[:5]:  # Показываем топ-5
                            subset = df_filtered[df_filtered[group_col] == group_val][num_col]
                            ax.hist(subset.dropna(), alpha=0.6, label=f'{group_val}', bins=15)  # Уменьшаем bins
                        ax.set_title(f'Распределение {num_col} по {group_col}', fontsize=10, fontweight='bold')
                        ax.set_xlabel(num_col, fontsize=9)
                        ax.set_ylabel('Частота', fontsize=9)
                        ax.legend(fontsize=8)
                        ax.grid(alpha=0.3)
                        plt.tight_layout()
                        st.pyplot(fig, use_container_width=True)
                        plt.close(fig)
                
                with col2:
                    # Boxplot по группам
                    with st.spinner("Построение boxplot..."):
                        fig, ax = plt.subplots(figsize=(8, 5))  # Уменьшаем размер
                        sns.boxplot(x=group_col, y=num_col, data=df_filtered, ax=ax)
                        ax.set_title(f'Boxplot {num_col} по {group_col}', fontsize=10, fontweight='bold')
                        ax.tick_params(axis='x', rotation=45, labelsize=8)
                        ax.grid(alpha=0.3, axis='y')
                        plt.tight_layout()
                        st.pyplot(fig, use_container_width=True)
                        plt.close(fig)
                
                # Статистика по группам
                grouped_stats = df_filtered.groupby(group_col)[num_col].agg(['mean', 'median', 'std', 'count'])
                st.dataframe(grouped_stats, use_container_width=True)
    else:
        st.warning("Недостаточно числовых признаков для корреляционного анализа")
