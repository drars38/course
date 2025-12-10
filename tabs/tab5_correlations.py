"""
Вкладка 5: Корреляции
"""
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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
        
        # Сравнение распределений по категориям
        if categorical_cols:
            st.subheader("5.2. Сравнение распределений по категориям")
            
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
