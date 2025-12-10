"""
Вкладка 3: Распределения
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats as scipy_stats
from scipy.stats import gaussian_kde


def render_distributions_tab(df, numeric_cols, categorical_cols):
    """Отображает вкладку анализа распределений"""
    # Устанавливаем флаг активной вкладки для изоляции
    st.session_state.current_active_tab = 2
    
    # Обновляем статус прогресс-бара
    if 'status_text' in st.session_state:
        st.session_state.status_text.text("📈 Обработка вкладки: Распределения")
    
    st.header("3. Анализ распределений")
    
    if numeric_cols:
        st.subheader("3.1. Распределения числовых признаков")
        
        # Выбор признака для детального анализа
        selected_num_col = st.selectbox("Выберите числовой признак для анализа", numeric_cols, key="dist_col")
        
        # Опция для отключения дополнительных графиков
        show_advanced = st.checkbox("Показать дополнительные графики (Q-Q plot, CDF)", value=True, key="show_advanced_dist")
        
        if selected_num_col:
            # Множественные графики
            col1, col2 = st.columns(2)
            
            with col1:
                # Гистограмма с KDE
                with st.spinner("Построение гистограммы..."):
                    fig, ax = plt.subplots(figsize=(8, 5))  # Уменьшаем размер
                    data = df[selected_num_col].dropna()
                    ax.hist(data, bins=25, color='skyblue', edgecolor='black', 
                           alpha=0.7, density=True, label='Гистограмма')  # Уменьшаем bins
                    # KDE кривая (только для небольших датасетов)
                    try:
                        if len(data) > 1 and len(data) < 10000:  # KDE только для небольших датасетов
                            kde = gaussian_kde(data)
                            x_range = np.linspace(data.min(), data.max(), 100)  # Уменьшаем точки
                            ax.plot(x_range, kde(x_range), 'r-', linewidth=1.5, label='KDE')
                    except:
                        pass
                    mean_val = data.mean()
                    median_val = data.median()
                    ax.axvline(mean_val, color='red', linestyle='--', linewidth=1.5, label=f'Среднее: {mean_val:.2f}')
                    ax.axvline(median_val, color='green', linestyle='--', linewidth=1.5, label=f'Медиана: {median_val:.2f}')
                    ax.set_title(f'Распределение {selected_num_col}', fontsize=11, fontweight='bold')
                    ax.set_xlabel(selected_num_col, fontsize=9)
                    ax.set_ylabel('Плотность', fontsize=9)
                    ax.legend(fontsize=8)
                    ax.grid(alpha=0.3)
                    plt.tight_layout()
                    st.pyplot(fig, use_container_width=True)
                    plt.close(fig)
            
            with col2:
                # Boxplot и Violin plot вместе
                with st.spinner("Построение boxplot и violin plot..."):
                    fig, axes = plt.subplots(2, 1, figsize=(8, 6))  # Уменьшаем размер
                    
                    # Boxplot
                    sns.boxplot(y=df[selected_num_col], ax=axes[0], color='lightblue')
                    axes[0].set_title(f'Boxplot для {selected_num_col}', fontsize=10, fontweight='bold')
                    axes[0].set_ylabel('Значение', fontsize=9)
                    axes[0].grid(alpha=0.3, axis='y')
                    
                    # Violin plot (может быть медленным, делаем опциональным)
                    if len(df) < 5000:  # Violin plot только для небольших датасетов
                        sns.violinplot(y=df[selected_num_col], ax=axes[1], color='lightcoral')
                        axes[1].set_title(f'Violin plot для {selected_num_col}', fontsize=10, fontweight='bold')
                    else:
                        # Для больших датасетов показываем только гистограмму
                        axes[1].hist(df[selected_num_col].dropna(), bins=20, color='lightcoral', alpha=0.7, edgecolor='black')
                        axes[1].set_title(f'Гистограмма {selected_num_col}', fontsize=10, fontweight='bold')
                    axes[1].set_ylabel('Значение', fontsize=9)
                    axes[1].grid(alpha=0.3, axis='y')
                    
                    plt.tight_layout()
                    st.pyplot(fig, use_container_width=True)
                    plt.close(fig)
            
            # Дополнительные графики (опционально)
            if show_advanced:
                col3, col4 = st.columns(2)
                
                with col3:
                    # Q-Q plot для проверки нормальности
                    with st.spinner("Построение Q-Q plot..."):
                        fig, ax = plt.subplots(figsize=(8, 5))  # Уменьшаем размер
                        sample = df[selected_num_col].dropna()
                        if len(sample) > 0:
                            # Используем выборку для больших датасетов
                            if len(sample) > 5000:
                                sample = sample.sample(n=5000, random_state=42)
                            scipy_stats.probplot(sample, dist="norm", plot=ax)
                            ax.set_title(f'Q-Q plot (проверка нормальности)', fontsize=10, fontweight='bold')
                            ax.grid(alpha=0.3)
                            plt.tight_layout()
                            st.pyplot(fig, use_container_width=True)
                            plt.close(fig)
                
                with col4:
                    # Cumulative Distribution Function
                    with st.spinner("Построение CDF..."):
                        fig, ax = plt.subplots(figsize=(8, 5))  # Уменьшаем размер
                        data = df[selected_num_col].dropna()
                        # Используем выборку для больших датасетов
                        if len(data) > 5000:
                            data = data.sample(n=5000, random_state=42).sort_values()
                        sorted_data = np.sort(data)
                        y_vals = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
                        ax.plot(sorted_data, y_vals, linewidth=1.5, color='purple')  # Уменьшаем толщину линии
                        ax.set_xlabel(selected_num_col, fontsize=9)
                        ax.set_ylabel('Кумулятивная вероятность', fontsize=9)
                        ax.set_title('Кумулятивная функция распределения (CDF)', fontsize=10, fontweight='bold')
                        ax.grid(alpha=0.3)
                        plt.tight_layout()
                        st.pyplot(fig, use_container_width=True)
                        plt.close(fig)
            
            # Статистика
            col_stat1, col_stat2 = st.columns(2)
            with col_stat1:
                st.write("**Основные статистики:**")
                stats_dict = {
                    'Среднее': f"{df[selected_num_col].mean():.2f}",
                    'Медиана': f"{df[selected_num_col].median():.2f}",
                    'Стд. отклонение': f"{df[selected_num_col].std():.2f}",
                    'Минимум': f"{df[selected_num_col].min():.2f}",
                    'Максимум': f"{df[selected_num_col].max():.2f}",
                }
                st.json(stats_dict)
            
            with col_stat2:
                st.write("**Дополнительные метрики:**")
                stats_dict2 = {
                    '25-й перцентиль': f"{df[selected_num_col].quantile(0.25):.2f}",
                    '75-й перцентиль': f"{df[selected_num_col].quantile(0.75):.2f}",
                    'Асимметрия': f"{df[selected_num_col].skew():.2f}",
                    'Эксцесс': f"{df[selected_num_col].kurtosis():.2f}",
                    'Коэффициент вариации': f"{(df[selected_num_col].std() / df[selected_num_col].mean() * 100):.2f}%"
                }
                st.json(stats_dict2)
    
    if categorical_cols:
        st.subheader("3.2. Распределения категориальных признаков")
        
        selected_cat_col = st.selectbox("Выберите категориальный признак", categorical_cols)
        
        if selected_cat_col:
            value_counts = df[selected_cat_col].value_counts()
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Countplot
                fig, ax = plt.subplots(figsize=(10, max(6, len(value_counts) * 0.4)))
                if len(value_counts) > 20:
                    top_20 = value_counts.head(20)
                    sns.barplot(x=top_20.values, y=top_20.index, ax=ax, palette='husl')
                    ax.set_title(f'Топ-20 значений для {selected_cat_col}', fontsize=12, fontweight='bold')
                else:
                    sns.barplot(x=value_counts.values, y=value_counts.index, ax=ax, palette='husl')
                    ax.set_title(f'Распределение {selected_cat_col}', fontsize=12, fontweight='bold')
                ax.set_xlabel('Количество', fontsize=10)
                ax.set_ylabel(selected_cat_col, fontsize=10)
                ax.grid(axis='x', alpha=0.3)
                st.pyplot(fig)
                plt.close(fig)
            
            with col2:
                # Круговая диаграмма (для небольшого числа категорий)
                if len(value_counts) <= 10:
                    fig, ax = plt.subplots(figsize=(8, 8))
                    ax.pie(value_counts.values, labels=value_counts.index, autopct='%1.1f%%', startangle=90)
                    ax.set_title(f'Распределение {selected_cat_col}', fontsize=12, fontweight='bold')
                    st.pyplot(fig)
                    plt.close(fig)
                else:
                    st.write("**Частоты значений:**")
                    st.dataframe(pd.DataFrame({
                        'Значение': value_counts.index,
                        'Количество': value_counts.values,
                        'Процент': (value_counts.values / len(df) * 100).round(2)
                    }), use_container_width=True)
