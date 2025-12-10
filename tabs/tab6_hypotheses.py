"""
Вкладка 6: Автоматическая генерация гипотез
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats as scipy_stats
from utils import sample_data_for_plotting


def render_hypotheses_tab(df, numeric_cols, categorical_cols, target_col, max_plot_points, use_sampling):
    """Отображает вкладку автоматической генерации гипотез"""
    # Устанавливаем флаг активной вкладки для изоляции
    st.session_state.current_active_tab = 5
    
    # Обновляем статус прогресс-бара
    if 'status_text' in st.session_state:
        st.session_state.status_text.text("🎯 Обработка вкладки: Гипотезы")
    
    st.header("6. Автоматическая генерация гипотез с визуализациями")
    
    # Используем кэшированную функцию для вычисления гипотез
    # @st.cache_data автоматически кэширует результаты
    hypotheses = _compute_hypotheses_data(df, numeric_cols, categorical_cols, target_col, max_plot_points, use_sampling)
    
    # Отображение гипотез с визуализациями
    if hypotheses:
        st.success(f"✅ Сгенерировано {len(hypotheses)} гипотез")
        st.markdown("---")
        
        for i, hyp in enumerate(hypotheses, 1):
            with st.expander(f"**Гипотеза {i}:** {hyp['Гипотеза']}", expanded=(i == 1)):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    if 'plot' in hyp and hyp['plot'] is not None:
                        st.pyplot(hyp['plot'])
                        plt.close(hyp['plot'])  # Закрываем фигуру для освобождения памяти
                
                with col2:
                    st.markdown("**📝 Обоснование:**")
                    st.write(hyp['Обоснование'])
                    st.markdown("---")
                    st.markdown("**🔬 Метод проверки:**")
                    st.write(hyp['Метод проверки'])
    else:
        st.info("💡 Гипотезы будут сгенерированы после более детального анализа данных. Убедитесь, что данные загружены корректно.")
        st.markdown("**Совет:** Проверьте, что:")
        st.markdown("- В данных есть числовые и категориальные признаки")
        st.markdown("- Данные не содержат критических ошибок")
        st.markdown("- Разделители в CSV файле корректны")


@st.cache_data(show_spinner=False)
def _compute_hypotheses_data(df, numeric_cols, categorical_cols, target_col, max_plot_points, use_sampling):
    """Кэшированная функция для вычисления гипотез (графики пересоздаются при каждом отображении)"""
    hypotheses = []
    
    # Гипотеза 1: Корреляция с целевой переменной
    if target_col and target_col in numeric_cols and len(numeric_cols) > 1:
        for col in numeric_cols:
            if col != target_col:
                try:
                    corr = df[target_col].corr(df[col])
                    if abs(corr) > 0.3:
                        # Выбираем данные для визуализации
                        plot_df = sample_data_for_plotting(df[[col, target_col]], max_plot_points, use_sampling)
                        
                        # Создаем scatter plot для визуализации
                        fig, ax = plt.subplots(figsize=(8, 5))  # Уменьшаем размер
                        ax.scatter(plot_df[col], plot_df[target_col], alpha=0.4, s=20)  # Уменьшаем размер точек
                        # Линия тренда (используем все данные для точности, но только если не слишком много)
                        if len(df) < 10000:
                            z = np.polyfit(df[col].dropna(), df[target_col].dropna(), 1)
                            p = np.poly1d(z)
                            x_line = np.linspace(df[col].min(), df[col].max(), 50)  # Уменьшаем точки
                            ax.plot(x_line, p(x_line), 
                                   "r--", alpha=0.7, linewidth=1.5, label=f'Тренд (r={corr:.3f})')
                        ax.set_xlabel(col, fontsize=10)
                        ax.set_ylabel(target_col, fontsize=10)
                        ax.set_title(f'Корреляция: {col} vs {target_col}', fontsize=11, fontweight='bold')
                        ax.legend(fontsize=8)
                        ax.grid(alpha=0.3)
                        plt.tight_layout()
                        
                        hypotheses.append({
                            'id': len(hypotheses),
                            'Гипотеза': f"Признак '{col}' имеет {'положительную' if corr > 0 else 'отрицательную'} корреляцию с '{target_col}'",
                            'Обоснование': f"Корреляция составляет {corr:.3f}, что указывает на {'прямую' if corr > 0 else 'обратную'} связь",
                            'Метод проверки': "Корреляционный анализ, регрессионное моделирование",
                            'plot': fig
                        })
                except:
                    pass
    
    # Гипотеза 2: Влияние категориальных признаков на числовые
    if categorical_cols and numeric_cols:
        for cat_col in categorical_cols[:5]:
            for num_col in numeric_cols[:5]:
                try:
                    grouped_means = df.groupby(cat_col)[num_col].mean()
                    if len(grouped_means) > 1 and grouped_means.std() > abs(grouped_means.mean()) * 0.1:
                        # Создаем boxplot для визуализации
                        fig, axes = plt.subplots(1, 2, figsize=(12, 5))  # Уменьшаем размер
                        
                        # Boxplot
                        top_groups = grouped_means.nlargest(10).index
                        df_filtered = df[df[cat_col].isin(top_groups)]
                        sns.boxplot(x=cat_col, y=num_col, data=df_filtered, ax=axes[0])
                        axes[0].set_title(f'Распределение {num_col} по {cat_col}', fontsize=10, fontweight='bold')
                        axes[0].tick_params(axis='x', rotation=45, labelsize=8)
                        axes[0].grid(alpha=0.3, axis='y')
                        
                        # Barplot средних значений
                        grouped_means_sorted = grouped_means.sort_values(ascending=False).head(10)
                        axes[1].barh(range(len(grouped_means_sorted)), grouped_means_sorted.values, color='skyblue')
                        axes[1].set_yticks(range(len(grouped_means_sorted)))
                        axes[1].set_yticklabels(grouped_means_sorted.index, fontsize=8)
                        axes[1].set_xlabel(f'Среднее значение {num_col}', fontsize=9)
                        axes[1].set_title(f'Средние значения {num_col} по группам', fontsize=10, fontweight='bold')
                        axes[1].grid(alpha=0.3, axis='x')
                        
                        plt.tight_layout()
                        
                        hypotheses.append({
                            'id': len(hypotheses),
                            'Гипотеза': f"Признак '{cat_col}' влияет на '{num_col}'",
                            'Обоснование': f"Средние значения '{num_col}' различаются по группам '{cat_col}' (разброс: {grouped_means.std():.2f})",
                            'Метод проверки': "ANOVA, t-test, визуализация boxplot",
                            'plot': fig
                        })
                except:
                    pass
    
    # Гипотеза 3: Выбросы и аномалии
    if numeric_cols:
        for col in numeric_cols[:5]:
            try:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                if IQR > 0:
                    outliers = df[(df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)]
                    outliers_count = len(outliers)
                    if outliers_count > len(df) * 0.05:  # Более 5% выбросов
                            # Создаем визуализацию выбросов
                        fig, axes = plt.subplots(1, 2, figsize=(12, 5))  # Уменьшаем размер
                        
                        # Boxplot
                        sns.boxplot(y=df[col], ax=axes[0], color='lightblue')
                        axes[0].axhline(Q1 - 1.5*IQR, color='red', linestyle='--', alpha=0.7, label='Нижняя граница')
                        axes[0].axhline(Q3 + 1.5*IQR, color='red', linestyle='--', alpha=0.7, label='Верхняя граница')
                        axes[0].set_title(f'Выбросы в {col}', fontsize=10, fontweight='bold')
                        axes[0].set_ylabel('Значение', fontsize=9)
                        axes[0].legend(fontsize=8)
                        axes[0].grid(alpha=0.3, axis='y')
                        
                        # Гистограмма с выделением выбросов
                        axes[1].hist(df[col].dropna(), bins=20, color='skyblue', alpha=0.7, edgecolor='black', label='Нормальные значения')  # Уменьшаем bins
                        if outliers_count > 0:
                            axes[1].hist(outliers[col], bins=20, color='red', alpha=0.7, edgecolor='black', label='Выбросы')
                        axes[1].set_xlabel(col, fontsize=9)
                        axes[1].set_ylabel('Частота', fontsize=9)
                        axes[1].set_title(f'Распределение с выделением выбросов', fontsize=10, fontweight='bold')
                        axes[1].legend(fontsize=8)
                        axes[1].grid(alpha=0.3)
                        
                        plt.tight_layout()
                        
                        hypotheses.append({
                            'id': len(hypotheses),
                            'Гипотеза': f"В признаке '{col}' присутствует значительное количество выбросов",
                            'Обоснование': f"Обнаружено {outliers_count} выбросов ({outliers_count/len(df)*100:.1f}% данных)",
                            'Метод проверки': "IQR метод, визуализация boxplot, анализ причин выбросов",
                            'plot': fig
                        })
            except:
                pass
    
    # Гипотеза 4: Распределения (асимметрия)
    if numeric_cols:
        for col in numeric_cols[:5]:
            try:
                skewness = df[col].skew()
                if abs(skewness) > 1:
                    # Создаем визуализацию распределения
                    fig, axes = plt.subplots(1, 2, figsize=(12, 5))  # Уменьшаем размер
                    
                    # Гистограмма
                    data = df[col].dropna()
                    axes[0].hist(data, bins=20, color='skyblue', alpha=0.7, edgecolor='black')  # Уменьшаем bins
                    mean_val = data.mean()
                    median_val = data.median()
                    axes[0].axvline(mean_val, color='red', linestyle='--', linewidth=1.5, label=f'Среднее: {mean_val:.2f}')
                    axes[0].axvline(median_val, color='green', linestyle='--', linewidth=1.5, label=f'Медиана: {median_val:.2f}')
                    axes[0].set_xlabel(col, fontsize=9)
                    axes[0].set_ylabel('Частота', fontsize=9)
                    axes[0].set_title(f'Распределение {col} (асимметрия: {skewness:.2f})', fontsize=10, fontweight='bold')
                    axes[0].legend(fontsize=8)
                    axes[0].grid(alpha=0.3)
                    
                    # Q-Q plot для проверки нормальности (только для небольших датасетов)
                    sample = data
                    if len(sample) > 0 and len(sample) < 5000:
                        if len(sample) > 2000:
                            sample = sample.sample(n=2000, random_state=42)
                        scipy_stats.probplot(sample, dist="norm", plot=axes[1])
                        axes[1].set_title(f'Q-Q plot для {col}', fontsize=10, fontweight='bold')
                    else:
                        # Для больших датасетов показываем только статистику
                        axes[1].text(0.5, 0.5, f'Асимметрия: {skewness:.2f}\nЭксцесс: {data.kurtosis():.2f}', 
                                    ha='center', va='center', fontsize=12, transform=axes[1].transAxes)
                        axes[1].set_title(f'Статистика распределения', fontsize=10, fontweight='bold')
                    axes[1].grid(alpha=0.3)
                    
                    plt.tight_layout()
                    
                    hypotheses.append({
                        'id': len(hypotheses),
                        'Гипотеза': f"Признак '{col}' имеет {'правостороннее' if skewness > 0 else 'левостороннее'} асимметричное распределение",
                        'Обоснование': f"Коэффициент асимметрии: {skewness:.2f} ({'сильная асимметрия' if abs(skewness) > 2 else 'умеренная асимметрия'})",
                        'Метод проверки': "Визуализация гистограммы, применение логарифмического преобразования",
                        'plot': fig
                    })
            except:
                pass
    
    # Гипотеза 5: Пропущенные значения
    missing_cols = [col for col in df.columns if df[col].isnull().sum() > 0]
    if missing_cols:
        for col in missing_cols[:3]:
            missing_pct = df[col].isnull().sum() / len(df) * 100
            if missing_pct > 10:
                # Создаем визуализацию пропусков
                fig, axes = plt.subplots(1, 2, figsize=(12, 5))  # Уменьшаем размер
                
                # Тепловая карта пропусков для этого признака (только для небольших датасетов)
                if len(df) < 5000:
                    missing_data = df[[col]].isnull()
                    sns.heatmap(missing_data, yticklabels=False, cbar=True, cmap='viridis', ax=axes[0])
                    axes[0].set_title(f'Паттерн пропусков в {col}', fontsize=10, fontweight='bold')
                else:
                    # Для больших датасетов показываем только статистику
                    axes[0].text(0.5, 0.5, f'Пропущено: {missing_pct:.1f}%', 
                                ha='center', va='center', fontsize=14, transform=axes[0].transAxes)
                    axes[0].set_title(f'Пропуски в {col}', fontsize=10, fontweight='bold')
                
                # Сравнение распределений: с пропусками vs без пропусков
                if col in numeric_cols:
                    not_missing = df[df[col].notna()][col]
                    axes[1].hist(not_missing, bins=15, alpha=0.7, color='green', label='Не пропущено', edgecolor='black')  # Уменьшаем bins
                    axes[1].set_xlabel(col, fontsize=9)
                    axes[1].set_ylabel('Частота', fontsize=9)
                    axes[1].set_title(f'Распределение (пропущено {missing_pct:.1f}%)', fontsize=10, fontweight='bold')
                    axes[1].legend(fontsize=8)
                    axes[1].grid(alpha=0.3)
                else:
                    value_counts = df[col].value_counts().head(10)
                    axes[1].barh(range(len(value_counts)), value_counts.values, color='coral')
                    axes[1].set_yticks(range(len(value_counts)))
                    axes[1].set_yticklabels(value_counts.index, fontsize=8)
                    axes[1].set_xlabel('Количество', fontsize=9)
                    axes[1].set_title(f'Распределение значений', fontsize=10, fontweight='bold')
                    axes[1].grid(alpha=0.3, axis='x')
                
                plt.tight_layout()
                
                hypotheses.append({
                    'id': len(hypotheses),
                    'Гипотеза': f"Пропущенные значения в '{col}' могут быть информативными",
                    'Обоснование': f"Пропущено {missing_pct:.1f}% значений, что может указывать на систематический паттерн",
                    'Метод проверки': "Анализ паттернов пропусков, создание бинарного признака 'есть/нет пропуск'",
                    'plot': fig
                })
    
    # Гипотеза 6: Временные тренды
    if len(numeric_cols) >= 3:
        # Проверяем, есть ли колонки, похожие на годы
        year_like_cols = [col for col in df.columns if any(str(col).isdigit() and 1900 <= int(str(col)) <= 2100 
                                                           for part in str(col).split()) or 
                         (isinstance(col, (int, float)) and 1900 <= col <= 2100)]
        
        if not year_like_cols and len(numeric_cols) > 0:
            # Берем последние несколько числовых колонок как возможные временные ряды
            potential_time_cols = numeric_cols[-min(5, len(numeric_cols)):]
            
            for time_col in potential_time_cols[:1]:  # Берем одну для примера
                if len(df) > 10:
                    # Создаем график тренда
                    fig, ax = plt.subplots(figsize=(10, 5))  # Уменьшаем размер
                    
                    # Используем выборку для больших датасетов
                    if len(df) > 5000:
                        df_plot = df.sample(n=5000, random_state=42).sort_index()
                    else:
                        df_plot = df
                    
                    # Если есть категориальный признак для группировки
                    if categorical_cols:
                        cat_col = categorical_cols[0]
                        top_cats = df_plot[cat_col].value_counts().head(5).index
                        
                        for cat in top_cats:
                            subset = df_plot[df_plot[cat_col] == cat]
                            if len(subset) > 0:
                                # Сортируем по индексу для временного ряда
                                subset_sorted = subset.sort_index()
                                ax.plot(range(len(subset_sorted)), subset_sorted[time_col], 
                                       marker='o', label=cat, linewidth=1.5, markersize=3)  # Уменьшаем размер
                        
                        ax.set_xlabel('Время / Порядок наблюдений', fontsize=10)
                        ax.set_ylabel(time_col, fontsize=10)
                        ax.set_title(f'Тренд {time_col} по группам {cat_col}', fontsize=11, fontweight='bold')
                        ax.legend(fontsize=8)
                        ax.grid(alpha=0.3)
                    else:
                        # Простой временной ряд
                        ax.plot(range(len(df_plot)), df_plot[time_col].sort_index(), 
                               marker='o', linewidth=1.5, markersize=2)  # Уменьшаем размер
                        ax.set_xlabel('Время / Порядок наблюдений', fontsize=10)
                        ax.set_ylabel(time_col, fontsize=10)
                        ax.set_title(f'Тренд {time_col}', fontsize=11, fontweight='bold')
                        ax.grid(alpha=0.3)
                    
                    plt.tight_layout()
                    
                    hypotheses.append({
                        'id': len(hypotheses),
                        'Гипотеза': f"В признаке '{time_col}' наблюдается временной тренд",
                        'Обоснование': f"Значения изменяются во времени, что может указывать на динамику процесса",
                        'Метод проверки': "Временной ряд анализ, тест на стационарность, декомпозиция",
                        'plot': fig
                    })
    
    # Отображение гипотез с визуализациями
    if hypotheses:
        st.success(f"✅ Сгенерировано {len(hypotheses)} гипотез")
        st.markdown("---")
        
        for i, hyp in enumerate(hypotheses, 1):
            with st.expander(f"**Гипотеза {i}:** {hyp['Гипотеза']}", expanded=(i == 1)):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    if 'plot' in hyp and hyp['plot'] is not None:
                        st.pyplot(hyp['plot'])
                        plt.close(hyp['plot'])  # Закрываем фигуру для освобождения памяти
                
                with col2:
                    st.markdown("**📝 Обоснование:**")
                    st.write(hyp['Обоснование'])
                    st.markdown("---")
                    st.markdown("**🔬 Метод проверки:**")
                    st.write(hyp['Метод проверки'])
    else:
        st.info("💡 Гипотезы будут сгенерированы после более детального анализа данных. Убедитесь, что данные загружены корректно.")
        st.markdown("**Совет:** Проверьте, что:")
        st.markdown("- В данных есть числовые и категориальные признаки")
        st.markdown("- Данные не содержат критических ошибок")
        st.markdown("- Разделители в CSV файле корректны")










