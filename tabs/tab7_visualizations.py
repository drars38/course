"""
Вкладка 7: Дополнительные визуализации
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils import sample_data_for_plotting


def render_visualizations_tab(df, numeric_cols, categorical_cols, target_col, max_plot_points, use_sampling):
    """Отображает вкладку дополнительных визуализаций"""
    # Устанавливаем флаг, что мы на вкладке визуализации
    # Это поможет изолировать выполнение кода
    st.session_state.current_active_tab = 6
    
    # Обновляем статус прогресс-бара
    if 'status_text' in st.session_state:
        st.session_state.status_text.text("📊 Обработка вкладки: Визуализации")
    
    st.header("7. Дополнительные визуализации")
    
    if len(numeric_cols) > 1:
        st.subheader("Интерактивный анализ пар признаков")
        st.info("💡 Выберите пару признаков для быстрого анализа. Это быстрее, чем полный Pairplot.")
        
        col1, col2 = st.columns(2)
        with col1:
            x_col = st.selectbox("Признак X (ось абсцисс)", numeric_cols, key="scatter_x")
        with col2:
            y_col = st.selectbox("Признак Y (ось ординат)", numeric_cols, key="scatter_y")
        
        # Добавляем категориальный признак для цвета, если есть
        hue_col = None
        if categorical_cols:
            hue_col = st.selectbox("Признак для цветовой группировки (опционально)", 
                                 [None] + categorical_cols[:5], key="scatter_hue")
        
        if x_col and y_col and x_col != y_col:
            with st.spinner("Построение scatter plot..."):
                try:
                    # Для цветовой группировки используем полный датасет для определения категорий
                    # Выборку применяем только для визуализации, чтобы не потерять категории
                    if hue_col:
                        # Используем полный датасет для определения всех категорий
                        # ВАЖНО: dropna только по hue_col, чтобы сохранить все категории
                        full_df = df[[x_col, y_col, hue_col]].dropna(subset=[hue_col])
                        
                        # Проверяем, что hue_col действительно есть в данных
                        if hue_col not in full_df.columns:
                            st.error(f"Ошибка: признак '{hue_col}' не найден в данных")
                            full_df = None
                        elif len(full_df) == 0:
                            st.warning(f"Нет данных с заполненным признаком '{hue_col}'")
                            full_df = None
                    else:
                        # Без группировки - используем выборку сразу
                        plot_df = sample_data_for_plotting(df[[x_col, y_col]], 
                                                          max_plot_points, use_sampling)
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    
                    # Определяем, есть ли данные для группировки по цвету
                    if hue_col:
                        if full_df is not None and len(full_df) > 0:
                            # Разные цвета для разных категорий
                            # Используем полный датасет для определения всех категорий
                            unique_cats = full_df[hue_col].unique()
                            if len(unique_cats) <= 10:  # Ограничиваем количество категорий
                                # Используем более контрастные цвета
                                colors_list = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                                              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
                                
                                # Собираем информацию о категориях для отладки
                                categories_info = []
                                
                                # Сортируем категории по количеству точек для более предсказуемого отображения
                                cat_counts = full_df[hue_col].value_counts()
                                sorted_cats = cat_counts.index.tolist()
                                
                                # Добавляем checkbox'ы для фильтрации категорий
                                st.markdown("**Фильтр по категориям:**")
                                selected_categories = {}
                                cols_filter = st.columns(min(len(sorted_cats), 5))  # Максимум 5 колонок
                                
                                for idx, cat in enumerate(sorted_cats):
                                    col_idx = idx % len(cols_filter)
                                    # По умолчанию все категории выбраны
                                    if f'filter_{cat}' not in st.session_state:
                                        st.session_state[f'filter_{cat}'] = True
                                    selected_categories[cat] = cols_filter[col_idx].checkbox(
                                        f"{cat} ({cat_counts[cat]})", 
                                        value=st.session_state.get(f'filter_{cat}', True),
                                        key=f'filter_{cat}'
                                    )
                                
                                # Отладочная информация
                                debug_info = []
                                
                                # Рисуем только выбранные категории
                                filtered_cats = [cat for cat in sorted_cats if selected_categories.get(cat, True)]
                                
                                if len(filtered_cats) == 0:
                                    st.warning("⚠️ Выберите хотя бы одну категорию для отображения")
                                
                                # Рисуем категории в прямом порядке: сначала большие (снизу, zorder=1), потом маленькие (сверху, zorder=высокий)
                                # Это гарантирует, что маленькие категории будут видны поверх больших
                                for i, cat in enumerate(filtered_cats):
                                    # Фильтруем данные для каждой категории из ПОЛНОГО датасета
                                    subset_full = full_df[full_df[hue_col] == cat].copy()
                                    
                                    if len(subset_full) > 0:
                                        # Очищаем от пропущенных значений в x_col и y_col
                                        subset_clean = subset_full[[x_col, y_col]].dropna()
                                        
                                        if len(subset_clean) > 0:
                                            # Применяем выборку для визуализации
                                            max_points_per_cat = max(50, max_plot_points // max(1, len(unique_cats)))
                                            
                                            if use_sampling and len(subset_clean) > max_points_per_cat:
                                                n_sample = min(max_points_per_cat, len(subset_clean))
                                                subset_plot = subset_clean.sample(n=n_sample, random_state=42 + i)
                                            else:
                                                subset_plot = subset_clean
                                            
                                            # Используем цвет по индексу категории в исходном списке (для консистентности цветов)
                                            original_idx = sorted_cats.index(cat)
                                            color = colors_list[original_idx % len(colors_list)]
                                            
                                            # zorder: большие категории (i=0) -> zorder=1 (снизу), маленькие -> zorder=высокий (сверху)
                                            zorder_value = i + 1
                                            
                                            # Размер точек: маленькие категории получают больший размер
                                            if len(subset_clean) < 50:
                                                point_size = 100
                                            else:
                                                point_size = 60
                                            
                                            # Рисуем scatter plot для этой категории
                                            scatter = ax.scatter(subset_plot[x_col], subset_plot[y_col], 
                                                         alpha=0.8, s=point_size, label=f'{cat} ({len(subset_clean)})', 
                                                         c=color, edgecolors='black', linewidths=1.2,
                                                         zorder=zorder_value)
                                            
                                            debug_info.append(f"{cat}: {len(subset_clean)} точек, цвет={color}, отображено={len(subset_plot)}, zorder={zorder_value}, x_range=[{subset_plot[x_col].min():.2f}, {subset_plot[x_col].max():.2f}], y_range=[{subset_plot[y_col].min():.2f}, {subset_plot[y_col].max():.2f}]")
                                            categories_info.append(f"{cat}: {len(subset_clean)} точек (отображено {len(subset_plot)})")
                                        else:
                                            categories_info.append(f"{cat}: нет данных (все пропущены в x/y)")
                                            debug_info.append(f"{cat}: нет данных после dropna")
                                    else:
                                        categories_info.append(f"{cat}: нет данных")
                                        debug_info.append(f"{cat}: не найдено в full_df")
                                
                                # Показываем информацию о категориях
                                if len(categories_info) > 0:
                                    with st.expander("ℹ️ Информация о категориях"):
                                        for info in categories_info:
                                            st.text(info)
                                        
                                        # Отладочная информация (без вложенного expander)
                                        st.markdown("---")
                                        st.markdown("**🔍 Отладочная информация:**")
                                        st.text(f"Всего категорий: {len(unique_cats)}")
                                        st.text(f"Уникальные категории: {list(unique_cats)}")
                                        for debug in debug_info:
                                            st.text(debug)
                                
                                if len(unique_cats) > 0:
                                    ax.legend(title=hue_col, fontsize=9, loc='best', framealpha=0.9)
                            else:
                                # Слишком много категорий - показываем без группировки
                                plot_df_clean = full_df[[x_col, y_col]].dropna()
                                if len(plot_df_clean) > 0:
                                    plot_df_sampled = sample_data_for_plotting(plot_df_clean, max_plot_points, use_sampling)
                                    ax.scatter(plot_df_sampled[x_col], plot_df_sampled[y_col], alpha=0.6, s=30)
                                st.warning(f"Слишком много категорий ({len(unique_cats)}). Показан график без группировки.")
                        else:
                            # hue_col указан, но full_df пуст или None - показываем без группировки
                            plot_df_clean = df[[x_col, y_col]].dropna()
                            plot_df_sampled = sample_data_for_plotting(plot_df_clean, max_plot_points, use_sampling)
                            if len(plot_df_sampled) > 0:
                                ax.scatter(plot_df_sampled[x_col], plot_df_sampled[y_col], alpha=0.6, s=30, color='steelblue')
                            st.warning(f"Признак '{hue_col}' не найден в данных или нет данных. Показан график без группировки.")
                    else:
                        # Без цветовой группировки - просто scatter plot
                        plot_df_clean = plot_df[[x_col, y_col]].dropna()
                        if len(plot_df_clean) > 0:
                            ax.scatter(plot_df_clean[x_col], plot_df_clean[y_col], alpha=0.6, s=30, color='steelblue')
                        else:
                            st.warning("⚠️ Нет данных для построения графика (все значения пропущены)")
                    
                    # Линия тренда
                    try:
                        mask = df[[x_col, y_col]].notna().all(axis=1)
                        if mask.sum() > 2:
                            z = np.polyfit(df.loc[mask, x_col], df.loc[mask, y_col], 1)
                            p = np.poly1d(z)
                            x_line = np.linspace(df[x_col].min(), df[x_col].max(), 100)
                            ax.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2, label='Линия тренда')
                            # Вычисляем корреляцию
                            corr = df[[x_col, y_col]].corr().iloc[0, 1]
                            ax.text(0.05, 0.95, f'Корреляция: {corr:.3f}', 
                                   transform=ax.transAxes, fontsize=11,
                                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                    except:
                        pass
                    
                    ax.set_xlabel(x_col, fontsize=11)
                    ax.set_ylabel(y_col, fontsize=11)
                    ax.set_title(f'Scatter plot: {x_col} vs {y_col}', fontsize=12, fontweight='bold')
                    ax.grid(alpha=0.3)
                    ax.legend()
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)
                    st.success("✅ График успешно построен!")
                except Exception as e:
                    st.error(f"Ошибка при построении графика: {str(e)}")
                    import traceback
                    with st.expander("Детали ошибки"):
                        st.code(traceback.format_exc())
        elif x_col == y_col:
            st.warning("⚠️ Выберите разные признаки для сравнения")
    
    # Дополнительные графики для числовых признаков
    if numeric_cols:
        st.subheader("Матрица корреляций с scatter plots")
        st.info("💡 Матрица scatter plots может быть медленной для большого количества признаков. Рекомендуется использовать интерактивный анализ пар признаков выше.")
        
        if len(numeric_cols) <= 6 and len(numeric_cols) > 1:
            # Делаем матрицу опциональной
            build_matrix = st.checkbox("Построить полную матрицу scatter plots", value=False, key="build_matrix")
            
            if not build_matrix:
                st.info("Отметьте чекбокс выше, чтобы построить полную матрицу scatter plots")
            
            if build_matrix:
                with st.spinner("Построение матрицы scatter plots..."):
                    try:
                        # Выбираем данные для визуализации
                        plot_df = sample_data_for_plotting(df[numeric_cols], max_plot_points, use_sampling)
                        st.write(f"📊 Используется выборка из {len(plot_df):,} строк для построения матрицы")
                        
                        # Создаем кастомную матрицу scatter plots
                        n = len(numeric_cols)
                        fig, axes = plt.subplots(n, n, figsize=(4*n, 4*n))
                        
                        # Убеждаемся, что axes - это двумерный массив
                        if not isinstance(axes, np.ndarray):
                            axes = np.array([[axes]])
                        elif axes.ndim == 1:
                            axes = axes.reshape(n, n)
                        
                        # Прогресс-бар для больших матриц (только если много графиков)
                        if n * n > 9:
                            progress_bar = st.progress(0)
                            total_plots = n * n
                        else:
                            progress_bar = None
                        
                        for i, col1 in enumerate(numeric_cols):
                            for j, col2 in enumerate(numeric_cols):
                                ax = axes[i, j]
                                
                                if i == j:
                                    # Диагональ - гистограмма (используем выборку)
                                    ax.hist(plot_df[col1].dropna(), bins=15, color='skyblue', alpha=0.7, edgecolor='black')  # Уменьшаем bins
                                    ax.set_title(col1, fontsize=8, fontweight='bold')
                                else:
                                    # Scatter plot (используем выборку)
                                    ax.scatter(plot_df[col2], plot_df[col1], alpha=0.4, s=8)  # Уменьшаем размер и прозрачность точек
                                    # Линия тренда (используем все данные для точности, но только если не слишком много данных)
                                    try:
                                        if len(df) < 10000:  # Линия тренда только для небольших датасетов
                                            mask = df[[col1, col2]].notna().all(axis=1)
                                            if mask.sum() > 2:
                                                z = np.polyfit(df.loc[mask, col2], df.loc[mask, col1], 1)
                                                p = np.poly1d(z)
                                                x_line = np.linspace(df[col2].min(), df[col2].max(), 50)  # Уменьшаем точки
                                                ax.plot(x_line, p(x_line), "r--", alpha=0.4, linewidth=0.8)
                                    except:
                                        pass
                                    ax.set_xlabel(col2, fontsize=7)
                                    ax.set_ylabel(col1, fontsize=7)
                                
                                ax.grid(alpha=0.2)  # Уменьшаем прозрачность сетки
                                ax.tick_params(labelsize=6)  # Уменьшаем размер шрифта
                                
                                # Обновляем прогресс (только если есть прогресс-бар)
                                if progress_bar:
                                    plot_idx = i * n + j + 1
                                    progress_bar.progress(plot_idx / total_plots)
                        
                        if progress_bar:
                            progress_bar.empty()  # Убираем прогресс-бар
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close(fig)  # Закрываем фигуру для освобождения памяти
                        st.success("✅ Матрица scatter plots успешно построена!")
                    except Exception as e:
                        st.error(f"Ошибка при построении матрицы scatter plots: {str(e)}")
                        import traceback
                        with st.expander("Детали ошибки"):
                            st.code(traceback.format_exc())
        elif len(numeric_cols) == 1:
            st.info("Для построения матрицы scatter plots необходимо минимум 2 числовых признака")
        else:
            st.info("Для построения матрицы scatter plots необходимо не более 6 числовых признаков")
    
    if numeric_cols and categorical_cols:
        st.subheader("Violin plots")
        cat_col = st.selectbox("Категориальный признак", categorical_cols, key="violin_cat")
        num_col = st.selectbox("Числовой признак", numeric_cols, key="violin_num")
        
        if cat_col and num_col:
            with st.spinner("Построение Violin plot..."):
                try:
                    # Используем выборку для violin plot
                    violin_df = sample_data_for_plotting(df[[cat_col, num_col]], max_plot_points, use_sampling)
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.violinplot(x=cat_col, y=num_col, data=violin_df, ax=ax, palette='Set2')
                    ax.set_title(f'Распределение {num_col} по {cat_col}', fontsize=12, fontweight='bold')
                    ax.tick_params(axis='x', rotation=45)
                    ax.grid(alpha=0.3, axis='y')
                    st.pyplot(fig)
                    plt.close(fig)  # Закрываем фигуру для освобождения памяти
                except Exception as e:
                    st.error(f"Ошибка при построении Violin plot: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
    
    # Финальная сводка
    st.subheader("Финальная сводка")
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Числовые признаки:**", ", ".join(numeric_cols) if numeric_cols else "Нет")
    with col2:
        st.write("**Категориальные признаки:**", ", ".join(categorical_cols) if categorical_cols else "Нет")
    
    st.write(f"**Пропущенных значений:** {df.isnull().sum().sum()}")
    if target_col:
        st.write(f"**Целевая переменная:** {target_col}")
