"""
Вкладка 2: Пропущенные значения
"""
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def render_missing_tab(df):
    """Отображает вкладку анализа пропущенных значений"""
    from utils import compute_missing_stats
    
    # Устанавливаем флаг активной вкладки для изоляции
    st.session_state.current_active_tab = 1
    
    # Обновляем статус прогресс-бара
    if 'status_text' in st.session_state:
        st.session_state.status_text.text("❌ Обработка вкладки: Пропущенные значения")
    
    st.header("2. Анализ пропущенных значений")
    
    # Используем кэшированную функцию
    missing_df = compute_missing_stats(df)
    
    if len(missing_df) > 0:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Тепловая карта пропусков")
            with st.spinner("Построение тепловой карты..."):
                # Используем выборку для больших датасетов
                if len(df) > 10000:
                    sample_df = df.sample(n=min(10000, len(df)), random_state=42)
                else:
                    sample_df = df
                fig, ax = plt.subplots(figsize=(10, max(5, len(df.columns) * 0.25)))  # Уменьшаем размер
                sns.heatmap(sample_df.isnull(), yticklabels=False, cbar=True, cmap='viridis', 
                          ax=ax, cbar_kws={'shrink': 0.8})
                ax.set_title('Тепловая карта пропущенных значений', fontsize=11, fontweight='bold')
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
        
        with col2:
            st.subheader("Гистограмма пропусков")
            if len(missing_df) > 0:
                with st.spinner("Построение гистограммы..."):
                    fig, ax = plt.subplots(figsize=(8, max(5, len(missing_df) * 0.4)))  # Уменьшаем размер
                    bars = ax.barh(missing_df.index, missing_df['Процент'], color='coral')
                    ax.set_xlabel('Процент пропусков (%)', fontsize=10)
                    ax.set_ylabel('Признаки', fontsize=10)
                    ax.set_title('Процент пропущенных значений', fontsize=11, fontweight='bold')
                    ax.grid(axis='x', alpha=0.3)
                    for i, bar in enumerate(bars):
                        width = bar.get_width()
                        ax.text(width + 0.5, bar.get_y() + bar.get_height()/2,
                               f'{width:.1f}%', ha='left', va='center', fontsize=8)  # Уменьшаем шрифт
                    plt.tight_layout()
                    st.pyplot(fig, use_container_width=True)
                    plt.close(fig)
        
        st.subheader("Детальная информация о пропусках")
        st.dataframe(missing_df, use_container_width=True)
    else:
        st.success("✅ Пропущенных значений не обнаружено!")
