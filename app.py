import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
import io
warnings.filterwarnings('ignore')

# Импорт утилит и модулей вкладок
from utils import load_data, sample_data_for_plotting, find_target_column
from tabs.tab1_overview import render_overview_tab
from tabs.tab2_missing import render_missing_tab
from tabs.tab3_distributions import render_distributions_tab
from tabs.tab4_outliers import render_outliers_tab
from tabs.tab5_correlations import render_correlations_tab
from tabs.tab6_hypotheses import render_hypotheses_tab
from tabs.tab7_visualizations import render_visualizations_tab

# Настройка страницы
st.set_page_config(
    page_title="Автоматический EDA анализ",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Настройка стиля
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Оптимизация matplotlib для производительности
plt.rcParams['figure.dpi'] = 80  # Уменьшаем DPI для ускорения
plt.rcParams['savefig.dpi'] = 80
plt.rcParams['figure.max_open_warning'] = 0  # Отключаем предупреждения о множественных фигурах
plt.rcParams['figure.facecolor'] = 'white'  # Упрощаем фон
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['font.size'] = 9  # Уменьшаем размер шрифта
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
# Дополнительные оптимизации для ускорения
plt.rcParams['path.simplify'] = True  # Упрощаем пути для ускорения рендеринга
plt.rcParams['path.simplify_threshold'] = 1.0
plt.rcParams['agg.path.chunksize'] = 10000  # Размер чанков для агрегации
plt.rcParams['figure.autolayout'] = False  # Отключаем автоматическую компоновку для ускорения

# Заголовок приложения
st.title("📊 Автоматический исследовательский анализ данных (EDA)")

# Глобальный прогресс-бар в верхней части страницы
progress_container = st.container()
with progress_container:
    progress_bar = st.progress(0)
    status_text = st.empty()
    status_text.text("⏳ Ожидание загрузки данных...")
    # Сохраняем status_text в session_state для доступа из других модулей
    st.session_state.status_text = status_text
st.markdown("---")

# Боковая панель для загрузки файла
st.sidebar.header("📁 Загрузка данных")

# Сначала проверяем, загружен ли файл через file_uploader
uploaded_file = st.sidebar.file_uploader(
    "Загрузите CSV/TSV файл",
    type=['csv', 'tsv', 'txt'],
    help="Выберите файл для анализа (CSV, TSV или TXT)"
)

# Флаг для использования примера данных
use_example_data = False
example_df = None

# Если пользователь загрузил файл, очищаем пример данных
if uploaded_file is not None:
    if 'example_df' in st.session_state:
        del st.session_state['example_df']
else:
    # Если файл не загружен, проверяем, есть ли пример данных в session_state
    example_df = st.session_state.get('example_df', None)
    if example_df is not None:
        use_example_data = True
        st.sidebar.success("✅ Используется пример: Titanic dataset")
        if st.sidebar.button("🔄 Очистить пример"):
            # Очищаем пример
            if 'example_df' in st.session_state:
                del st.session_state['example_df']
            st.rerun()

# Настройки разделителя
st.sidebar.subheader("⚙️ Настройки загрузки")
delimiter_option = st.sidebar.radio(
    "Разделитель",
    ["Автоопределение", "Запятая (,)", "Табуляция (\\t)", "Точка с запятой (;)"],
    help="Выберите разделитель или оставьте автоопределение"
)

delimiter_map = {
    "Автоопределение": None,
    "Запятая (,)": ",",
    "Табуляция (\\t)": "\t",
    "Точка с запятой (;)": ";"
}
selected_delimiter = delimiter_map[delimiter_option]

# Настройки производительности
st.sidebar.subheader("⚡ Настройки производительности")
max_plot_points = st.sidebar.slider(
    "Максимум точек для графиков",
    min_value=1000,
    max_value=50000,
    value=10000,
    step=1000,
    help="Ограничение количества точек для ускорения построения графиков"
)
use_sampling = st.sidebar.checkbox(
    "Использовать выборку для больших датасетов",
    value=True,
    help="Автоматически применять выборку для датасетов > 10000 строк"
)
fast_mode = st.sidebar.checkbox(
    "Быстрый режим (упрощенные графики)",
    value=False,
    help="Упрощает графики для ускорения (меньше деталей, быстрее построение)"
)

# Все функции перенесены в utils.py

# Загрузка данных
if uploaded_file is not None or use_example_data:
    # Обновляем прогресс-бар (он уже создан выше)
    if uploaded_file is not None:
        status_text.text("📂 Загрузка файла...")
        progress_bar.progress(10)
        
        # Вычисляем хеш файла для определения, изменились ли данные
        import hashlib
        file_hash = hashlib.md5(uploaded_file.getvalue()).hexdigest()
        
        # Если файл изменился, сбрасываем состояние
        if 'last_file_hash' not in st.session_state or st.session_state.last_file_hash != file_hash:
            st.session_state.last_file_hash = file_hash
            st.session_state.tabs_initialized = False
            st.session_state.last_active_tab = -1
            # Очищаем кэш гипотез при загрузке нового файла
            for key in list(st.session_state.keys()):
                if key.startswith('hypotheses_cache_'):
                    del st.session_state[key]
        
        df, error, has_shift = load_data(uploaded_file, selected_delimiter)
        progress_bar.progress(30)
    else:
        # Используем пример данных напрямую
        status_text.text("📂 Загрузка примера данных...")
        progress_bar.progress(10)
        df = example_df.copy()
        error = None
        has_shift = False
        progress_bar.progress(30)
        
        # Для примера данных используем хеш на основе DataFrame
        import hashlib
        example_hash = hashlib.md5(str(df.values.tobytes()).encode()).hexdigest()
        if 'last_file_hash' not in st.session_state or st.session_state.last_file_hash != example_hash:
            st.session_state.last_file_hash = example_hash
            st.session_state.tabs_initialized = False
            st.session_state.last_active_tab = -1
            # Очищаем кэш гипотез при загрузке нового файла
            for key in list(st.session_state.keys()):
                if key.startswith('hypotheses_cache_'):
                    del st.session_state[key]
    
    if error:
        progress_bar.progress(40)
        status_text.text("⚠️ Обработка предупреждений...")
        
        # Проверяем, это ошибка сдвига или ошибка загрузки
        if "Обнаружено несовпадение типов данных" in str(error):
            # Это предупреждение о сдвиге - показываем, но продолжаем работу
            progress_bar.progress(50)
            st.warning("⚠️ Обнаружена проблема с данными")
            st.info(error)
            # Продолжаем работу с данными, если они загружены
            if df is None:
                status_text.text("❌ Ошибка загрузки данных")
                progress_bar.progress(0)
                st.error("Не удалось загрузить данные для анализа")
                df = None
        else:
            # Это критическая ошибка загрузки - останавливаемся
            status_text.text("❌ Ошибка загрузки файла")
            progress_bar.progress(0)
            st.error(f"Ошибка при загрузке файла: {error}")
            df = None
    
    if df is not None:
        status_text.text("🔍 Анализ структуры данных...")
        progress_bar.progress(50)
        
        # Показываем информацию о структуре данных
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        progress_bar.progress(60)
        
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        progress_bar.progress(70)
        
        # Поиск целевой переменной (для использования во всех вкладках)
        status_text.text("🎯 Поиск целевой переменной...")
        target_col = find_target_column(df, numeric_cols, categorical_cols)
        progress_bar.progress(80)
        
        status_text.text("✅ Инициализация интерфейса...")
        progress_bar.progress(90)
        
        # Обновляем прогресс-бар до 100% и показываем финальный статус
        progress_bar.progress(100)
        status_text.text(f"✅ Готово: {df.shape[0]} строк × {df.shape[1]} столбцов | Выберите вкладку для анализа")
        
        st.success(f"✅ Данные успешно загружены! Размер: {df.shape[0]} строк × {df.shape[1]} столбцов")
        
        # Экспорт отчета
        st.sidebar.markdown("---")
      
        
        # Предупреждение о выборке данных для больших датасетов
        if use_sampling and len(df) > max_plot_points:
            st.info(f"ℹ️ Для ускорения визуализаций используется выборка из {max_plot_points:,} строк (из {len(df):,} всего). "
                   f"Статистические расчеты выполняются на полном датасете.")
        
        # Основной контент
        # ОПТИМИЗАЦИЯ: Выполняем код только для активной вкладки
        # Определяем активную вкладку по ключам виджетов, которые были изменены
        
        # Инициализируем session_state
        if 'last_active_tab' not in st.session_state:
            st.session_state.last_active_tab = -1
        if 'tabs_initialized' not in st.session_state:
            st.session_state.tabs_initialized = False
        if 'widget_values' not in st.session_state:
            st.session_state.widget_values = {}
        
        # Флаги для статичных вкладок (выполняются только 1 раз)
        # Вкладки 0, 1, 5 - статичные (без интерактивных графиков)
        # Вкладки 2, 3, 4, 6 - интерактивные (с виджетами для изменения графиков)
        static_tabs = {0, 1, 5}  # Обзор, Пропущенные значения, Гипотезы
        interactive_tabs = {2, 3, 4, 6}  # Распределения, Выбросы, Корреляции, Визуализации
        
        # Флаги для статичных вкладок больше не нужны - они всегда выполняются
        # (тяжелые операции внутри них кэшируются через @st.cache_data)
        
        # Маппинг ключей виджетов на индексы вкладок
        widget_to_tab = {
            'dist_col': 2,             # Распределения
            'show_advanced_dist': 2,  # Распределения
            'outlier': 3,              # Выбросы
            'scatter': 3,              # Выбросы
            'scatter_display_mode': 3, # Выбросы
            'group': 4,                # Корреляции
            'num_group': 4,            # Корреляции
            'scatter_x': 6,            # Визуализации
            'scatter_y': 6,            # Визуализации
            'scatter_hue': 6,          # Визуализации
            'build_matrix': 6,         # Визуализации
            'violin_cat': 6,           # Визуализации
            'violin_num': 6            # Визуализации
        }
        
        # ПРИОРИТЕТ 1: Проверяем query_params для переключения вкладок (самый высокий приоритет)
        query_params = st.query_params
        active_tab_index = None
        if 'tab' in query_params:
            try:
                tab_value = query_params['tab']
                if isinstance(tab_value, list) and len(tab_value) > 0:
                    tab_index = int(tab_value[0])
                elif isinstance(tab_value, str):
                    tab_index = int(tab_value)
                else:
                    tab_index = None
                
                if tab_index is not None and 0 <= tab_index <= 6:
                    active_tab_index = tab_index
            except (ValueError, IndexError, TypeError, AttributeError):
                pass
        
        # ПРИОРИТЕТ 2: Если query_params нет, определяем по последнему измененному виджету
        # НО: если пользователь переключился на статичную вкладку, она должна быть активна
        if active_tab_index is None:
            # Сначала проверяем, не переключился ли пользователь на статичную вкладку
            # (определяем по изменению last_active_tab, если он был установлен ранее)
            if st.session_state.last_active_tab != -1:
                # Если last_active_tab указывает на статичную вкладку, используем её
                if st.session_state.last_active_tab in static_tabs:
                    active_tab_index = st.session_state.last_active_tab
                else:
                    # Иначе определяем по виджетам
                    active_tab_index = st.session_state.last_active_tab
            else:
                active_tab_index = 0
            
            # Проверяем, какие виджеты были изменены (сравниваем текущие значения с предыдущими)
            # Важно: проверяем в обратном порядке, чтобы последний измененный виджет имел приоритет
            # НО: если активная вкладка статичная, не переопределяем её по виджетам
            widget_changed_tab = None
            for widget_key, tab_idx in reversed(list(widget_to_tab.items())):
                if widget_key in st.session_state:
                    current_value = st.session_state[widget_key]
                    previous_value = st.session_state.widget_values.get(widget_key, None)
                    
                    # Если значение изменилось, значит эта вкладка активна
                    if previous_value is None or current_value != previous_value:
                        widget_changed_tab = tab_idx
                        st.session_state.widget_values[widget_key] = current_value
                        break  # Берем последний измененный виджет
            
            # Используем вкладку из виджета только если текущая активная вкладка не статичная
            # или если мы не знаем, какая вкладка активна
            if widget_changed_tab is not None and active_tab_index not in static_tabs:
                active_tab_index = widget_changed_tab
        
        # Определяем, изменилась ли активная вкладка
        tab_changed = st.session_state.last_active_tab != active_tab_index
        
        # Определяем, нужно ли обрабатывать вкладки:
        # - Статичные вкладки: только если еще не обработаны
        # - Интерактивные вкладки: если они активны или их виджеты изменились
        # - При первом запуске: обрабатываем все вкладки
        should_render_all = not st.session_state.tabs_initialized or tab_changed
        
        # Определяем, нужно ли обрабатывать конкретную вкладку
        def should_render_tab(tab_idx):
            if tab_idx in static_tabs:
                # Статичная вкладка:
                # - Всегда выполняется, если она активна (определяется по query_params или active_tab_index)
                # - Или если еще не обработана (при первом запуске)
                # - НЕ выполняется, если она неактивна и уже обработана (оптимизация)
                flag_name = f'tab_{tab_idx}_rendered'
                is_rendered = st.session_state.get(flag_name, False)
                is_active = active_tab_index == tab_idx
                
                # Проверяем также query_params напрямую (на случай, если active_tab_index еще не обновился)
                is_active_by_url = False
                if 'tab' in query_params:
                    try:
                        tab_value = query_params['tab']
                        if isinstance(tab_value, list) and len(tab_value) > 0:
                            url_tab = int(tab_value[0])
                        elif isinstance(tab_value, str):
                            url_tab = int(tab_value)
                        else:
                            url_tab = None
                        if url_tab == tab_idx:
                            is_active_by_url = True
                    except (ValueError, IndexError, TypeError, AttributeError):
                        pass
                
                # Если вкладка активна (по любому признаку) - всегда выполняем
                if is_active or is_active_by_url:
                    return True
                # Если вкладка неактивна, но еще не обработана - выполняем
                if not is_rendered:
                    return True
                # Если вкладка неактивна и уже обработана - не выполняем (оптимизация)
                return False
            elif tab_idx in interactive_tabs:
                # Интерактивная вкладка - если она активна или при первом запуске
                return should_render_all or active_tab_index == tab_idx
            else:
                # По умолчанию - как раньше
                return should_render_all or active_tab_index == tab_idx
        
        # Отладочная информация в sidebar
        with st.sidebar.expander("🔧 Отладка вкладок", expanded=False):
            st.write(f"**Активная вкладка:** {active_tab_index}")
            st.write(f"**Последняя вкладка:** {st.session_state.last_active_tab}")
            st.write(f"**Вкладка изменилась:** {tab_changed}")
            st.write(f"**Обрабатывать все:** {should_render_all}")
            st.write(f"**Инициализировано:** {st.session_state.tabs_initialized}")
            query_params = st.query_params
            if 'tab' in query_params:
                st.write(f"**URL параметр tab:** {query_params.get('tab')}")
            else:
                st.write(f"**URL параметр tab:** отсутствует")
            if 'widget_values' in st.session_state:
                st.write(f"**Измененные виджеты:** {list(st.session_state.widget_values.keys())}")
            st.write("**Статичные вкладки:** Всегда выполняются (легкие операции)")
            st.write("**Интерактивные вкладки:** Выполняются только при активности")
        
        # Обновляем состояние
        if tab_changed or not st.session_state.tabs_initialized:
            st.session_state.last_active_tab = active_tab_index
            st.session_state.tabs_initialized = True
        
        tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
            "📋 Обзор данных",
            "❌ Пропущенные значения",
            "📈 Распределения",
            "🔍 Выбросы",
            "🔗 Корреляции",
            "🎯 Гипотезы",
            "📊 Визуализации"
        ])
        
        # JavaScript для управления вкладками и URL
        st.markdown(f"""
        <script>
            (function() {{
                // Функция для обновления URL с параметром tab
                function updateURL(tabIndex) {{
                    try {{
                        const currentUrl = window.location.href;
                        const baseUrl = currentUrl.split('?')[0];
                        const newUrl = baseUrl + '?tab=' + tabIndex;
                        window.history.replaceState({{}}, '', newUrl);
                        console.log('URL updated to tab=' + tabIndex);
                    }} catch (e) {{
                        console.error('Error updating URL:', e);
                        // Fallback: используем более простой метод
                        const newUrl = window.location.pathname + '?tab=' + tabIndex;
                        window.history.replaceState({{}}, '', newUrl);
                    }}
                }}
                
                // Функция для получения текущей активной вкладки
                function getCurrentTabIndex() {{
                    // Пробуем разные селекторы для определения активной вкладки
                    const selectors = [
                        'button[data-baseweb="tab"][aria-selected="true"]',
                        '[role="tab"][aria-selected="true"]',
                        '.stTabs [role="tablist"] button[aria-selected="true"]',
                        '.stTabs button[aria-selected="true"]'
                    ];
                    
                    for (let selector of selectors) {{
                        const activeTab = document.querySelector(selector);
                        if (activeTab) {{
                            // Находим индекс среди всех вкладок
                            const baseSelector = selector.replace('[aria-selected="true"]', '');
                            const allTabs = document.querySelectorAll(baseSelector);
                            
                            for (let i = 0; i < allTabs.length; i++) {{
                                if (allTabs[i] === activeTab) {{
                                    console.log('getCurrentTabIndex: found tab', i, 'using selector:', selector);
                                    return i;
                                }}
                            }}
                        }}
                    }}
                    
                    // Альтернативный метод: ищем по классу stTabsActive или другим признакам
                    const tabsByClass = document.querySelectorAll('.stTabs button, [role="tab"]');
                    for (let i = 0; i < tabsByClass.length; i++) {{
                        const ariaSelected = tabsByClass[i].getAttribute('aria-selected');
                        if (ariaSelected === 'true') {{
                            console.log('getCurrentTabIndex: found tab', i, 'by aria-selected');
                            return i;
                        }}
                    }}
                    
                    console.log('getCurrentTabIndex: could not find active tab, returning -1');
                    return -1;
                }}
                
                // Функция для переключения на нужную вкладку
                function switchToTab(tabIndex) {{
                    const selectors = [
                        'button[data-baseweb="tab"]',
                        '[role="tab"]',
                        '.stTabs [role="tablist"] button',
                        '.stTabs button'
                    ];
                    
                    for (let selector of selectors) {{
                        const tabs = document.querySelectorAll(selector);
                        if (tabs.length > tabIndex) {{
                            const targetTab = tabs[tabIndex];
                            if (targetTab) {{
                                const currentIndex = getCurrentTabIndex();
                                if (currentIndex !== tabIndex) {{
                                    targetTab.click();
                                    updateURL(tabIndex);
                                    return true;
                                }} else {{
                                    // Вкладка уже активна, просто обновляем URL
                                    updateURL(tabIndex);
                                    return true;
                                }}
                            }}
                        }}
                    }}
                    return false;
                }}
                
                // Отслеживаем изменения активной вкладки и обновляем URL
                function setupTabMonitoring() {{
                    let lastTabIndex = -1;
                    
                    // Функция для проверки и обновления URL
                    function checkAndUpdateTab() {{
                        const currentIndex = getCurrentTabIndex();
                        const urlParams = new URLSearchParams(window.location.search);
                        const urlTabIndex = urlParams.get('tab');
                        
                        // Если вкладка изменилась или URL не соответствует активной вкладке
                        if (currentIndex !== lastTabIndex || urlTabIndex !== String(currentIndex)) {{
                            lastTabIndex = currentIndex;
                            updateURL(currentIndex);
                        }}
                    }}
                    
                    // Используем MutationObserver для отслеживания изменений
                    const observer = new MutationObserver(function(mutations) {{
                        checkAndUpdateTab();
                    }});
                    
                    // Наблюдаем за изменениями в контейнере вкладок
                    const tabContainer = document.querySelector('.stTabs') || 
                                       document.querySelector('[role="tablist"]') ||
                                       document.body;
                    
                    if (tabContainer) {{
                        observer.observe(tabContainer, {{
                            attributes: true,
                            attributeFilter: ['aria-selected', 'class', 'data-testid'],
                            childList: false,
                            subtree: true
                        }});
                    }}
                    
                    // Также отслеживаем клики напрямую
                    function attachTabClickListeners() {{
                        const selectors = [
                            'button[data-baseweb="tab"]',
                            '[role="tab"]',
                            '.stTabs [role="tablist"] button',
                            '.stTabs button'
                        ];
                        
                        for (let selector of selectors) {{
                            const tabs = document.querySelectorAll(selector);
                            tabs.forEach((tab, index) => {{
                                // Удаляем старые обработчики, если есть
                                if (tab.dataset.listenerAttached === 'true') {{
                                    return; // Уже прикреплен
                                }}
                                tab.dataset.listenerAttached = 'true';
                                
                                // Добавляем новый обработчик
                                tab.addEventListener('click', function(e) {{
                                    e.stopPropagation();
                                    console.log('Tab clicked:', index);
                                    // Обновляем URL СРАЗУ при клике (до rerun Streamlit)
                                    updateURL(index);
                                    // Дополнительная проверка через небольшую задержку
                                    setTimeout(function() {{
                                        checkAndUpdateTab();
                                    }}, 50);
                                }}, true);  // Используем capture phase для более раннего срабатывания
                            }});
                        }}
                    }}
                    
                    // Прикрепляем обработчики сразу и периодически (на случай динамического обновления DOM)
                    attachTabClickListeners();
                    setInterval(attachTabClickListeners, 1000);
                    
                    // Периодическая проверка (на случай, если MutationObserver не сработал)
                    setInterval(checkAndUpdateTab, 300);
                    
                    // Первая проверка сразу (несколько попыток для надежности)
                    setTimeout(checkAndUpdateTab, 100);
                    setTimeout(checkAndUpdateTab, 300);
                    setTimeout(checkAndUpdateTab, 500);
                    setTimeout(checkAndUpdateTab, 1000);
                }}
                
                // Инициализация
                const targetTabIndex = {active_tab_index};
                
                // Устанавливаем мониторинг вкладок
                setupTabMonitoring();
                
                // Функция для первоначальной установки URL
                function initializeTabURL() {{
                    const currentIndex = getCurrentTabIndex();
                    if (currentIndex >= 0) {{
                        updateURL(currentIndex);
                        console.log('initializeTabURL: set URL to tab', currentIndex, 'from active tab');
                    }} else {{
                        // Если не удалось определить, используем значение из URL или 0
                        const urlParams = new URLSearchParams(window.location.search);
                        const urlTabIndex = urlParams.get('tab');
                        if (urlTabIndex !== null) {{
                            const tabNum = parseInt(urlTabIndex);
                            updateURL(tabNum);
                            console.log('initializeTabURL: set URL to tab', tabNum, 'from URL param');
                        }} else {{
                            updateURL(0);
                            console.log('initializeTabURL: set URL to tab 0 (default)');
                        }}
                    }}
                }}
                
                // Обновляем URL сразу при загрузке (несколько попыток для надежности)
                setTimeout(initializeTabURL, 50);
                setTimeout(initializeTabURL, 150);
                setTimeout(initializeTabURL, 300);
                setTimeout(initializeTabURL, 500);
                setTimeout(initializeTabURL, 1000);
                
                // Если нужно переключиться на другую вкладку при загрузке
                if (targetTabIndex > 0) {{
                    let attempts = 0;
                    const maxAttempts = 25;
                    
                    const trySwitch = function() {{
                        attempts++;
                        const currentIndex = getCurrentTabIndex();
                        
                        if (currentIndex === targetTabIndex) {{
                            // Уже на нужной вкладке
                            updateURL(targetTabIndex);
                            clearInterval(interval);
                            return;
                        }}
                        
                        if (switchToTab(targetTabIndex) || attempts >= maxAttempts) {{
                            clearInterval(interval);
                            // Обновляем URL после переключения
                            setTimeout(function() {{
                                const finalIndex = getCurrentTabIndex();
                                if (finalIndex >= 0) {{
                                    updateURL(finalIndex);
                                }} else {{
                                    updateURL(targetTabIndex);
                                }}
                            }}, 200);
                        }}
                    }};
                    
                    // Первая попытка сразу
                    setTimeout(trySwitch, 150);
                    
                    // Дополнительные попытки с задержкой
                    const interval = setInterval(trySwitch, 100);
                }}
            }})();
        </script>
        """, unsafe_allow_html=True)
        
        # ========== ВКЛАДКА 1: ОБЗОР ДАННЫХ (СТАТИЧНАЯ) ==========
        with tab1:
            # Статичные вкладки всегда выполняются (легкие, тяжелые операции кэшируются)
            render_overview_tab(df, numeric_cols, categorical_cols)
        
        # ========== ВКЛАДКА 2: ПРОПУЩЕННЫЕ ЗНАЧЕНИЯ (СТАТИЧНАЯ) ==========
        with tab2:
            # Статичные вкладки всегда выполняются (легкие, тяжелые операции кэшируются)
            render_missing_tab(df)
        
        # ========== ВКЛАДКА 3: РАСПРЕДЕЛЕНИЯ (ИНТЕРАКТИВНАЯ) ==========
        with tab3:
            # Все вкладки всегда выполняются (тяжелые операции кэшируются)
            render_distributions_tab(df, numeric_cols, categorical_cols)
        
        # ========== ВКЛАДКА 4: ВЫБРОСЫ (ИНТЕРАКТИВНАЯ) ==========
        with tab4:
            # Все вкладки всегда выполняются (тяжелые операции кэшируются)
            render_outliers_tab(df, numeric_cols, max_plot_points, use_sampling)
        
        # ========== ВКЛАДКА 5: КОРРЕЛЯЦИИ (ИНТЕРАКТИВНАЯ) ==========
        with tab5:
            # Все вкладки всегда выполняются (тяжелые операции кэшируются)
            render_correlations_tab(df, numeric_cols, categorical_cols)
        
        # ========== ВКЛАДКА 6: ГИПОТЕЗЫ (СТАТИЧНАЯ) ==========
        with tab6:
            # Статичные вкладки всегда выполняются (тяжелые операции кэшируются)
            render_hypotheses_tab(df, numeric_cols, categorical_cols, target_col, max_plot_points, use_sampling)
        
        # ========== ВКЛАДКА 7: ДОПОЛНИТЕЛЬНЫЕ ВИЗУАЛИЗАЦИИ (ИНТЕРАКТИВНАЯ) ==========
        with tab7:
            # Все вкладки всегда выполняются (тяжелые операции кэшируются)
            render_visualizations_tab(df, numeric_cols, categorical_cols, target_col, max_plot_points, use_sampling)
        
        # Обновляем финальный статус после обработки всех вкладок
        status_text.text(f"✅ Готово: {df.shape[0]} строк × {df.shape[1]} столбцов | Анализ завершен")
        
        # Экспорт отчета
        st.sidebar.markdown("---")
        st.sidebar.subheader("📤 Экспорт отчета")
        
        # Подготовка данных для экспорта
        from utils import generate_html_report, generate_pdf_report, compute_correlation_matrix
        from tabs.tab6_hypotheses import _compute_hypotheses_data
        
        # Вычисляем данные для отчета
        correlation_matrix = compute_correlation_matrix(df, numeric_cols) if len(numeric_cols) > 1 else None
        
        # VIF данные (упрощенная версия для экспорта)
        vif_data = None
        if len(numeric_cols) >= 2:
            try:
                from statsmodels.stats.outliers_influence import variance_inflation_factor
                from statsmodels.tools.tools import add_constant
                df_vif = df[numeric_cols].dropna()
                if len(df_vif) > len(numeric_cols):
                    X = add_constant(df_vif)
                    vif_data = []
                    for i, col in enumerate(numeric_cols):
                        try:
                            vif = variance_inflation_factor(X.values, i + 1)
                            vif_data.append({
                                'Признак': col,
                                'VIF': f"{vif:.2f}",
                                'Оценка': 'Сильная' if vif >= 10 else ('Умеренная' if vif >= 5 else 'Слабая')
                            })
                        except:
                            pass
            except:
                pass
        
        # Гипотезы для экспорта (без графиков)
        hypotheses_export = None
        try:
            hypotheses_full = _compute_hypotheses_data(df, numeric_cols, categorical_cols, target_col, max_plot_points, use_sampling)
            if hypotheses_full:
                hypotheses_export = []
                for hyp in hypotheses_full:
                    hyp_export = {
                        'Гипотеза': hyp.get('Гипотеза', ''),
                        'Обоснование': hyp.get('Обоснование', ''),
                        'Метод проверки': hyp.get('Метод проверки', ''),
                    }
                    if 'statistical_test' in hyp:
                        hyp_export['statistical_test'] = hyp['statistical_test']
                    hypotheses_export.append(hyp_export)
        except:
            pass
        
        # Кнопки экспорта
        col1, col2 = st.sidebar.columns(2)
        with col1:
            html_report = generate_html_report(df, numeric_cols, categorical_cols, target_col, 
                                              correlation_matrix, vif_data, hypotheses_export)
            st.sidebar.download_button(
                label="📄 Скачать HTML",
                data=html_report,
                file_name=f"eda_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.html",
                mime="text/html",
                use_container_width=True
            )
        
        with col2:
            try:
                pdf_report = generate_pdf_report(df, numeric_cols, categorical_cols, target_col,
                                               correlation_matrix, vif_data, hypotheses_export)
                st.sidebar.download_button(
                    label="📑 Скачать PDF",
                    data=pdf_report,
                    file_name=f"eda_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
            except Exception as e:
                st.sidebar.error(f"Ошибка генерации PDF: {str(e)}")
    
    else:
        st.info("👆 Пожалуйста, загрузите CSV файл в боковой панели для начала анализа")
else:
    st.info("👆 Пожалуйста, загрузите CSV файл в боковой панели для начала анализа")
    
    # Пример данных для демонстрации
    st.markdown("---")
    st.subheader("📥 Загрузка примеров датасетов")
    
    # Вкладки для разных источников данных
    example_tab1, example_tab2 = st.tabs(["🌐 Seaborn (встроенные)", "📊 Kaggle"])
    
    with example_tab1:
        st.markdown("**Загрузка встроенных датасетов из библиотеки Seaborn**")
        if st.button("🛳️ Загрузить Titanic (Seaborn)"):
            try:
                df_example = sns.load_dataset('titanic')
                if df_example is not None and not df_example.empty:
                    st.session_state['example_df'] = df_example
                    st.success(f"✅ Пример загружен! Размер: {df_example.shape[0]} строк × {df_example.shape[1]} столбцов")
                    st.rerun()
                else:
                    st.error("❌ Не удалось загрузить пример: датасет пуст")
            except Exception as e:
                error_msg = str(e)
                if "URLError" in error_msg or "HTTPError" in error_msg or "timeout" in error_msg.lower():
                    st.error("❌ Не удалось загрузить пример: нет подключения к интернету. "
                            "Seaborn требует интернет для загрузки датасетов.")
                elif "dataset" in error_msg.lower() or "not found" in error_msg.lower():
                    st.error("❌ Не удалось загрузить пример: датасет не найден в библиотеке seaborn")
                else:
                    st.error(f"❌ Не удалось загрузить пример: {error_msg}")
    
    with example_tab2:
        st.markdown("**Скачивание датасетов с Kaggle**")
        
        # Настройка Kaggle API
        with st.expander("⚙️ Настройка Kaggle API", expanded=True):
            st.markdown("""
            **Что такое kaggle.json?**
            
            `kaggle.json` — это файл с вашими учетными данными для доступа к Kaggle API. 
            Он содержит ваш username и API ключ в формате JSON:
            ```json
            {
                "username": "ваш_username",
                "key": "ваш_api_ключ_длинная_строка"
            }
            ```
            
            **Как получить kaggle.json:**
            1. Зарегистрируйтесь на [kaggle.com](https://www.kaggle.com) (если еще не зарегистрированы)
            2. Войдите в свой аккаунт
            3. Перейдите в настройки: **Account** → **API** → **Create New API Token**
            4. Файл `kaggle.json` автоматически скачается на ваш компьютер
            5. Откройте этот файл в текстовом редакторе, чтобы увидеть username и key
            """)
            
            # Два способа настройки: загрузка файла или ввод вручную
            setup_method = st.radio(
                "Способ настройки:",
                ["📁 Загрузить файл kaggle.json", "✍️ Ввести вручную"],
                help="Выберите удобный способ"
            )
            
            if setup_method == "📁 Загрузить файл kaggle.json":
                uploaded_kaggle = st.file_uploader(
                    "Загрузите файл kaggle.json",
                    type=['json'],
                    help="Выберите файл kaggle.json, который вы скачали с Kaggle"
                )
                
                if uploaded_kaggle is not None:
                    try:
                        import json
                        kaggle_data = json.load(uploaded_kaggle)
                        kaggle_username = kaggle_data.get('username', '')
                        kaggle_key = kaggle_data.get('key', '')
                        
                        if kaggle_username and kaggle_key:
                            from utils import setup_kaggle_api
                            success, message = setup_kaggle_api(kaggle_username, kaggle_key)
                            if success:
                                st.success(message)
                            else:
                                st.error(message)
                        else:
                            st.error("❌ Файл не содержит username или key. Убедитесь, что это правильный файл kaggle.json")
                    except json.JSONDecodeError:
                        st.error("❌ Ошибка чтения JSON файла. Убедитесь, что файл не поврежден")
                    except Exception as e:
                        st.error(f"❌ Ошибка: {str(e)}")
            
            else:  # Ввод вручную
                st.markdown("**Введите данные из файла kaggle.json:**")
                col1, col2 = st.columns(2)
                with col1:
                    kaggle_username = st.text_input("Kaggle Username", value="", help="Ваш username на Kaggle (из файла kaggle.json)")
                with col2:
                    kaggle_key = st.text_input("Kaggle API Key", type="password", value="", help="Ваш API key из файла kaggle.json")
                
                if st.button("💾 Сохранить учетные данные"):
                    if kaggle_username and kaggle_key:
                        from utils import setup_kaggle_api
                        success, message = setup_kaggle_api(kaggle_username, kaggle_key)
                        if success:
                            st.success(message)
                        else:
                            st.error(message)
                    else:
                        st.warning("⚠️ Пожалуйста, введите username и API key")
            
            # Показываем статус текущей настройки
            import os
            from pathlib import Path
            kaggle_json_path = Path.home() / '.kaggle' / 'kaggle.json'
            if kaggle_json_path.exists():
                st.info("✅ Kaggle API уже настроен. Файл найден в системе.")
            else:
                st.warning("⚠️ Kaggle API не настроен. Настройте его выше для скачивания датасетов.")
        
        # Список доступных датасетов
        from utils import get_kaggle_datasets, download_kaggle_dataset
        
        datasets = get_kaggle_datasets()
        
        st.markdown("**Выберите датасет для скачивания:**")
        
        for dataset_name, dataset_info in datasets.items():
            with st.expander(f"📦 {dataset_name} - {dataset_info['description']} ({dataset_info['size']})"):
                st.write(f"**Описание:** {dataset_info['description']}")
                st.write(f"**Размер:** {dataset_info['size']}")
                st.write(f"**Путь:** `{dataset_info['dataset']}`")
                
                # Показываем предупреждение, если требуется принятие правил
                if dataset_info.get('requires_acceptance', False):
                    st.warning(f"⚠️ {dataset_info.get('note', 'Требуется принять правила использования на Kaggle')}")
                
                # Добавляем ссылку на страницу датасета
                # Для соревнований (c/) используем другой URL
                if dataset_info['dataset'].startswith('c/'):
                    dataset_url = f"https://www.kaggle.com/competitions/{dataset_info['dataset'][2:]}"
                else:
                    dataset_url = f"https://www.kaggle.com/datasets/{dataset_info['dataset']}"
                st.markdown(f"🔗 [Открыть страницу датасета на Kaggle]({dataset_url})")
                
                if st.button(f"⬇️ Скачать {dataset_name}", key=f"download_{dataset_name}"):
                    with st.spinner(f"Скачивание {dataset_name}..."):
                        df_downloaded, error = download_kaggle_dataset(dataset_name, dataset_info['dataset'])
                        
                        if df_downloaded is not None:
                            st.session_state['example_df'] = df_downloaded
                            st.success(f"✅ Датасет {dataset_name} успешно загружен! "
                                     f"Размер: {df_downloaded.shape[0]} строк × {df_downloaded.shape[1]} столбцов")
                            st.rerun()
                        else:
                            # Показываем ошибку с форматированием
                            if "Доступ запрещен" in error or "403" in error or "Forbidden" in error.lower():
                                st.error("❌ Доступ запрещен")
                                # Парсим сообщение об ошибке для красивого отображения
                                error_lines = error.split('\n')
                                st.markdown("**📋 Что нужно сделать:**")
                                for line in error_lines:
                                    if line.strip() and not line.startswith("Доступ запрещен"):
                                        if line.strip().startswith("1.") or line.strip().startswith("2.") or line.strip().startswith("3.") or line.strip().startswith("4."):
                                            st.markdown(f"- {line.strip()}")
                                        elif "https://" in line:
                                            st.markdown(f"🔗 {line.strip()}")
                                        elif line.strip().startswith("💡"):
                                            st.info(line.strip())
                                        else:
                                            st.markdown(line.strip())
                                
                                # Добавляем кнопку для открытия страницы датасета
                                st.markdown(f"👉 [Открыть страницу датасета и принять правила]({dataset_url})")
                            elif "авторизации" in error.lower() or "401" in error.lower():
                                st.error(f"❌ {error}")
                                st.info("💡 Проверьте настройки Kaggle API в разделе выше")
                            else:
                                st.error(f"❌ {error}")

