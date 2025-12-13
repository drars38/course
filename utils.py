"""
Общие утилиты для EDA приложения
"""
import pandas as pd
import numpy as np
import streamlit as st
import os
import json
from pathlib import Path


def sample_data_for_plotting(df, max_points=None, use_sampling=True):
    """Выбирает данные для визуализации, если датасет слишком большой"""
    if df is None or df.empty:
        return df
    
    if not use_sampling:
        return df
    
    if max_points is None:
        max_points = 10000
    
    if len(df) <= max_points:
        return df
    
    # Используем случайную выборку
    sampled_df = df.sample(n=max_points, random_state=42)
    return sampled_df


def detect_and_fix_shift(df):
    """Обнаруживает и исправляет сдвиги в данных из-за запятых в значениях"""
    if df is None or df.empty:
        return df, False
    
    # Определяем тип разделителя
    sample_row = df.iloc[0].astype(str).str.cat(sep=' ')
    has_tabs = '\t' in sample_row
    
    fixed = False
    original_shape = df.shape[1]
    
    # Проверяем числовые колонки - если в них текст, возможно есть сдвиг
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Если слишком мало числовых колонок, возможно есть проблема
    if len(numeric_cols) < df.shape[1] * 0.3:
        # Проверяем последние колонки - они должны быть числовыми
        last_cols = df.columns[-5:].tolist()
        for col in last_cols:
            if df[col].dtype == 'object':
                # Пытаемся преобразовать в число
                non_numeric = df[col].astype(str).str.contains(r'[^0-9.\-]', na=False, regex=True)
                if non_numeric.sum() > len(df) * 0.1:  # Более 10% не числовых
                    fixed = True
                    break
    
    # Если обнаружен сдвиг, пытаемся исправить
    if fixed or has_tabs:
        return df, False  # Пока возвращаем как есть, но добавим предупреждение
    
    return df, fixed


def fix_data_shift(df):
    """Проверяет типы данных в первых и последних 15 строках на наличие сдвигов"""
    if df is None or df.empty or df.shape[0] < 30:
        return df, False, None
    
    # Берем первые и последние 15 строк
    first_15 = df.head(15)
    last_15 = df.tail(15)
    
    # Определяем типы данных для каждой колонки в первых и последних строках
    def get_column_type(series):
        """Определяет тип колонки: numeric или text"""
        if series.dtype in [np.int64, np.float64]:
            return 'numeric'
        
        # Пробуем преобразовать в число
        numeric_count = pd.to_numeric(series, errors='coerce').notna().sum()
        numeric_ratio = numeric_count / len(series) if len(series) > 0 else 0
        
        if numeric_ratio > 0.7:  # Более 70% числовых - считаем числовой
            return 'numeric'
        else:
            return 'text'
    
    # Получаем типы для каждой колонки
    first_types = {}
    last_types = {}
    
    for col in df.columns:
        first_types[col] = get_column_type(first_15[col])
        last_types[col] = get_column_type(last_15[col])
    
    # Проверяем несовпадения
    mismatches = []
    for col in df.columns:
        if first_types[col] != last_types[col]:
            mismatches.append({
                'column': col,
                'first_15_type': first_types[col],
                'last_15_type': last_types[col]
            })
    
    # Если есть несовпадения - возвращаем ошибку
    if mismatches:
        error_msg = "⚠️ Обнаружено несовпадение типов данных между первыми и последними строками:\n\n"
        for mm in mismatches:
            error_msg += f"- Колонка '{mm['column']}': первые 15 строк - {mm['first_15_type']}, последние 15 строк - {mm['last_15_type']}\n"
        error_msg += "\nЭто может указывать на сдвиг данных в середине датасета (например, из-за запятых в текстовых полях)."
        return df, False, error_msg
    
    return df, False, None


@st.cache_data
def load_data(uploaded_file, delimiter=None):
    """Загружает данные из файла с обработкой сдвигов"""
    if uploaded_file is not None:
        try:
            # Читаем первые строки для определения разделителя
            content = uploaded_file.read().decode('utf-8')
            uploaded_file.seek(0)
            
            # Если разделитель не указан, определяем автоматически
            if delimiter is None:
                # Считаем количество табуляций и запятых в первых строках
                first_lines = content.split('\n')[:5]
                tab_counts = [line.count('\t') for line in first_lines if line.strip()]
                comma_counts = [line.count(',') for line in first_lines if line.strip()]
                
                avg_tabs = np.mean(tab_counts) if tab_counts else 0
                avg_commas = np.mean(comma_counts) if comma_counts else 0
                
                # Если табуляций больше и они более равномерны - используем табуляцию
                if avg_tabs > avg_commas and avg_tabs > 2:
                    delimiter = '\t'
                elif avg_commas > 2:
                    delimiter = ','
                else:
                    delimiter = '\t'  # По умолчанию табуляция для TSV
            
            # Загружаем данные
            if delimiter == '\t':
                df = pd.read_csv(uploaded_file, sep='\t', encoding='utf-8', on_bad_lines='skip', engine='python')
            else:
                df = pd.read_csv(uploaded_file, sep=delimiter, quotechar='"', encoding='utf-8', on_bad_lines='skip', engine='python')
            
            uploaded_file.seek(0)
            
            # Применяем проверку сдвигов
            df, was_fixed, shift_error = fix_data_shift(df)
            
            return df, shift_error, was_fixed
        except Exception as e:
            try:
                # Пробуем альтернативные кодировки
                uploaded_file.seek(0)
                if delimiter == '\t':
                    df = pd.read_csv(uploaded_file, sep='\t', encoding='latin-1', on_bad_lines='skip', engine='python')
                else:
                    df = pd.read_csv(uploaded_file, sep=delimiter or ',', encoding='latin-1', on_bad_lines='skip', engine='python')
                uploaded_file.seek(0)
                df, was_fixed, shift_error = fix_data_shift(df)
                return df, shift_error, was_fixed
            except Exception as e2:
                return None, f"{str(e)} / {str(e2)}", False
    return None, None, False


def find_target_column(df, numeric_cols, categorical_cols):
    """Находит целевую переменную в датасете"""
    target_col = None
    for col in df.columns:
        if col.lower() in ['survived', 'target', 'label', 'y', 'class']:
            target_col = col
            break
    
    # Если нет явной целевой переменной, используем первый категориальный или числовой
    if target_col is None:
        if categorical_cols:
            target_col = categorical_cols[0]
        elif numeric_cols:
            target_col = numeric_cols[0]
    
    return target_col


@st.cache_data
def compute_correlation_matrix(df, numeric_cols):
    """Кэшированное вычисление корреляционной матрицы"""
    if len(numeric_cols) < 2:
        return None
    return df[numeric_cols].corr()


@st.cache_data
def compute_basic_stats(df, numeric_cols):
    """Кэшированное вычисление базовой статистики"""
    if not numeric_cols:
        return None
    return df[numeric_cols].describe()


@st.cache_data
def compute_value_counts(df, col, top_n=10):
    """Кэшированное вычисление частот значений"""
    return df[col].value_counts().head(top_n)


@st.cache_data
def compute_outliers(df, col):
    """Кэшированное вычисление выбросов"""
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    return Q1, Q3, IQR, lower_bound, upper_bound, outliers


@st.cache_data
def compute_missing_stats(df):
    """Кэшированное вычисление статистики пропусков"""
    missing_data = df.isnull().sum()
    missing_percent = (missing_data / len(df)) * 100
    missing_df = pd.DataFrame({
        'Количество': missing_data,
        'Процент': missing_percent
    })
    return missing_df[missing_df['Количество'] > 0].sort_values('Количество', ascending=False)


# ========== ФУНКЦИИ ДЛЯ РАБОТЫ С KAGGLE ==========

def get_kaggle_datasets():
    """Возвращает список популярных датасетов для скачивания"""
    return {
        'Titanic': {
            'dataset': 'c/titanic',
            'description': 'Классический датасет о пассажирах Титаника (891 строка, 12 столбцов)',
            'size': '~60 KB',
            'requires_acceptance': True,  # Соревнование, требует принятия правил
            'note': '⚠️ Требуется принять правила соревнования на Kaggle'
        },
        'House Prices': {
            'dataset': 'c/house-prices-advanced-regression-techniques',
            'description': 'Предсказание цен на дома (1460 строк, 81 столбец)',
            'size': '~300 KB',
            'requires_acceptance': True,  # Соревнование, требует принятия правил
            'note': '⚠️ Требуется принять правила соревнования на Kaggle'
        },
        'Sales Data': {
            'dataset': 'rohanrao/aisles-and-sales-data',
            'description': 'Данные о продажах продуктов (примерно 10000+ строк)',
            'size': '~500 KB',
            'requires_acceptance': False,
            'note': None
        },
        'Customer Segmentation': {
            'dataset': 'vjchoudhary7/customer-segmentation-tutorial-in-python',
            'description': 'Сегментация клиентов для маркетинга (2000 строк, 8 столбцов)',
            'size': '~50 KB',
            'requires_acceptance': False,
            'note': None
        },
        'Iris': {
            'dataset': 'uciml/iris',
            'description': 'Классический датасет Iris для классификации (150 строк, 5 столбцов)',
            'size': '~5 KB',
            'requires_acceptance': False,
            'note': None
        },
        'Wine Quality': {
            'dataset': 'uciml/red-wine-quality-cortez-et-al-2009',
            'description': 'Качество красного вина (1599 строк, 12 столбцов)',
            'size': '~30 KB',
            'requires_acceptance': False,
            'note': None
        }
    }


def setup_kaggle_api(username=None, api_key=None):
    """Настраивает Kaggle API с учетными данными"""
    try:
        kaggle_dir = Path.home() / '.kaggle'
        kaggle_dir.mkdir(exist_ok=True)
        
        kaggle_json = kaggle_dir / 'kaggle.json'
        
        if username and api_key:
            # Сохраняем учетные данные
            credentials = {
                'username': username,
                'key': api_key
            }
            with open(kaggle_json, 'w') as f:
                json.dump(credentials, f)
            # Устанавливаем правильные права доступа (только для Unix)
            if os.name != 'nt':  # Не Windows
                os.chmod(kaggle_json, 0o600)
            return True, "✅ Kaggle API настроен успешно!"
        elif kaggle_json.exists():
            # Учетные данные уже есть
            return True, "✅ Используются существующие учетные данные Kaggle"
        else:
            return False, "❌ Необходимо указать username и API key"
    except Exception as e:
        return False, f"❌ Ошибка настройки Kaggle API: {str(e)}"


def download_kaggle_dataset(dataset_name, dataset_path):
    """Скачивает датасет с Kaggle"""
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        
        # Инициализируем API
        api = KaggleApi()
        try:
            api.authenticate()
        except Exception as auth_error:
            error_msg = str(auth_error)
            if "401" in error_msg or "Unauthorized" in error_msg:
                return None, "Ошибка авторизации. Проверьте username и API key в настройках Kaggle"
            else:
                return None, f"Ошибка аутентификации Kaggle API: {error_msg}"
        
        # Создаем временную директорию для скачивания
        import tempfile
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                # Определяем, это соревнование или обычный датасет
                if dataset_path.startswith('c/'):
                    # Это соревнование - используем competition_download_files
                    competition_name = dataset_path[2:]
                    api.competition_download_files(competition_name, path=temp_dir, unzip=True)
                else:
                    # Обычный датасет
                    api.dataset_download_files(dataset_path, path=temp_dir, unzip=True)
            except Exception as download_error:
                error_msg = str(download_error)
                if "403" in error_msg or "Forbidden" in error_msg:
                    # Обработка будет в основном блоке except
                    raise download_error
                elif "404" in error_msg or "Not Found" in error_msg:
                    return None, f"Датасет не найден: {dataset_path}. Проверьте правильность пути к датасету."
                else:
                    return None, f"Ошибка скачивания датасета: {error_msg}"
            
            # Ищем CSV файлы в скачанной директории
            csv_files = list(Path(temp_dir).glob('*.csv'))
            
            if not csv_files:
                # Если нет CSV в корне, ищем в подпапках
                csv_files = list(Path(temp_dir).rglob('*.csv'))
            
            if csv_files:
                # Берем первый CSV файл (обычно основной файл датасета)
                # Или файл с именем, похожим на название датасета
                main_file = csv_files[0]
                if len(csv_files) > 1:
                    # Пытаемся найти файл train.csv или файл с названием датасета
                    for f in csv_files:
                        if 'train' in f.name.lower() or dataset_name.lower() in f.name.lower():
                            main_file = f
                            break
                
                # Читаем CSV файл с обработкой ошибок
                try:
                    df = pd.read_csv(main_file, encoding='utf-8')
                except UnicodeDecodeError:
                    # Пробуем другие кодировки
                    try:
                        df = pd.read_csv(main_file, encoding='latin-1')
                    except:
                        df = pd.read_csv(main_file, encoding='cp1252')
                
                if df is not None and not df.empty:
                    return df, None
                else:
                    return None, f"Датасет загружен, но файл {main_file.name} пуст или поврежден"
            else:
                # Показываем, какие файлы были найдены (для отладки)
                all_files = list(Path(temp_dir).rglob('*'))
                file_extensions = [f.suffix for f in all_files if f.is_file()]
                return None, f"Не найдено CSV файлов в датасете. Найдены файлы с расширениями: {set(file_extensions)}"
                
    except ImportError:
        return None, "Библиотека kaggle не установлена. Установите: pip install kaggle"
    except Exception as e:
        error_msg = str(e)
        if "401" in error_msg or "Unauthorized" in error_msg:
            return None, "Ошибка авторизации. Проверьте username и API key в настройках Kaggle"
        elif "403" in error_msg or "Forbidden" in error_msg:
            # Формируем URL датасета для принятия правил
            # Для соревнований (c/) используем другой URL
            if dataset_path.startswith('c/'):
                dataset_url = f"https://www.kaggle.com/competitions/{dataset_path[2:]}"
                competition_name = dataset_path[2:]
            else:
                dataset_url = f"https://www.kaggle.com/datasets/{dataset_path}"
                competition_name = None
            
            error_text = (
                f"Доступ запрещен. Для скачивания этого датасета необходимо принять правила использования на Kaggle.\n\n"
                f"📋 Что делать:\n"
            )
            
            if competition_name:
                error_text += (
                    f"1. Откройте страницу соревнования: {dataset_url}\n"
                    f"2. Нажмите кнопку 'Join Competition' или 'I Understand and Accept'\n"
                    f"3. Примите правила соревнования (обычно требуется подтверждение по email)\n"
                )
            else:
                error_text += (
                    f"1. Откройте страницу датасета: {dataset_url}\n"
                    f"2. Нажмите кнопку 'I Understand and Accept' или 'Accept Rules'\n"
                )
            
            error_text += (
                f"4. После принятия правил попробуйте скачать датасет снова\n\n"
                f"💡 Альтернатива: Вы можете скачать датасет вручную с сайта Kaggle и загрузить через 'Загрузить CSV файл'"
            )
            
            return None, error_text
        elif "404" in error_msg or "Not Found" in error_msg:
            return None, f"Датасет не найден: {dataset_path}. Проверьте правильность пути к датасету."
        else:
            return None, f"Ошибка скачивания: {error_msg}"


# ========== ФУНКЦИИ ДЛЯ ЭКСПОРТА ОТЧЕТОВ ==========

def generate_html_report(df, numeric_cols, categorical_cols, target_col, correlation_matrix=None, vif_data=None, hypotheses=None):
    """Генерирует HTML отчет с результатами анализа"""
    from datetime import datetime
    import base64
    import io
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>EDA Отчет - {datetime.now().strftime('%Y-%m-%d %H:%M')}</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 20px;
                background-color: #f5f5f5;
            }}
            .container {{
                max-width: 1200px;
                margin: 0 auto;
                background-color: white;
                padding: 30px;
                box-shadow: 0 0 10px rgba(0,0,0,0.1);
            }}
            h1 {{
                color: #2c3e50;
                border-bottom: 3px solid #3498db;
                padding-bottom: 10px;
            }}
            h2 {{
                color: #34495e;
                margin-top: 30px;
                border-left: 4px solid #3498db;
                padding-left: 10px;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
            }}
            th, td {{
                border: 1px solid #ddd;
                padding: 12px;
                text-align: left;
            }}
            th {{
                background-color: #3498db;
                color: white;
            }}
            tr:nth-child(even) {{
                background-color: #f2f2f2;
            }}
            .stat-box {{
                background-color: #ecf0f1;
                padding: 15px;
                margin: 10px 0;
                border-radius: 5px;
            }}
            .warning {{
                background-color: #fff3cd;
                border-left: 4px solid #ffc107;
                padding: 10px;
                margin: 10px 0;
            }}
            .success {{
                background-color: #d4edda;
                border-left: 4px solid #28a745;
                padding: 10px;
                margin: 10px 0;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>📊 Отчет EDA анализа</h1>
            <p><strong>Дата создания:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <h2>1. Общая информация о датасете</h2>
            <div class="stat-box">
                <p><strong>Размер датасета:</strong> {df.shape[0]} строк × {df.shape[1]} столбцов</p>
                <p><strong>Числовых признаков:</strong> {len(numeric_cols)}</p>
                <p><strong>Категориальных признаков:</strong> {len(categorical_cols)}</p>
                <p><strong>Целевая переменная:</strong> {target_col if target_col else 'Не определена'}</p>
            </div>
            
            <h2>2. Пропущенные значения</h2>
            <table>
                <tr>
                    <th>Признак</th>
                    <th>Количество пропусков</th>
                    <th>Процент</th>
                </tr>
    """
    
    missing_data = df.isnull().sum()
    missing_percent = (missing_data / len(df)) * 100
    for col in df.columns:
        if missing_data[col] > 0:
            html_content += f"""
                <tr>
                    <td>{col}</td>
                    <td>{missing_data[col]}</td>
                    <td>{missing_percent[col]:.2f}%</td>
                </tr>
            """
    
    html_content += """
            </table>
    """
    
    if correlation_matrix is not None:
        html_content += """
            <h2>3. Корреляционный анализ</h2>
            <p>Корреляционная матрица вычислена для числовых признаков.</p>
        """
        
        if vif_data:
            html_content += """
                <h3>3.1. Анализ мультиколлинеарности (VIF)</h3>
                <table>
                    <tr>
                        <th>Признак</th>
                        <th>VIF</th>
                        <th>Оценка</th>
                    </tr>
            """
            for vif_row in vif_data:
                html_content += f"""
                    <tr>
                        <td>{vif_row['Признак']}</td>
                        <td>{vif_row['VIF']}</td>
                        <td>{vif_row['Оценка']}</td>
                    </tr>
                """
            html_content += """
                </table>
            """
    
    if hypotheses:
        html_content += """
            <h2>4. Сгенерированные гипотезы</h2>
        """
        for i, hyp in enumerate(hypotheses, 1):
            html_content += f"""
                <div class="stat-box">
                    <h3>Гипотеза {i}: {hyp.get('Гипотеза', 'N/A')}</h3>
                    <p><strong>Обоснование:</strong> {hyp.get('Обоснование', 'N/A')}</p>
                    <p><strong>Метод проверки:</strong> {hyp.get('Метод проверки', 'N/A')}</p>
            """
            if 'statistical_test' in hyp and hyp['statistical_test']:
                html_content += f"<p><strong>Статистический тест:</strong><br>{hyp['statistical_test'].replace(chr(10), '<br>')}</p>"
            html_content += "</div>"
    
    html_content += """
        </div>
    </body>
    </html>
    """
    
    return html_content


def generate_pdf_report(df, numeric_cols, categorical_cols, target_col, correlation_matrix=None, vif_data=None, hypotheses=None):
    """Генерирует PDF отчет с результатами анализа"""
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.lib import colors
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_CENTER, TA_LEFT
    from datetime import datetime
    import io
    
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=18)
    
    story = []
    styles = getSampleStyleSheet()
    
    # Заголовок
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#2c3e50'),
        spaceAfter=30,
        alignment=TA_CENTER
    )
    story.append(Paragraph("📊 Отчет EDA анализа", title_style))
    story.append(Paragraph(f"<i>Дата создания: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</i>", styles['Normal']))
    story.append(Spacer(1, 0.5*inch))
    
    # Общая информация
    story.append(Paragraph("1. Общая информация о датасете", styles['Heading2']))
    info_data = [
        ['Параметр', 'Значение'],
        ['Размер датасета', f"{df.shape[0]} строк × {df.shape[1]} столбцов"],
        ['Числовых признаков', str(len(numeric_cols))],
        ['Категориальных признаков', str(len(categorical_cols))],
        ['Целевая переменная', target_col if target_col else 'Не определена']
    ]
    info_table = Table(info_data, colWidths=[3*inch, 3*inch])
    info_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3498db')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(info_table)
    story.append(Spacer(1, 0.3*inch))
    
    # Пропущенные значения
    story.append(Paragraph("2. Пропущенные значения", styles['Heading2']))
    missing_data = df.isnull().sum()
    missing_percent = (missing_data / len(df)) * 100
    missing_table_data = [['Признак', 'Количество пропусков', 'Процент']]
    for col in df.columns:
        if missing_data[col] > 0:
            missing_table_data.append([col, str(missing_data[col]), f"{missing_percent[col]:.2f}%"])
    
    if len(missing_table_data) > 1:
        missing_table = Table(missing_table_data, colWidths=[2.5*inch, 2*inch, 1.5*inch])
        missing_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3498db')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(missing_table)
    else:
        story.append(Paragraph("Пропущенных значений не обнаружено.", styles['Normal']))
    
    story.append(Spacer(1, 0.3*inch))
    
    # VIF анализ
    if vif_data:
        story.append(Paragraph("3. Анализ мультиколлинеарности (VIF)", styles['Heading2']))
        vif_table_data = [['Признак', 'VIF', 'Оценка']]
        for vif_row in vif_data:
            vif_table_data.append([vif_row['Признак'], vif_row['VIF'], vif_row['Оценка']])
        
        vif_table = Table(vif_table_data, colWidths=[2.5*inch, 1.5*inch, 2*inch])
        vif_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3498db')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(vif_table)
        story.append(Spacer(1, 0.3*inch))
    
    # Гипотезы
    if hypotheses:
        story.append(Paragraph("4. Сгенерированные гипотезы", styles['Heading2']))
        for i, hyp in enumerate(hypotheses, 1):
            story.append(Paragraph(f"<b>Гипотеза {i}:</b> {hyp.get('Гипотеза', 'N/A')}", styles['Heading3']))
            story.append(Paragraph(f"<b>Обоснование:</b> {hyp.get('Обоснование', 'N/A')}", styles['Normal']))
            story.append(Paragraph(f"<b>Метод проверки:</b> {hyp.get('Метод проверки', 'N/A')}", styles['Normal']))
            if 'statistical_test' in hyp and hyp['statistical_test']:
                story.append(Paragraph(f"<b>Статистический тест:</b> {hyp['statistical_test']}", styles['Normal']))
            story.append(Spacer(1, 0.2*inch))
    
    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()
