import models

import pandas as pd
from math import floor
from sklearn.metrics import mean_absolute_percentage_error

class Auto_params_selection_Error(Exception):
    """Ошибка, возникающая при автоматическом выборе параметров для базовых функций прогнозирования"""
    pass
class Auto_real_forecast_Error(Exception):
    """Ошибка, возникающая при посторение реального прогноза моделями"""
    
class Filter_Forecast_Error(Exception):
    """Ошибка, возникающая при не подходящая частотность или длина ряда."""
    pass

# Фильтрация рядов для работы с прогнозными моделями. 
def Time_Serial_filter(Data: dict):
    
    """
    Проверяет временной ряд перед прогнозированием: 
    - наличие исключений, 
    - корректность частоты,
    - достаточную длину.
    """
    
    # Проверяем наличие ключа 'exceptions' и что он пуст
    if 'exceptions' in Data and Data['exceptions']:
        raise Filter_Forecast_Error(f"Обнаружены исключения в данных: {Data['exceptions']}")
    
    df = pd.DataFrame(Data['observations'])
    Freq = Data['pandas_frequency']
    
    if Freq == 'YE' and len(df) >= 16:
        return Data
    elif Freq == 'Q' and len(df) > 12:
        return Data
    elif Freq == 'ME' and len(df) > 20: # для работы сезонных моделей длина ряда > 12 (период сезонности) + 40%от длины ряда
        return Data
    else:
        raise Filter_Forecast_Error('Не подходящая частотность или длина ряда.', Freq, len(df))#f'Частотноть ряда: {Freq}.', f'Длина ряда: {len(df)}.')

def Filter_for_dataset(dataset: dict):
    """
    Для применения ко всему датасету данных функции Filter_data_forecast().
    dataset: {key -> Data_dict}
    Возвращает:
      passed: {key -> Data_dict}  — прошли фильтр
      failed: {key -> "reason"}   — не прошли, с описанием ошибки
    """
    passed = []
    failed = []

    for idx, data_item in enumerate(dataset):
        try:
            passed.append(Time_Serial_filter(data_item))
        except Filter_Forecast_Error as e:
            failed.append((idx, str(e)))
        except Exception as e:
            # Подстраховка от неожиданных ошибок
            failed.append((idx, f"{type(e).__name__}: {e}"))

    return passed, failed

# Фукция автоматического подбора параметров для базовых моделей
def Auto_params_selection(Data: dict):
    
    """
    Автоматически подбирает базовые параметры для моделей прогнозирования по данным ряда.

    Функция:
    1) определяет частоту временного ряда (месячная, квартальная, годовая);
    2) рассчитывает:
       - Forecast_horizon — горизонт псевдовневыборочного прогноза,
       - Deep_forecast_period — длину псевдовневыборочного периода,
       - Seasonality — сезонность (при наличии),
       - Window_size — ширину скользящего окна;
    3) использует фиксированные экспертные значения и долю от длины ряда для коротких временных рядов;
    4) проверяет корректность частоты и при невозможности подобрать параметр возбуждает Auto_params_selection_Error.

    Возвращает:
        dict — словарь с ключами 'Forecast_horizon', 'Deep_forecast_period',
               'Seasonality', 'Window_size'.
    """
    
    df = pd.DataFrame(Data['observations'])
    Freq = Data['pandas_frequency']
    
    dic_auto_params = {'Forecast_horizon': [], # Горизонт псевдовневыборочного прогноза
                       'Deep_forecast_period':[], # Длина псевдовневыборочного прогноза
                       'Seasonality': [], # параметр сезонности, учавствующий в уравнении модели (дополнительный снос)
                       'Window_size': []} # ширина скользящего окна
    
    # горизонт прогнозирования от длины дф
    # дописать что 'Forecast horizon' не может быть больше 'Deep forecast period'
    '''Экспертами устанавливаются фиксированные значения для Forecast horizon'''
    '''Для коротких рядов допускается взятие 20% от длины ряда'''
    horizon_values = {'horizon_m': 12,
                      'horizon_q': 4,
                      'horizon_y': 3}
    ratio_horizon_to_len = floor(0.2 * len(df)) # Отношение горизонта псевдовневыборочного прогноза к длине всего ряда. актуально для коротких рядов.
    if Freq == 'ME':
        if ratio_horizon_to_len > 12:
            dic_auto_params['Forecast_horizon'] = horizon_values['horizon_m']   # ряд стандартной длины
        else: 
            dic_auto_params['Forecast_horizon'] = ratio_horizon_to_len     # для коротких рядов
    elif Freq == 'Q':
        if ratio_horizon_to_len > 4:
            dic_auto_params['Forecast_horizon'] = horizon_values['horizon_q']
        else: 
            dic_auto_params['Forecast_horizon'] = ratio_horizon_to_len
    elif Freq == 'YE':
        if ratio_horizon_to_len > 3:
            dic_auto_params['Forecast_horizon'] = horizon_values['horizon_y']
        else: 
            dic_auto_params['Forecast_horizon'] = ratio_horizon_to_len
    else:
        raise Auto_params_selection_Error("Невозможно определить горизонт псевдовневыборочного прогноза")
    
    '''Экспертами устанавливаются фиксированные значения для Deep forecast period'''
    '''Для коротких рядов допускается взятие 40% от длины ряда'''
    
    deep_forecast_values = {'deep_forecast_period_m': 24,
                            'deep_forecast_period_q': 8,
                            'deep_forecast_period_y': 6}
    ratio_deep_forcast_period_to_len = floor(0.4 * len(df)) # Отношение длины псевдовневыборочного прогноза к длине всего ряда. актуально для коротких рядов.
    if Freq == 'ME':
        if ratio_deep_forcast_period_to_len > 24:   # изменено с 12 , так что бы 24 было 40% от ддлины ряда Для других частотнсотей так же
            dic_auto_params['Deep_forecast_period'] = deep_forecast_values['deep_forecast_period_m']
        else: 
            dic_auto_params['Deep_forecast_period'] = ratio_deep_forcast_period_to_len
    elif Freq == 'Q':
        if ratio_deep_forcast_period_to_len > 8:
            dic_auto_params['Deep_forecast_period'] = deep_forecast_values['deep_forecast_period_q']
        else: 
            dic_auto_params['Deep_forecast_period'] = ratio_deep_forcast_period_to_len
    elif Freq == 'YE':
        if ratio_deep_forcast_period_to_len > 6:
            dic_auto_params['Deep_forecast_period'] = deep_forecast_values['deep_forecast_period_y']
        else: 
            dic_auto_params['Deep_forecast_period'] = ratio_deep_forcast_period_to_len
    else:
        raise Auto_params_selection_Error("Невозможно определить длину псевдовневыборочного прогноза")
    
    '''Экспертами устанавливаются фиксированные значения для Seasonality'''
    seasonality_values = {'seasonality_m': 12,  # для годовых рядов отсутствует сезонности
                          'seasonality_q': 4,
                          'seasonality_y' : None}
    if Freq == 'ME':
        dic_auto_params['Seasonality'] = seasonality_values['seasonality_m']
    elif Freq == 'Q':
        dic_auto_params['Seasonality'] = seasonality_values['seasonality_q']
    elif Freq == 'YE':
        dic_auto_params['Seasonality'] = seasonality_values['seasonality_y']

    '''Экспертами устанавливаются фиксированные значения для Windowsize'''      # редактировать параметры
    windowsize_values = {'windowsize_m': 36,
                         'windowsize_q': 12,
                         'windowsize_y': 3}
    ratio_windowsize_to_len = floor(0.2 * len(df)) # Отношение ширины скользящего окна псевдовневыборочного прогноза к длине всего ряда. актуально для коротких рядов.
    if Freq == 'ME':
        if ratio_windowsize_to_len > 12:
            dic_auto_params['Window_size'] = windowsize_values['windowsize_m']
        else: 
            dic_auto_params['Window_size'] = ratio_windowsize_to_len
    elif Freq == 'Q':
        if ratio_windowsize_to_len > 4:
            dic_auto_params['Window_size'] = windowsize_values['windowsize_q']
        else: 
            dic_auto_params['Window_size'] = ratio_windowsize_to_len
    elif Freq == 'YE':
        if ratio_windowsize_to_len > 3:
            dic_auto_params['Window_size'] = windowsize_values['windowsize_y']
        else: 
            dic_auto_params['Window_size'] = ratio_windowsize_to_len
    else:
        raise Auto_params_selection_Error("Невозможно определить ширину скользящего окна")
    return dic_auto_params

# Функция подсчета усредненных  MAPE по шагам
def MAPE_step_by_step(Data: dict,
                      Dataframe_model: pd.DataFrame,
                      Deep_forecast_period: int,
                      Forecast_horizon: int):
    
    """
    Вычисляет усреднённые пошаговые значения MAPE для прогнозной модели.

    Функция:
    1) формирует реальные и модельные значения для каждого шага псевдовневыборочного периода;
    2) рассчитывает MAPE отдельно для каждого шага горизонта;
    3) возвращает список ошибок, где i-й элемент — MAPE на i-м шаге.

    Возвращает:
        list — значения MAPE длиной Forecast_horizon.
    """
    
    df_real = pd.DataFrame(Data['observations'])
    df_real.obs = df_real.obs.astype(float) # значения переводятся в формат float
    
    df_model = Dataframe_model.copy()
    
    train_period =  len(df_real.obs) - Deep_forecast_period
    quantity_pseudo_foracasts = Deep_forecast_period - Forecast_horizon + 1
    
    Real_Data_list = [df_real.obs[train_period+i:train_period+quantity_pseudo_foracasts+i] for i in range(Forecast_horizon)]
    #print(Real_Data_list[0])
    Model_Data_list = [df_model.iloc[:,i] for i in range(Forecast_horizon)]
    #print(Model_Data_list[0])
    
    Errors = []
    for i in range(Forecast_horizon):
        Errors.append(round(mean_absolute_percentage_error(Real_Data_list[i], Model_Data_list[i]), 2))
    
    return Errors

# Функция определения минимального MAPE на каждом шаге и формирование списка с "лучшими" моделями для каждого шага прогнозирования
def Psevdo_forecast_test_MAPE(Data: dict,
                              Deep_forecast_period: int,
                              Forecast_horizon: int,
                              Seasonality: int,
                              Reassessment: bool):
    
    """
    Определяет модель с минимальным MAPE на каждом шаге псевдовневыборочного прогноза.

    Функция:
    1) строит псевдопрогнозы для всех доступных моделей;
    2) вычисляет пошаговый MAPE для каждой модели;
    3) на каждом шаге горизонта выбирает модель с минимальной ошибкой;
    4) формирует и возвращает список моделей, оптимальных для каждого шага.

    Возвращает:
        list — последовательность имён моделей длиной Forecast_horizon.
    """
    
    Freq = Data['pandas_frequency']
    
    Window_in_years = None # задел на будущее
    
    if Freq == 'YE' or (len(Data) < 20 and Freq == 'ME'):
        Forecast_dict_Y = {'RW' : models.ps_RW(Data, Deep_forecast_period, Forecast_horizon),            # строим псевдо прогноз для всех моделей
                           'RWD' : models.ps_RWD(Data, Deep_forecast_period, Forecast_horizon, Window_in_years),
                           'TS' : models.ps_TS(Data, Deep_forecast_period, Forecast_horizon, Window_in_years, Reassessment),
                           'ARIMA' : models.ps_ARIMA(Data, Deep_forecast_period, Forecast_horizon, Window_in_years, Reassessment),
                           'custom_ARIMA' : models.ps_custom_ARIMA(Data, Deep_forecast_period, Forecast_horizon, Reassessment)
                        }
        
        Erorr_dict_Y = {k : MAPE_step_by_step(Data,
                                        Forecast_dict_Y[k],
                                        Deep_forecast_period,
                                        Forecast_horizon)
                        for k in Forecast_dict_Y.keys()
                        }
    
        List_of_model_number_Y = [min(Erorr_dict_Y, key=lambda key: Erorr_dict_Y[key][i]) for i in range(Forecast_horizon)] # функция min() примененная к словарю обращается к его ключам. С помощью lambda функции выбирается сначала только первые элементы, затем только вторые и тд. затем функция min() выбирает минимальный элемент среди первых, затем среди вторых и тд, когда выбирается миниму, выражение возвращает ключ в котором он содержится
        return List_of_model_number_Y
    
    else:
        Forecast_dict = {'RW' : models.ps_RW(Data, Deep_forecast_period, Forecast_horizon),            # строим псевдо прогноз для всех моделей
                         'RWS' : models.ps_RWS(Data, Deep_forecast_period, Forecast_horizon, Seasonality),
                         'RWD' : models.ps_RWD(Data, Deep_forecast_period, Forecast_horizon, Window_in_years),
                         'RWDS' : models.ps_RWDS(Data, Deep_forecast_period, Forecast_horizon, Seasonality, Window_in_years),
                         'TS' : models.ps_TS(Data, Deep_forecast_period, Forecast_horizon, Window_in_years, Reassessment),
                         'ARIMA' : models.ps_ARIMA(Data, Deep_forecast_period, Forecast_horizon, Window_in_years, Reassessment),
                         'custom_ARIMA' : models.ps_custom_ARIMA(Data, Deep_forecast_period, Forecast_horizon, Reassessment)
                        }
        
        Erorr_dict = {k : MAPE_step_by_step(Data,
                                        Forecast_dict[k],
                                        Deep_forecast_period,
                                        Forecast_horizon)
                        for k in Forecast_dict.keys()
                        }

        List_of_model_number = [min(Erorr_dict, key=lambda key: Erorr_dict[key][i]) for i in range(Forecast_horizon)] # функция min() примененная к словарю обращается к его ключам. С помощью lambda функции выбирается сначала только первые элементы, затем только вторые и тд. затем функция min() выбирает минимальный элемент среди первых, затем среди вторых и тд, когда выбирается миниму, выражение возвращает ключ в котором он содержится
        return List_of_model_number
    
# Функция Автоматического прогноза - Конструктор моделей
def Auto_forecast(Data : dict,
                  Deep_research : bool=True):
    """
    Автоматически строит комбинированный прогноз временного ряда.

    Функция:
    1) преобразует входные данные и определяет частоту ряда;
    2) автоматически подбирает параметры прогноза (горизонт, сезонность и др.);
    3) выполняет псевдовневыборочный тест (MAPE) и выбирает лучшие модели;
    4) пошагово формирует прогноз, используя оптимальную модель для каждого шага;
    5) собирает итоговый результат (даты, шаги, модели, значения) в DataFrame.

    Возвращает:
        pandas.DataFrame — прогноз по шагам с указанием используемых моделей.
    """
    df = pd.DataFrame(Data['observations'])
    Freq = Data['pandas_frequency']

    # Создается 2 словоря available_models и model_args, это делается для оптимизации работы функции.
    # Для построени комбинированного по шагам прогноза будут задействованы только необходимые функции,
    # которые получаются в результате работы функции Psevdo_forecast_test_MAPE. 
    
    # Функции для построения реального прогноза
    available_models = {
        'RW': models.RW_real,
        'RWS': models.RWS_real,
        'RWD': models.RWD_real,
        'RWDS': models.RWDS_real,
        'TS': models.TS_real,
        'ARIMA': models.ARIMA_real,
        'custom_ARIMA': models.custom_ARIMA_real
    }
    # Параметры функций для построения реального прогноза
    model_args = {
        'RW': ['Data', 'Forecast_horizon'],
        'RWS': ['Data', 'Forecast_horizon', 'Seasonality'],
        'RWD': ['Data', 'Forecast_horizon'],
        'RWDS': ['Data', 'Forecast_horizon', 'Seasonality'],
        'TS': ['Data', 'Forecast_horizon'],
        'ARIMA': ['Data', 'Forecast_horizon'],
        'custom_ARIMA': ['Data', 'Forecast_horizon']
    }
    # dic_auto_params - Словарь с автоматически продобранными значениями базовых прогнозных функций. 
    dic_auto_params = Auto_params_selection(Data)

    # Для ускорения будем выключать переоценку моделей при построение псевдовневыборочного прогноза
    #if Deep_research == True:
    #    Reassessment = False
    #else:
    #    Reassessment = True
    #print(Reassessment)
    
    # List_of_model_number - Список с результатами по псевдовневыборочному тесту 
    # то есть, список моделей, которые участвуют в построение реального прогноза.
    List_of_model_number = Psevdo_forecast_test_MAPE(Data = Data,
                                                     Deep_forecast_period = dic_auto_params['Deep_forecast_period'],
                                                     Forecast_horizon = dic_auto_params['Forecast_horizon'],
                                                     Seasonality = dic_auto_params['Seasonality'], 
                                                     Reassessment = Deep_research)
    
    # Конструирование реального прогноза по шагам прогнозирования. 
    Forecast, Model_name, Steps = [], [], []    # Для записи результатов
    
    param_pool = {'Data': Data, **dic_auto_params}

    for i, model_name in enumerate(List_of_model_number):
        model_func = available_models[model_name]
        required_args = model_args[model_name]

        kwargs = {arg: param_pool[arg] for arg in required_args}
        result = model_func(**kwargs)

        idx = min(i, len(result) - 1)  # «прилипает» к последнему элементу
        Forecast.append(result.iloc[idx].round(2))
        
        Model_name.append(List_of_model_number[i])
        Steps.append(f'Горизонт {i+1}')
        
    
    
    # Создание временной даты соответсвующей прогнозным значениям
    df["date"] = pd.to_datetime(df["date"], dayfirst=True, format="%d.%m.%Y")
    Date = pd.date_range(df.iloc[-1,0] + (df.iloc[-1,0] - df.iloc[-2,0]),
                         periods = dic_auto_params['Forecast_horizon'],
                         freq = Freq
                         ).strftime('%d.%m.%Y').tolist()
    
    # Записываем итоговый результат в виде датафрейма
    results = pd.DataFrame({
    'Дата': Date,
    'Шаги': Steps,
    'Модель': Model_name,
    'Прогноз': Forecast
    })
    
    return results