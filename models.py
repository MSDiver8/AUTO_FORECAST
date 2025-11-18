import pandas as pd
import statsmodels.formula.api as smf
import pmdarima as pmd
from copy import deepcopy
from get_arima import GetARIMA
import statsmodels.tsa.arima.model as sma

class Psevdo_Forecast_Error(Exception):
    """Ошибка, возникающая при посторение псевдо прогноза моделями"""
    pass

class Real_Forecast_Error(Exception):
    """Ошибка, возникающая при посторение реального прогноза моделями"""
    pass

# Модели для псевдовневыборочного прогноза
def ps_RW(Data: dict,
      Deep_forecast_period: int,
      Forecast_horizon: int):
    
    '''
    Функция строит псевдовневыборочный наивный прогноз.
    Наивный прогноз (модель случайного блуждания, RW). Прогноз в момент времени T - H на h = 1, ..., n шагов, n - горизонт прогнозирования, 
    будет равен значению показателя в этот момент времени и т.д.
    Возвращает:
    pandas.DatsFrame — прогноз длиной Forecast_horizon с датами в индексе.
    '''

    df = pd.DataFrame(Data['observations'])
    df.obs = df.obs.astype(float) # значения переводятся в формат float
    base_period =  len(df['obs']) - Deep_forecast_period  # базовый период, на котором оценивается модель
    quantity_pseudo_forecasts = Deep_forecast_period - Forecast_horizon + 1 # количество псевдовневыборочных прогнозов
    forecast_table = [] # двумерный массив прогнозов
    for i in range(quantity_pseudo_forecasts):
        forecast_table.append([df.iloc[base_period - 1 + i, 1] for _ in range(Forecast_horizon)]) # из дф берется последняя известная точка, прогноз - значения этой точки, итог массив длиной forecast_horizon с одним и тем же числом.
    ## групировка по шагам
    steps_table=[] # таблица прогнозов по шагам. каждому столбцу соответсвует номер шага
    for i in range(Forecast_horizon):
            steps_table.append([forecast_table[j][i] for j in range(quantity_pseudo_forecasts)])
            
    dic = { f's{i+1}' : steps_table[i] for i in range(Forecast_horizon)} # записываем в словарь
    df = pd.DataFrame.from_dict(dic)
    return(df)
def ps_RWS(Data: dict,
           Deep_forecast_period: int,
           Forecast_horizon: int,
           Seasonality : int):

    '''
    Функция строит псевдовневыборочный наивный сезонный прогноз.
    Наивный сезонный прогноз (модель сезонного случайного блуждания, RWS). Прогноз в момент времени T - H на h = 1, ...,  n шагов, n - горизонт прогнозирования, 
    будет равен значению показателя в этот момент времени минус коэфицент сезонности и т.д.
    Возвращает:
    pandas.DatsFrame — прогноз длиной Forecast_horizon с датами в индексе.
    '''

    df = pd.DataFrame(Data['observations'])
    df.obs = df.obs.astype(float) # значения переводятся в формат float
    base_period =  len(df['obs']) - Deep_forecast_period # длина базового периода
    quantity_pseudo_foracasts = Deep_forecast_period - Forecast_horizon + 1 # количество псевдовневыборочных прогнозов  
    
    forecast_table = [] # двумерный массив прогнозов
    for i in range(quantity_pseudo_foracasts):
        forecast_table.append(df.iloc[base_period - Seasonality + i : base_period - Seasonality + i + Forecast_horizon, 1].to_list()) # из дф берется масив с данными смещенными на один месяц, так для каждого момента прогнозирования
    
    # групировка по шагам
    steps_table=[]
    for i in range(Forecast_horizon):                                                
            steps_table.append([forecast_table[j][i] for j in range(quantity_pseudo_foracasts)])
    #print(s)
    dic = { f's{i+1}' : steps_table[i] for i in range(Forecast_horizon)}
    df = pd.DataFrame.from_dict(dic)
    
    return(df)
def ps_RWD(Data: dict,
       Deep_forecast_period: int,
       Forecast_horizon: int,
       Window_in_years: int = None
       ):

    '''
    Функция строит псевдовневыборочный наивный прогноз с дрейфом.
    Наивный прогноз c дрейфом (модель случайного блуждания с дрейфом, RWD). Прогноз в момент времени T - H на h = 1, ..., n шагов, n - горизонт прогнозирования, 
    будет равен значению показателя в этот момент времени плюс коэфицент "дрейфа" и т.д.
    Возвращает:
    pandas.DatsFrame — прогноз длиной Forecast_horizon с датами в индексе.
    '''

    df = pd.DataFrame(Data['observations'])
    df.obs = df.obs.astype(float) # значения переводятся в формат float
    df["date"] = pd.to_datetime(df["date"], dayfirst=True, format="%d.%m.%Y")
    df.index = df['date'] # Индекс дата
    df = df.drop('date', axis = 1)
    base_period =  len(df['obs']) - Deep_forecast_period  
    quantity_pseudo_foracasts = Deep_forecast_period - Forecast_horizon + 1                 # количество псевдовневыборочных прогнозов
    # составление двумерного массива ркурсивного прогноза 
    if Window_in_years == None or Window_in_years == 0:                                                                             
        forecast_table = []
        const_RWD_r = df['obs'].diff().mean()
        for i in range(quantity_pseudo_foracasts):
                forecast_table.append([round(df.iloc[base_period - 1 + i, 0] + (j + 1) * const_RWD_r, 2) for j in range(Forecast_horizon)])                         # формула модели RWD, заполняется масив для каждого момента прогнозирования. Затем это записывается в общий масив прогнозов.
    
    # скользящее окно       
    else:   
                                                                                                          
        if Window_in_years < 2:
            raise Psevdo_Forecast_Error('Окно слишком мало')
        window = 12 * Window_in_years                                                                                    
        forecast_table = []      
        for i in range(quantity_pseudo_foracasts):
                const_RWD_w = df['obs'][base_period - window + i:base_period + i].diff().mean()
                forecast_table.append([round(df.iloc[base_period - 1 + i, 0] + (j + 1) * const_RWD_w, 2) for j in range(Forecast_horizon)])      
    
    # групировка по шагам
    steps_table=[]
    for i in range(Forecast_horizon):                                                
            steps_table.append([forecast_table[j][i] for j in range(quantity_pseudo_foracasts)])
    dic = { f's{i+1}' : steps_table[i] for i in range(Forecast_horizon)}
    df = pd.DataFrame.from_dict(dic)
    
    return(round(df,2))
def ps_RWDS(Data: dict,
            Deep_forecast_period: int,
            Forecast_horizon: int,
            Seasonality : int = 0,
            Window_in_years: int = None):
    
    '''
    Функция строит псевдовневыборочный наивный сезонный прогноз с дрейфом.
    Наивный сезонный прогноз c дрейфом (модель случайного сезонного блуждания с дрейфом, RWDS). Прогноз в момент времени T - H на h = 1, ..., n шагов, n - горизонт прогнозирования, 
    будет равен значению показателя в этот момент времени мину сезоннный сдвиг и плюс коэфицент "дрейфа" и т.д.
    Возвращает:
    pandas.DatsFrame — прогноз длиной Forecast_horizon с датами в индексе.
    '''
    
    df = pd.DataFrame(Data['observations'])
    df.obs = df.obs.astype(float) # значения переводятся в формат float
    base_period =  len(df['obs']) - Deep_forecast_period     # длина базового периода
    quantity_pseudo_foracasts = Deep_forecast_period - Forecast_horizon + 1 # количество псевдовневыборочных прогнозов
                                                                                    
    forecast_list = []   # список прогнозов
    
    # составление двумерного массива ркурсивного прогноза 
    if Window_in_years == None or Window_in_years == 0:  
        for i in range(quantity_pseudo_foracasts):
            forecast_list.append(
                [
                round(
                    df.iloc[base_period + j - Seasonality - 1 + i, 1] +
                    df.iloc[:base_period + j, 1].diff(Seasonality).mean(),
                    2)
                for j in range(Forecast_horizon)
                ]
            )
    
    # скользящее окно       
    else:                                                                                                          
        if Window_in_years < 2:
            raise Psevdo_Forecast_Error('Окно слишком мало')
        else:
            # !! добавить параметры для других частотностей 
            window = 12 * Window_in_years                                                                                    
            for i in range(quantity_pseudo_foracasts):
                forecast_list.append(
                    [
                    round(
                        df.iloc[base_period + j - Seasonality + 1 + i, 1] +
                        df.iloc[base_period + j - window:base_period + j, 1].diff(Seasonality).mean(),
                        2
                        )
                    for j in range(Forecast_horizon)
                    ]
                )
            
    # групировка по шагам
    steps_list=[]
    for i in range(Forecast_horizon):                                                
            steps_list.append([forecast_list[j][i] for j in range(quantity_pseudo_foracasts)])
    #print(s)
    dic = { f's{i+1}' : steps_list[i] for i in range(Forecast_horizon)}
    df = pd.DataFrame.from_dict(dic)
    
    return df
def ps_TS(Data: dict,
          Deep_forecast_period: int,
          Forecast_horizon: int,
          Window_in_years: int = None,
          Reassessment: bool = False
          ):
    
    '''
    Функция строит псевдовневыборочный прогноз используя модель линейного тренда TS.
    
    Возвращает:
    pandas.DatsFrame — прогноз длиной Forecast_horizon с датами в индексе.
    '''
    
    df = pd.DataFrame(Data['observations'])
    df['T'] = [i + 1 for i in range(len(df['obs']))]
    df.obs = df.obs.astype(float)
    base_period =  len(df) - Deep_forecast_period  
    quantity_pseudo_foracasts = Deep_forecast_period - Forecast_horizon + 1                 # количество псевдовневыборочных прогнозов
                                                                                  
    forecast_list = []
    
    if Window_in_years == None or Window_in_years == 0:  
        if Reassessment == True:
            for i in range(quantity_pseudo_foracasts):
                alpha = smf.ols('obs ~ T', data=df.iloc[:base_period + i,[1, 2]]).fit().params['Intercept']     # рассчитываем константу 1
                betta = smf.ols('obs ~ T', data=df.iloc[:base_period + i,[1, 2]]).fit().params['T']     # рассчитываем константу 2
                forecast_list.append([round(alpha + df.iloc[base_period + i + j, 2] * betta, 2) for j in range(Forecast_horizon)]) #посмотреть
        else:
            alpha = smf.ols('obs ~ T', data=df.iloc[:base_period,[1, 2]]).fit().params['Intercept']     # рассчитываем константу 1
            betta = smf.ols('obs ~ T', data=df.iloc[:base_period,[1, 2]]).fit().params['T']     # рассчитываем константу 2
            for i in range(quantity_pseudo_foracasts):
                forecast_list.append([round(alpha + df.iloc[base_period + i + j, 2] * betta, 2) for j in range(Forecast_horizon)]) #посмотреть
            
        
    else:
        # добавить для других частотностей
        if Window_in_years < 2:
            raise Psevdo_Forecast_Error('Окно слишком мало')
        
        window = 12 * Window_in_years                                                                                    
        for i in range(quantity_pseudo_foracasts):
            alpha = smf.ols('obs ~ T', data=df.iloc[base_period - window + i:base_period + i,[1, 2]]).fit().params['Intercept']    # рассчитываем константу 1
            betta = smf.ols('obs ~ T', data=df.iloc[base_period - window + i:base_period + i,[1, 2]]).fit().params['T']     # рассчитываем константу 2
            forecast_list.append([round(alpha + df.iloc[base_period + i + j, 2] * betta, 2) for j in range(Forecast_horizon)]) 
    
    
    # групировка по шагам
    steps_list=[]
    for i in range(Forecast_horizon):                                                
            steps_list.append([forecast_list[j][i] for j in range(quantity_pseudo_foracasts)])
    #print(s)
    dic = { f's{i+1}' : steps_list[i] for i in range(Forecast_horizon)}
    df = pd.DataFrame.from_dict(dic)
    df = round(df,2)
    return(df)
def ps_ARIMA(Data: dict,
             Deep_forecast_period: int,
             Forecast_horizon: int,
             Window_in_years: int = None,
             Reassessment: bool = False):
    
    '''
    Функция строит псевдовневыборочный прогноз используя модель ARIMA.
    Подбор параметров осуществляется с помощью функции pmd.auto_arima(),
    на основание информационного критерия Шварца. 
    
    Возвращает:
    pandas.DatsFrame — прогноз длиной Forecast_horizon с датами в индексе.
    '''
    
    df = pd.DataFrame(Data['observations'])
    Freq = Data['pandas_frequency']
    df.obs = df.obs.astype(float) # значения переводятся в формат float
    df["date"] = pd.to_datetime(df["date"], dayfirst=True, format="%d.%m.%Y")
    df.index = df['date'] # Индекс дата
    df = df.drop('date', axis = 1)
    df = df.asfreq(Freq) # Установка частотности
    base_period =  len(df['obs']) - Deep_forecast_period  
    quantity_pseudo_foracasts = Deep_forecast_period - Forecast_horizon + 1                 # количество псевдовневыборочных прогнозов
                                                                               
    forecast_table = []
    
    if Window_in_years == None or Window_in_years == 0:
        if Reassessment == True:
            for i in range(quantity_pseudo_foracasts):
                forecast_table.append(pmd.auto_arima(df.obs[:base_period + i],
                                                     information_criterion = 'bic',
                                                     max_p = 5,
                                                     max_d = 2,
                                                     max_q = 5,
                                                     error_action="ignore",
                                                     stepwise=True).predict(Forecast_horizon).tolist())
        else: 
            params = pmd.auto_arima(df.obs[:base_period],
                                     information_criterion = 'bic',
                                     max_p = 5,
                                     max_d = 2,
                                     max_q = 5,
                                     error_action="ignore",
                                     stepwise=True).order
            for i in range(quantity_pseudo_foracasts):
                forecast_table.append(list(sma.ARIMA(df.obs[:base_period + i], order=params)
                                           .fit()
                                           .get_forecast(steps=Forecast_horizon)
                                           .predicted_mean))
                
            
    else:
        # добавить для других частотностей
        if Window_in_years < 2:
            raise Psevdo_Forecast_Error('Окно слишком мало')
        
        window = 12 * Window_in_years
        for i in range(quantity_pseudo_foracasts):
            forecast_table.append(pmd.auto_arima(df.obs[base_period - window + i:base_period + i],
                                                 information_criterion = 'bic',
                                                 max_p = 5,
                                                 max_d = 2,
                                                 max_q = 5,
                                                 error_action="ignore",
                                                 stepwise=True).predict(Forecast_horizon).tolist())
    
    # групировка по шагам
    steps_table=[]
    for i in range(Forecast_horizon):                                                
            steps_table.append([forecast_table[j][i] for j in range(quantity_pseudo_foracasts)])
    #print(s)
    dic = { f's{i+1}' : steps_table[i] for i in range(Forecast_horizon)}
    df = pd.DataFrame.from_dict(dic)
    df = round(df,2)
    
    return df
def ps_custom_ARIMA(Data: dict,
                    Deep_forecast_period: int,
                    Forecast_horizon: int,
                    Reassessment: bool = False):
    
    '''
    Функция строит псевдовневыборочный прогноз используя модель custom_ARIMA.
    Подбор параметров осуществляется экспертным автоматизированным методом. 
    
    Возвращает:
    pandas.DatsFrame — прогноз длиной Forecast_horizon с датами в индексе.
    '''
    
    base_period = Data["observations_count"] - Deep_forecast_period
    n_forecasts = Deep_forecast_period - Forecast_horizon + 1  # количество псевдовневыборочных прогнозов

    y = pd.DataFrame(Data["observations"])
    y.index = pd.to_datetime(y.date, format="%d.%m.%Y")
    y.obs = y.obs.apply(float)
    y.drop(["date"], axis=1, inplace=True)

    forecast_table = []

    # Для ускорения введем опцию для выбора осуществвлять переоценку параметров или нет
    if Reassessment == True:
        for i in range(n_forecasts):
            _data = deepcopy(Data)
            _data["observations"] = _data["observations"][: (base_period + i)]
            _data["observations_count"] = base_period + i

            estimated_model = GetARIMA(_data).fit(criteria="bic")[1].fit()
            #print(estimated_model)
            forecast_table.append(
                estimated_model.get_forecast(steps=Forecast_horizon).predicted_mean.tolist()
            )

    else:
        _data = deepcopy(Data)
        _data["observations"] = _data["observations"][:base_period]
        _data["observations_count"] = base_period
        GA = GetARIMA(_data)
        model = GA.fit(criteria="bic")[1]
        
        for i in range(n_forecasts):
            if i == 0:
                estimated_model = model.fit()
            else:
                estimated_model = estimated_model.append([y.obs.iloc[base_period + i - 1]])

            forecast_table.append(
                estimated_model.get_forecast(steps=Forecast_horizon).predicted_mean.tolist()
            )
    # групировка по шагам
    steps_table = []
    for i in range(Forecast_horizon):
        steps_table.append([forecast_table[j][i] for j in range(n_forecasts)])
    # print(s)
    dic = {f"s{i + 1}": steps_table[i] for i in range(Forecast_horizon)}
    df = pd.DataFrame.from_dict(dic)
    df = round(df, 2)

    return df


# Модели для реального прогноза
def RW_real(Data: dict,
            Forecast_horizon: int):
    
    '''
    Функция строит реальный наивный прогноз.
    Наивный прогноз (модель случайного блуждания, RW). Прогноз в момент времени T   H на h = 1, ..., 12 шагов,
    будет равен значению показателя в этот момент времени и т.д.
    Возвращает:
    pandas.Series — прогноз длиной Forecast_horizon с датами в индексе.
    '''

    df = pd.DataFrame(Data['observations'])
    Freq = Data['pandas_frequency']
    
    df.obs = df.obs.astype(float) # значения переводятся в формат float
    RW_forecast_list = [df.iloc[-1,1] for _ in range(Forecast_horizon)] # из дф берется последняя известная точка, прогноз - значения этой точки, итог массив длиной forecast_horizon с одним и тем же числом.
    
    # Создание временной даты соответсвующей прогнозным значениям
    df["date"] = pd.to_datetime(df["date"], dayfirst=True, format="%d.%m.%Y")
    Date = pd.date_range(df.iloc[-1,0] + (df.iloc[-1,0] - df.iloc[-2,0]),
                         periods = Forecast_horizon,
                         freq = Freq
                         )
    
    results = pd.Series(data = RW_forecast_list,
                        index = Date,
                        name = 'predicted')
    
    return(results)
def RWS_real(Data: dict,       
             Forecast_horizon: int,
             Seasonality: int):
    
    '''
    Функция строит реальный наивный сезонный прогноз.
    Наивный сезонный прогноз (модель сезонного случайного блуждания, RWS). Прогноз в момент времени T - H на h = 1, ...,  n шагов, n - горизонт прогнозирования, 
    будет равен значению показателя в этот момент времени минус коэфицент сезонности и т.д.
    Возвращает:
    pandas.DatsFrame — прогноз длиной Forecast_horizon с датами в индексе.
    '''
    
    if Seasonality < Forecast_horizon:
        raise Real_Forecast_Error('Seasonality < Forecast_horizon, Функция подходит только для горизонта прогноза не привышающего величину сезонного сдвига')
    
    df = pd.DataFrame(Data['observations'])
    Freq = Data['pandas_frequency']
    df.obs = df.obs.astype(float) # значения переводятся в формат float 
    
    RWS_forecast_list = [df.iloc[- Seasonality + i ,1] for i in range(Forecast_horizon)] # из дф берется масив с данными смещенными на один месяц, так для каждого момента прогнозирования
    # Создание временной даты соответсвующей прогнозным значениям
    df["date"] = pd.to_datetime(df["date"], dayfirst=True, format="%d.%m.%Y")
    Date = pd.date_range(df.iloc[-1,0] + (df.iloc[-1,0] - df.iloc[-2,0]),
                         periods = Forecast_horizon,
                         freq = Freq
                         )
    
    results = pd.Series(data = RWS_forecast_list,
                        index = Date,
                        name = 'predicted')
    
    return(results)
def RWD_real(Data: dict,
         Forecast_horizon: int,
         Window_in_years: int = None
                 ):

    '''
    Функция строит реальный наивный прогноз с дрейфом.
    Наивный прогноз c дрейфом (модель случайного блуждания с дрейфом, RWD). Прогноз в момент времени T - H на h = 1, ..., n шагов, n - горизонт прогнозирования, 
    будет равен значению показателя в этот момент времени плюс коэфицент "дрейфа" и т.д.
    Возвращает:
    pandas.DatsFrame — прогноз длиной Forecast_horizon с датами в индексе.
    '''

    df = pd.DataFrame(Data['observations'])
    Freq = Data['pandas_frequency']
    df.obs = df.obs.astype(float) # значения переводятся в формат float
                                                                     
    # рекурсивный
    if Window_in_years == None or Window_in_years == 0:      
        const_RWD_r = df['obs'].diff().mean()
        RWD_forecast_list = [round(df.iloc[-1, 1] + (j + 1) * const_RWD_r, 2) for j in range(Forecast_horizon)]    # формула модели RWD
        
        # Создание временной даты соответсвующей прогнозным значениям
        df["date"] = pd.to_datetime(df["date"], dayfirst=True, format="%d.%m.%Y")
        Date = pd.date_range(df.iloc[-1,0] + (df.iloc[-1,0] - df.iloc[-2,0]),
                             periods = Forecast_horizon,
                             freq = Freq
                             )
        results = pd.Series(data = RWD_forecast_list,
                            index = Date,
                            name = 'predicted')

        return(results)
    # скользящее окно       
    else:                                                                                                          
        if Window_in_years < 2:
            return print('слишком маленькое окно')
        # !! добавить параметры для других частотностей
        window = 12 * Window_in_years 
        const_RWD_w = df['obs'][ - window:].diff().mean() # оценка константы происходит только на последних известных значениях попадающих в окно
        RWD_forecast_list = [round(df.iloc[-1, 1] + (j + 1) * const_RWD_w, 2) for j in range(Forecast_horizon)] 
            # Создание временной даты соответсвующей прогнозным значениям
        df["date"] = pd.to_datetime(df["date"], dayfirst=True, format="%d.%m.%Y")
        Date = pd.date_range(df.iloc[-1,0] + (df.iloc[-1,0] - df.iloc[-2,0]),
                             periods = Forecast_horizon,
                             freq = Freq
                             )
        results = pd.Series(data = RWD_forecast_list,
                            index = Date,
                            name = 'predicted')
        return(results)
def RWDS_real(Data: dict,
              Forecast_horizon: int,
              Seasonality: int = 0,
              Window_in_years: int = None):
    
    '''
    Функция строит реальныйнаивный сезонный прогноз с дрейфом.
    Наивный сезонный прогноз c дрейфом (модель случайного сезонного блуждания с дрейфом, RWDS). Прогноз в момент времени T - H на h = 1, ..., n шагов, n - горизонт прогнозирования, 
    будет равен значению показателя в этот момент времени мину сезоннный сдвиг и плюс коэфицент "дрейфа" и т.д.
    Возвращает:
    pandas.DatsFrame — прогноз длиной Forecast_horizon с датами в индексе.
    '''
    
    if Seasonality < Forecast_horizon:
        raise Real_Forecast_Error('Seasonality < Forecast_horizon, Функция подходит только для горизонта прогноза не привышающего величину сезонного сдвига')
    
    df = pd.DataFrame(Data['observations'])
    Freq = Data['pandas_frequency']
    df.obs = df.obs.astype(float)   
    
    if Window_in_years == None or Window_in_years == 0:                                                           
        RWDS_forecast_list = [round(df.iloc[- Seasonality - 1 + i, 1]
                                    + df.obs.diff(Seasonality).mean(), 2) for i in range(Forecast_horizon)]
        # Создание временной даты соответсвующей прогнозным значениям
        df["date"] = pd.to_datetime(df["date"], dayfirst=True, format="%d.%m.%Y")
        Date = pd.date_range(df.iloc[-1,0] + (df.iloc[-1,0] - df.iloc[-2,0]),
                             periods = Forecast_horizon,
                             freq = Freq
                             )
        
        results = pd.Series(data = RWDS_forecast_list,
                            index = Date,
                            name = 'predicted')
        
        return(results)
    # скользящее окно       
    else:                                                                                                          
        if Window_in_years < 2:
            raise Real_Forecast_Error('Окно слишеом мало, следует ставить более 2 лет')
        else:
            # !! добавить параметры для других частотностей
            window = 12 * Window_in_years 
            RWDS_forecast_list = [round(df.iloc[- Seasonality - 1 + i, 1]
                                    + df.obs[-window:].diff(Seasonality).mean(), 2) for i in range(Forecast_horizon)]       
    
         # Создание временной даты соответсвующей прогнозным значениям
        df["date"] = pd.to_datetime(df["date"], dayfirst=True, format="%d.%m.%Y")
        Date = pd.date_range(df.iloc[-1,0] + (df.iloc[-1,0] - df.iloc[-2,0]),
                             periods = Forecast_horizon,
                             freq = Freq
                             )
        
        results = pd.Series(data = RWDS_forecast_list,
                            index = Date,
                            name = 'predicted')
        
        return(results)
def TS_real(Data: dict,            
            Forecast_horizon: int,
            Window_in_years: int = None):
    
    '''
    Функция строит реальный прогноз используя модель линейного тренда TS.
    
    Возвращает:
    pandas.DatsFrame — прогноз длиной Forecast_horizon с датами в индексе.
    '''
    
    df = pd.DataFrame(Data['observations'])
    Freq = Data['pandas_frequency']
    df['T'] = [i + 1 for i in range(len(df['obs']))]
    df.obs = df.obs.astype(float)
                                                                                 
    if Window_in_years == None or Window_in_years == 0:
        alpha = smf.ols('obs ~ T', data=df.iloc[:,[1, 2]]).fit().params['Intercept']    # рассчитываем константу 1
        betta = smf.ols('obs ~ T', data=df.iloc[:,[1, 2]]).fit().params['T']     # рассчитываем константу 2
        TS_forecast_list = [round(alpha + (df[-1:].index.to_list()[0] + i) * betta, 2) for i in range(Forecast_horizon)]
                # Создание временной даты соответсвующей прогнозным значениям
        df["date"] = pd.to_datetime(df["date"], dayfirst=True, format="%d.%m.%Y")
        Date = pd.date_range(df.iloc[-1,0] + (df.iloc[-1,0] - df.iloc[-2,0]),
                             periods = Forecast_horizon,
                             freq = Freq
                             )
        
        results = pd.Series(data = TS_forecast_list,
                            index = Date,
                            name = 'predicted')
        
        return(results)

    else:
        # добавить для других частотностей
        if Window_in_years < 2:
            raise Real_Forecast_Error('Окно слишком мало')
        window = 12 * Window_in_years                                                                                    
        alpha = smf.ols('obs ~ T', data=df.iloc[ - window:,[1, 2]]).fit().params['Intercept']    # рассчитываем константу 1
        betta = smf.ols('obs ~ T', data=df.iloc[ - window:,[1, 2]]).fit().params['T']     # рассчитываем константу 2
        TS_forecast_list = [round(alpha + (df[-1:].index.to_list()[0] + i) * betta, 2) for i in range(Forecast_horizon)]
        # Создание временной даты соответсвующей прогнозным значениям
        df["date"] = pd.to_datetime(df["date"], dayfirst=True, format="%d.%m.%Y")
        Date = pd.date_range(df.iloc[-1,0] + (df.iloc[-1,0] - df.iloc[-2,0]),
                             periods = Forecast_horizon,
                             freq = Freq
                             )
        
        results = pd.Series(data = TS_forecast_list,
                            index = Date,
                            name = 'predicted')
        
        return(results)
def ARIMA_real(Data: dict,
               Forecast_horizon : int, 
               Window_in_years: int = None):
    
        
    '''
    Функция строит реальный прогноз используя модель custom_ARIMA.
    Подбор параметров осуществляется с помощью функции pmd.auto_arima(),
    на основание информационного критерия Шварца. 
    
    Возвращает:
    pandas.Series — прогноз длиной Forecast_horizon с датами в индексе.
    '''
    
    df = pd.DataFrame(Data['observations'])
    Freq = Data['pandas_frequency']
    df.obs = df.obs.astype(float) # значения переводятся в формат float
    df["date"] = pd.to_datetime(df["date"], dayfirst=True, format="%d.%m.%Y")
    df.index = df['date'] # Индекс дата
    df = df.drop('date', axis = 1)
    df = df.asfreq(Freq) # Установка частотности  
    
    if Window_in_years == None or Window_in_years == 0: 
        ARIMA_forecast = pmd.auto_arima(df.obs, information_criterion = 'bic', d = 1, stepwise=True).predict(Forecast_horizon)
        
        return(ARIMA_forecast)
    else:
        # добавить для других частотностей
        if Window_in_years < 2:
            raise Real_Forecast_Error('Окно слишком мало')
        window = 12 * Window_in_years 
        ARIMA_forecast = pmd.auto_arima(df.obs[-window:], information_criterion = 'bic', d = 1, stepwise=True).predict(Forecast_horizon)
    
    return ARIMA_forecast
def custom_ARIMA_real(Data: dict,
                      Forecast_horizon : int):
    
    '''
    Функция строит реальный прогноз используя модель custom_ARIMA.
    Подбор параметров осуществляется экспертным автоматизированным методом. 
    
    Возвращает:
    pandas.Series — прогноз длиной Forecast_horizon с датами в индексе.
    '''
    
    ARIMA_forecast = GetARIMA(Data).fit(criteria="aic")[1].fit().get_forecast(steps=Forecast_horizon).predicted_mean    
    
    return ARIMA_forecast