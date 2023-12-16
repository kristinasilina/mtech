import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import io
import chardet

st.set_page_config(layout="wide") # широкий формат страницы
st.title('Исследование зависимости частоты взятия больничных работниками от пола и возраста.')

# Виджет для загрузки CSV файла
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

# Если файл был загружен, читаем данные из него
if uploaded_file is not None:
    # Определение кодировки файла - при тестах у меня показал кодировку файла windows-1251, поэтому дефолтная utf_8 не срабатывала (в любом случае так код универсальнее)
    raw_content = uploaded_file.getvalue()
    detector = chardet.universaldetector.UniversalDetector()
    detector.feed(raw_content)
    detector.close()
    file_encoding = detector.result['encoding']
    # чтение данных с определенной кодировкой
    content = raw_content.decode(file_encoding)
    data = pd.read_csv(io.StringIO(content), engine='python')

    # приведение датафрейма к рабочему виду, распределение по колонкам
    data = data['Количество больничных дней,"Возраст","Пол"'].str.split(',', expand=True)
    data.columns = ['Количество больничных дней', 'Возраст', 'Пол']
    data['Пол'] = data['Пол'].str.replace('"', '')
    data['Количество больничных дней'] = data['Количество больничных дней'].astype('int')
    data['Возраст'] = data['Возраст'].astype('int')

    st.success("File successfully uploaded and data loaded.")

    # Отображаем виджеты только после загрузки файла
    # Отображение сырых данных
    if st.checkbox('Show raw data'):
        st.subheader('Raw data')
        st.write(data)

    # диаграммы распределений по первому пункту заданий - гендерные различия
    st.subheader('Гистограмма распределения частоты и продолжительности больничных по гендеру:')
    col1, col2 = st.columns(2)

    # Фильтрация данных
    work_days = st.slider('Select work_days', 0, 7, 2) # тут можно задать количество рабочих дней черех слайдер
    filtered_data_men = data[(data['Пол'] == 'М') & (data['Количество больничных дней'] > work_days)]
    hist_values_men, bin_edges_men = np.histogram(filtered_data_men['Количество больничных дней'], bins=filtered_data_men['Количество больничных дней'].nunique())

    col1.subheader('Мужчины') #строим гистограмму распределения
    fig, ax = plt.subplots()
    ax.hist(filtered_data_men['Количество больничных дней'], bins=bin_edges_men, color="#00bfff", edgecolor='black')

    # Добавляем подписи к осям
    ax.set_xlabel('Number of days')
    ax.set_ylabel('Frequency')
    ax.set_ylim(0, max(hist_values_men) + 5) # масштаб оси y специально один у двух графиков - для наглядности
    ax.set_xticks(range(work_days, work_days + len(hist_values_men)))

    # Отображаем график
    col1.pyplot(fig)

    # То же самое для женщин
    filtered_data_women = data[(data['Пол'] == 'Ж') & (data['Количество больничных дней'] > work_days)]
    hist_values_women, bin_edges_women = np.histogram(filtered_data_women['Количество больничных дней'], bins=filtered_data_women['Количество больничных дней'].nunique())

    col2.subheader('Женщины')
    fig, ax = plt.subplots()
    ax.hist(filtered_data_women['Количество больничных дней'], bins=bin_edges_women, color="#a9ea44", edgecolor='black')

    # Добавляем подписи к осям
    ax.set_xlabel('Number of days')
    ax.set_ylabel('Frequency')
    ax.set_ylim(0, max(hist_values_men) + 5)
    ax.set_xticks(range(work_days, work_days + len(hist_values_women)))

    # Отображаем график второй
    col2.pyplot(fig)

    # Для сравнения частоты я не догадалась как использовать статтесты, но можем вывести обычное процентное соотношение, тоже интерактивное со слайдером
    st.subheader('Процентное соотношение частоты взятия больничных:')
    st.text(f'Мужчины берут больничный более {work_days} work-days на {(filtered_data_men.shape[0]-filtered_data_women.shape[0])*100/filtered_data_women.shape[0] :.0f}% чаще, чем женщины.')

    # Статтест для проверки гипотез, и вывод его параметров и результатов
    st.subheader('T-test (двухвыборочный) для проверки гипотезы о гендерных различиях в статистике:')
    h0_gender = f'Среднее количество пропущенных рабочих дней по болезни (более {work_days} work-days) у мужчин и женщин одинаково.'
    h1_gender = f'Среднее количество пропущенных рабочих дней по болезни (более {work_days} work-days) у мужчин и женщин различается значимо.'
    st.text(f'HO: {h0_gender}\nH1: {h1_gender}')
    t_stat_gender, p_value_gender = stats.ttest_ind(filtered_data_women['Количество больничных дней'],
                                      filtered_data_men['Количество больничных дней'])
    if p_value_gender < 0.05:
        hyp_gender = h1_gender
    else:
        hyp_gender = h0_gender
    st.text(f'Men vs Women:\n'
                  f't-статистика: {t_stat_gender:.2f}\n'
                  f'p-значение: {p_value_gender:.4f}')
    st.text(f'Принятый уровень значимости: 0.05\n'
            f'По итогу теста верна гипотеза: {hyp_gender}\n')

    # ВТОРАЯ ЧАСТЬ задания - исследования возрастных различий
    # Гистограмма
    st.subheader('Гистограмма распределения частоты и продолжительности больничных по возрасту:')
    col3, col4 = st.columns(2)

    # Фильтрация данных - используем сразу два слайдера
    work_days_2 = st.slider('Select work_days', 0, 7, 2, key = 42)
    age = st.slider('Select age', data['Возраст'].min(), data['Возраст'].max(), 35)
    filtered_data_old = data[(data['Возраст'] > age ) & (data['Количество больничных дней'] > work_days_2)]
    hist_values_old, bin_edges_old = np.histogram(filtered_data_old['Количество больничных дней'], bins=filtered_data_old['Количество больничных дней'].nunique())

    col3.subheader(f'Работники старше, чем {age} лет')
    fig, ax = plt.subplots()
    ax.hist(filtered_data_old['Количество больничных дней'], bins=bin_edges_old, color="#00bf21", edgecolor='black')

    # Добавляем подписи к осям
    ax.set_xlabel('Number of days')
    ax.set_ylabel('Frequency')
    ax.set_ylim(0, max(hist_values_old) + 5) # масштаб опять одинаковый для обоих графиков
    ax.set_xticks(range(work_days_2, work_days_2 + len(hist_values_old)))

    # Отображаем график
    col3.pyplot(fig)

    # То же самое для молодых работников
    filtered_data_young = data[(data['Возраст'] <= age ) & (data['Количество больничных дней'] > work_days_2)]
    hist_values_young, bin_edges_young = np.histogram(filtered_data_young['Количество больничных дней'], bins=filtered_data_young['Количество больничных дней'].nunique())

    col4.subheader(f'Работники моложе {age} лет')
    fig, ax = plt.subplots()
    ax.hist(filtered_data_young['Количество больничных дней'], bins=bin_edges_young, color="#a9eaff", edgecolor='black')

    # Добавляем подписи к осям
    ax.set_xlabel('Number of days')
    ax.set_ylabel('Frequency')
    ax.set_ylim(0, max(hist_values_old) + 5)
    ax.set_xticks(range(work_days_2, work_days_2 + len(hist_values_young)))

    # Отображаем график
    col4.pyplot(fig)

    # Процентная интерактивная статистика соотношения
    st.subheader('Процентное соотношение частоты взятия больничных:')
    st.text(f'Работники старше {age} лет берут больничный более {work_days} work-days на {(filtered_data_old.shape[0] - filtered_data_young.shape[0]) * 100 / filtered_data_young.shape[0] :.0f}% чаще, чем работники моложе {age} лет.')

    #  Статтест для проверки гипотез, и вывод его параметров и результатов
    st.subheader('T-test (двухвыборочный) для проверки гипотезы о возрастных различиях в статистике:')
    h0_age = f'Среднее количество пропущенных рабочих дней по болезни (более {work_days_2} work-days) у работников старше {age} лет и моложе {age} лет одинаково.'
    h1_age = f'Среднее количество пропущенных рабочих дней по болезни (более {work_days_2} work-days) у работников старше {age} лет и моложе {age} лет различается значимо.'
    st.text(f'HO: {h0_age}\nH1: {h1_age}')
    t_stat_age, p_value_age = stats.ttest_ind(filtered_data_old['Количество больничных дней'],
                                      filtered_data_young['Количество больничных дней'])
    if p_value_age < 0.05:
        hyp_age = h1_age
    else:
        hyp_age = h0_age
    st.text(f'Old vs Young:\n'
                  f't-статистика: {t_stat_age:.2f}\n'
                  f'p-значение: {p_value_age:.4f}')
    st.text(f'Принятый уровень значимости: 0.05\n'
            f'По итогу теста верна гипотеза: {hyp_age}\n')