import pandas as pd
import numpy as np
import time
import warnings
from glob import glob
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta, TH
from pandas.core.common import SettingWithCopyWarning
from math import sqrt, exp, log, pi
from scipy.stats import norm
import streamlit as st
import matplotlib.pyplot as plt
from io import BytesIO
import base64

warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)


def user_inputs():
    def time(time_str):
        if time_str.find(':00') == 5:
            m = str(int(time_str.split(':')[1]) - 1)
            h = time_str.split(':')[0]
            s = '59'
            time_str = h + ':' + m + ':' + s
            return time_str
        elif time_str.find('00:00') == 3:
            m = '59'
            h = str(int(time_str.split(':')[0]) - 1)
            s = '59'
            time_str = h + ':' + m + ':' + s
            return time_str
        else:
            return time_str

    # Title #
    # st.sidebar.title("User Inputs")
    # st.title("Welcome to Streamlit!")
    st.title('Option Chain Trading')
    st.header('ATM Strategy')
    # Trade #
    st.subheader('Trade')
    ###########
    sl, sp, en, ex, hp = st.columns([1, 1, 2, 2, 1])
    ###### stop loss
    with sl:
        st_loss = st.text_input('Stop Loss (%)', '25%')
    ##### stop profit
    with sp:
        st_profit = st.text_input('Stop Profit (%)', '30%')
    # Entry Time #
    with en:
        entry_time_str = time(st.text_input('Entry time (hh:mm:ss):', '09:20:00'))
    # Exit time #
    with ex:
        exit_time_str = time(st.text_input('Exit time (hh:mm:ss):', '15:20:00'))
    with hp:
        hedging_option = st.selectbox('Delta Hedging?', ('No', 'Yes'))

    st_loss_pct = int(str(st_loss).split('%')[0]) / 100

    ######### Trailing Profit position
    # new_st_loss = st.sidebar.text_input('Enter stop loss which is near to above stop loss', '0%')
    # new_st_loss_pct = int(str(new_st_loss).split('%')[0])/100
    # new_st_loss_lot = st.sidebar.selectbox('Select loss position', ('Half', 'Double'))

    st_profit_pct = int(str(st_profit).split('%')[0]) / 100

    ######### Trailing Profit position
    # new_st_profit = st.sidebar.text_input('Enter stop profit which is near to above stop profit', '0%')
    # new_st_profit_pct = int(str(new_st_profit).split('%')[0])/100
    # new_st_profit_lot = st.sidebar.selectbox('Select profit position', ('Half', 'Double'))

    # if (new_st_profit_pct < (st_profit_pct-1)) & (new_st_loss_pct < (st_loss_pct-1)):
    #     st.success('#Trailing\n\nTrailing SL: `%s`\n\nTrailing SL position:  `%s`\n\nTrailing SP:  `%s`\n\nTrailing SP position:  `%s`' % (new_st_loss,new_st_profit,new_st_loss_lot,new_st_profit_lot))
    # else:
    #     st.error('Enter the trailing SL % and trailing PB % less than SL% and PB% respectively')

    # # Entry Time #
    # if (int(entry_time_str.split(':')[0]) > 8) | (int(entry_time_str.split(':')[0]) < 16):
    #     entry_time = datetime.strptime(entry_time_str, "%H:%M:%S").time()
    # else:
    #     st.error("Enter the time in given format and in 12 hours format and between 08:00:00 and 16:00:00")

    # # Exit Time #
    # if ((int(exit_time_str.split(':')[0]) > 8) | (int(exit_time_str.split(':')[0]) < 16)):
    #     exit_time = datetime.strptime(time(exit_time_str), "%H:%M:%S").time()
    # else:
    #     st.error("Enter the time in in given format and in 12 hours format and between 09:00:00 and 16:00:00")

    entry_time = datetime.strptime(entry_time_str, "%H:%M:%S").time()
    exit_time = datetime.strptime(time(exit_time_str), "%H:%M:%S").time()
    start_time = datetime.strptime('09:15:59', "%H:%M:%S").time()
    end_time = datetime.strptime('15:30:59', "%H:%M:%S").time()

    if (start_time <= entry_time < exit_time <= end_time) == False:
        st.error('Error: Time should be between 9:16:00 and 15:30:00')

        # st.success('Stop Loss: `%s` \n Stop Profit: `%s` \n Start date: `%s` \n End date: `%s`' % (st_loss,st_profit,entry_time, exit_time))
    # else:

    ##### Hedging Options
    col1, col2 = st.columns(2)

    # % for square off all positions and delta hedging
    if hedging_option == 'Yes':
        with col1:
            sqr_off_position = st.text_input('What % change in Nifty, square off all positions', '0%')
        sqr_off_position_pct = int(str(sqr_off_position).split('%')[0]) / 100
        with col2:
            delta_position = st.text_input('What % change in Nifty, look at the delta', '0%')
        delta_pct = int(str(delta_position).split('%')[0]) / 100
        st.text('If Delta (D) > 0 , "Sell Call"\n\nIf Delta (D) < 0 , "Sell Put"')

        # store user input  in dictionary #
        trade_keys = ['st_loss', 'st_profit', 'entry_time_str', 'entry_time', 'exit_time_str', 'exit_time', 'end_time',
                      'hedging_option', 'sqr_off_position_pct', 'delta_pct']
        trade_values = [st_loss_pct, st_profit_pct, entry_time_str, entry_time, exit_time_str, exit_time, end_time,
                        hedging_option, sqr_off_position_pct, delta_pct]
        trade_n_dict = dict(zip(trade_keys, trade_values))
        # print(trade_n_dict)
    else:
        trade_keys = ['st_loss', 'st_profit', 'entry_time_str', 'entry_time', 'exit_time_str', 'exit_time', 'end_time',
                      'hedging_option']
        trade_values = [st_loss_pct, st_profit_pct, entry_time_str, entry_time, exit_time_str, exit_time, end_time,
                        hedging_option]
        trade_n_dict = dict(zip(trade_keys, trade_values))

    # Legs #
    st.subheader('Legs')
    n_legs = int(st.selectbox('How many legs you want to execute?', (2, 1)))
    legs = {}
    ### add key for dublicate inputs
    for i in range(n_legs):
        name_leg_key = 'Leg' + str(i + 1)
        lg, inst, act, opt, strk, exp, lt = st.columns([2, 4, 3, 3, 3, 4, 3])
        with lg:
            st.markdown(name_leg_key)
        with inst:
            instrument = st.selectbox('Symbol', ('NIFTY', 'BANK NIFTY'), key='a1' + str(i + 1))
        with act:
            action = st.selectbox('Action', ('BUY', 'SELL'), key='a' + str(i + 1))
        if name_leg_key == 'Leg1':
            with opt:
                option_type = st.selectbox('Option Type', ('CE', 'PE'), key='b' + str(i + 1))
        elif name_leg_key == 'Leg2':
            with opt:
                option_type = st.selectbox('Option Type', ('PE', 'CE'), key='b' + str(i + 1))
        with strk:
            strike_given = st.text_input('Strike', 'ATM', key='c' + str(i + 1))  # user I/O as well as ATM
        # expiry on strike is condition based so
        if strike_given.upper() == 'ATM':
            strike_given = 'ATM'
            expiry = 'Nearest Weekly'
            with exp:
                lot_size = st.text_input('Lot', "", key='f' + str(i + 1))
        else:
            strike_given = int(strike_given)
            with exp:
                expiry_str = str(st.date_input('Expiry Date', key='e' + str(i + 1)))  # user I/O
            expiry = datetime.strptime(expiry_str, "%Y-%m-%d").date()
            with lt:
                lot_size = st.text_input('Lot', "", key='f' + str(i + 1))

        try:
            No_of_lots = int(lot_size)
            directory = os.getcwd()
#             print(directory)
            path = directory+'\\Test_code\\'
            leg_keys = ['instrument', 'action', 'option_type', 'strike_given', 'expiry', 'No_of_lots', 'path']
            leg_values = [instrument, action, option_type, strike_given, expiry, No_of_lots, path]
            leg_n_dict = dict(zip(leg_keys, leg_values))
            legs[name_leg_key] = leg_n_dict
        except:
            st.error("Please make sure that you have entered all inputs")
            st.stop()
        # store in Dictionary

        # st.success('#Legs\n\nInstrument: `%s`\n Action:  `%s`\n Option Type:  `%s`\n Strike:  `%s`\n Expiry: `%s`\n No. of Lots: `%s`' % (instrument,action,option_type,strike_given,expiry,No_of_lots))
    st.success("If you have chosen 'ATM' strike then expiry will be 'Nearest Weekly'")
    # print(legs)
    return legs, trade_n_dict


def data_file_df_creation(legs):
    for i in legs.items():
        # print(i)
        way = pd.DataFrame(glob(i[1]['path'] + '*'), columns=['link'])
        way['data_date'] = way['link'].apply(lambda x: x.split('_')[-1].split('.')[0])
        way['data_date'] = way['data_date'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))
        way.sort_values(['data_date'], inplace=True)
        way.reset_index(drop=True, inplace=True)
        return way


def data_format_change(df):
    df['Expiry'] = df['Expiry'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))
    df['Date'] = df['Date'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d').date())
    df['Time'] = df['Time'].apply(lambda x: datetime.strptime(x, '%H:%M:%S').time())
    return df


def identify_atm_nwe(df, current_file_date):
    expiries = list(df['Expiry'].value_counts().index)
    expiries_type = [i for i in range(len(expiries))]
    expiries_dict = dict(zip(expiries, expiries_type))

    def set_values(line, value):
        return value[line]

    df['Expiry_Type'] = np.where((df['Type'] == 'FUT'), df['Expiry'].apply(set_values, args=(expiries_dict,)), "")

    ft_exp = list(df[df['Type'] == 'FUT']['Expiry_Type'].value_counts().index)
    future_expiries_type = []
    initial = 'I'
    for type_exp in range(len(ft_exp)):
        type_exp += 1
        type_exp_modified = initial * type_exp
        future_expiries_type.append(type_exp_modified)

    future_expiries_dict = dict(zip(ft_exp, future_expiries_type))
    df['Expiry_Type'].replace(future_expiries_dict, inplace=True)

    future_expiry_category = 'I'

    futures_data = df[(df['Type'] == 'FUT') & (df['Expiry_Type'] == future_expiry_category)]
    futures_data.reset_index(drop=True, inplace=True)
    # print('Futures_Shape: ', futures_data.shape)
    # print('DF Shape: ', df.shape)

    # futures_data['Date'] = pd.to_datetime(futures_data['Date'], format='%Y:%m:%d').dt.date
    atm_future = futures_data[(futures_data['Date'] == current_file_date) & (futures_data['Time'] == entry_time)][
        'Close'].iloc[0]
    # print('Nearest ATM Future: ', atm_future)

    # row['data_date'].date() for 'DATE' and str(entry_time) for entry time
    # from dateutil.relativedelta import relativedelta, TH
    nearest_expiry = current_file_date + relativedelta(weekday=TH(+1))
    # print('Nearest Expiry: ', nearest_expiry)
    df['Expiry'] = pd.to_datetime(df['Expiry'], format='%Y:%m:%d').dt.date
    return atm_future, nearest_expiry, futures_data


def day_output_file_name_creation(legs_items, file_name_end):
    output_file_name = ''
    leg_file_name = str(legs_items[0]) + "_" + legs_items[1]['option_type'] + "_" + legs_items[1]['action'] + "_" + \
                    legs_items[1]['strike_given'] + "_" + file_name_end
    # print(leg_file_name)
    output_file_name += leg_file_name[:-4]
    # print(output_file_name)
    return leg_file_name, output_file_name


def leg_df_creation(df, atm_future_price, legs_items, current_nearest_expiry):
    df['Abs_Diff'] = abs(df['Strike'] - atm_future_price)
    #         print(df.head(5))

    min_diff = df['Abs_Diff'].min()
    df_leg = df[(df['Abs_Diff'] == min_diff) & (df['Type'] == legs_items[1]['option_type']) & (
            (df['Expiry'] == current_nearest_expiry) | (df['Expiry'] == current_nearest_expiry - timedelta(days=1))
            | (df['Expiry'] == current_nearest_expiry - timedelta(days=2)))]
    # print('Intial leg shape: ', df_leg.shape)
    # print(df_leg.head(2))
    # print(df_leg.tail(2))
    return df_leg


def d(sigma, S, K, r, q, t):
    d1 = 1 / (sigma * sqrt(t)) * (log(S / K) + (r - q + sigma ** 2 / 2) * t)
    d2 = d1 - sigma * sqrt(t)
    return d1, d2


def call_price(sigma, S, K, r, q, t, d1, d2):
    C = norm.cdf(d1) * S * exp(-q * t) - norm.cdf(d2) * K * exp(-r * t)
    return C


def put_price(sigma, S, K, r, q, t, d1, d2):
    P = - S * exp(-q * t) * norm.cdf(-d1) + K * exp(-r * t) * norm.cdf(-d2)
    return P


def call_IV(S, K, r, q, t, C0):
    #  Tolerances
    if t != 0:
        tol = 1e-3
        epsilon = 1

        #  Variables to log and manage number of iterations
        count = 0
        max_iter = 1000

        #  We need to provide an initial guess for the root of our function
        vol = 0.50

        while epsilon > tol:
            #  Count how many iterations and make sure while loop doesn't run away
            count += 1
            if count >= max_iter:
                print('Breaking on count')
                break

            #  Log the value previously calculated to computer percent change between iterations
            orig_vol = vol

            #  Calculate the vale of the call price
            d1, d2 = d(vol, S, K, r, q, t)
            function_value = call_price(vol, S, K, r, q, t, d1, d2) - C0

            #  Calculate vega, the derivative of the price with respect to volatility
            vega = S * norm.pdf(d1) * sqrt(t) * exp(-q * t)

            #  Update for value of the volatility
            vol = -function_value / vega + vol

            #  Check the percent change between current and last iteration
            epsilon = abs((vol - orig_vol) / orig_vol)
    else:
        vol = 1

    return vol


def DeltaC(S, K, r, q, t, call_vol):
    if t != 0:
        d1, d2 = d(call_vol, S, K, r, q, t)
        DC = exp(-q * t) * norm.cdf(d1)
    else:
        d1, d2 = d(call_vol, S, K, r, q, 0.00001)
        DC = exp(-q * 0.00001) * norm.cdf(d1)
    return DC


# This Python code can be used to calculate the value for Implied Volatility for a European put


def put_IV(S, K, r, q, t, P0):
    if t != 0:
        #  Tolerances
        tol = 1e-3
        epsilon = 1

        #  Variables to log and manage number of iterations
        count = 0
        max_iter = 1000

        #  We need to provide an initial guess for the root of our function

        vol = 0.50
        while epsilon > tol:
            #  Count how many iterations and make sure while loop doesn't run away
            count += 1
            if count >= max_iter:
                print('Breaking on count')
                break

            #  Log the value previously calculated to computer percent change
            #  between iterations
            orig_vol = vol

            #  Calculate the vale of the call price
            d1, d2 = d(vol, S, K, r, q, t)
            function_value = put_price(vol, S, K, r, q, t, d1, d2) - P0

            #  Calculate vega, the derivative of the price with respect to
            #  volatility
            vega = S * norm.pdf(d1) * sqrt(t) * exp(-q * t)

            #  Update for value of the volatility
            vol = -function_value / vega + vol

            #  Check the percent change between current and last iteration
            epsilon = abs((vol - orig_vol) / orig_vol)
    else:
        vol = 1

    return vol


def DeltaP(S, K, r, q, t, put_vol):
    if t != 0:
        d1, d2 = d(put_vol, S, K, r, q, t)
        DP = -exp(-q * t) * norm.cdf(-d1)
    else:
        d1, d2 = d(put_vol, S, K, r, q, 0.00001)
        DP = -exp(-q * 0.00001) * norm.cdf(-d1)
    return DP


def legwise_output_file_generation(df_leg, entry_time, atm_future_price, legs_items, futures_data):
    df_leg['ATM_Entry'] = atm_future_price
    entry_price = df_leg[(df_leg['Time'] == entry_time) | (df_leg['Time'] == str(entry_time))]
    # print(entry_price.shape)
    if (entry_price.shape[0] > 1) & (legs_items[1]['option_type'] == 'CE'):
        entry_price = entry_price[(entry_price['Type'] == 'CE') & (entry_price['Strike'] >=
                                                                   atm_future_price)]['Close'].iloc[0]
    # print(entry_price)
    elif (entry_price.shape[0] > 1) & (legs_items[1]['option_type'] == 'PE'):
        entry_price = entry_price[(entry_price['Type'] == 'PE') & (entry_price['Strike'] <=
                                                                   atm_future_price)]['Close'].iloc[0]
    # print(entry_price)
    else:
        entry_price = df_leg[(df_leg['Time'] == entry_time) | (df_leg['Time'] == str(entry_time))]['Close'].iloc[0]
    # print(entry_price)
    # print('Entry Price: ', entry_price)

    df_leg = df_leg[(df_leg['Time'] >= df_leg[df_leg['Time'] == entry_time]['Time'].iloc[0]) &
                    (df_leg['Time'] != ignore_time)]
    df_leg['Entry_Price'] = entry_price
    df_leg['Action'] = legs_items[1]['action']
    df_leg['PnL'] = np.where(df_leg['Action'] == 'BUY', df_leg['Close'] - df_leg['Entry_Price'],
                             -(df_leg['Close'] - df_leg['Entry_Price']))
    # print(df_leg[['Action','PnL']])
    df_leg['PnL*Lots'] = df_leg['PnL'] * legs_items[1]['No_of_lots']
    df_leg['Days_Left_to_Expiry'] = ((df_leg['Expiry'].iloc[0]) - (df_leg['Date'].iloc[0])).days / 365
    for index, row in df_leg.iterrows():
        current_time = row['Time']
        current_asset_price = futures_data[futures_data['Time'] == current_time]['Close'].iloc[0]
        df_leg.loc[index, 'Current_Asset_Futures_Price'] = current_asset_price
        if row['Type'] == 'CE':
            call_vol = call_IV(atm_future_price, row['Strike'], r, q, row['Days_Left_to_Expiry'], row['Close'])
            call_delta = DeltaC(atm_future_price, row['Strike'], r, q, row['Days_Left_to_Expiry'], call_vol)
            df_leg.loc[index, 'delta_call'] = call_delta
            if row['Action'] == 'SELL':
                delta_lots = (-call_delta * legs_items[1]['No_of_lots'])
                df_leg.loc[index, 'delta_call*lots'] = delta_lots
            else:
                delta_lots = (call_delta * legs_items[1]['No_of_lots'])
                df_leg.loc[index, 'delta_call*lots'] = delta_lots
        elif row['Type'] == 'PE':
            put_vol = put_IV(atm_future_price, row['Strike'], r, q, row['Days_Left_to_Expiry'], row['Close'])
            put_delta = DeltaP(atm_future_price, row['Strike'], r, q, row['Days_Left_to_Expiry'], put_vol)
            df_leg.loc[index, 'delta_put'] = put_delta
            if row['Action'] == 'SELL':
                delta_lots = (-put_delta * legs_items[1]['No_of_lots'])
                df_leg.loc[index, 'delta_put*lots'] = delta_lots
            else:
                delta_lots = (put_delta * legs_items[1]['No_of_lots'])
                df_leg.loc[index, 'delta_put*lots'] = delta_lots
        else:
            pass

    df_leg['Change_in_asset'] = abs(df_leg['Current_Asset_Futures_Price'] - df_leg['ATM_Entry'])
    df_leg['Change_in_asset(%)'] = round((df_leg['Change_in_asset'] / atm_future_price) * 100, 2)
    df_leg.reset_index(drop=True, inplace=True)
    # print(df_leg.head(7))
    print(df_leg.shape)
    # output_dfs.append(df_leg)
    return df_leg


def n_leg_output_generation(output_dfs, atm_future_price, st_loss, st_profit, legs_items, leg_file_name,
                            output_file_name):
    output_df = pd.concat(output_dfs, axis=1)
    output_dfs.clear()
    print(output_df['Entry_Price'])
    output_df['Sum_Entry_Price_Legs'] = output_df['Entry_Price'].sum(axis=1)
    output_df['Total PnL'] = output_df['PnL*Lots'].sum(axis=1, numeric_only=True)
    output_df['Total Delta'] = output_df['delta_call*lots'] + output_df['delta_put*lots']
    output_df['st_loss_point'] = round(
        (st_loss * (-(output_df['Sum_Entry_Price_Legs'])) * (legs_items[1]['No_of_lots'])), 2)
    output_df['book_profit_point'] = round(
        (st_profit * (+(output_df['Sum_Entry_Price_Legs'])) * (legs_items[1]['No_of_lots'])), 2)
    return output_df


def stop_conditions_applied(n_leg_df, current_file_name, exit_time):
    exit_cond = []
    # error first iteration : treated Total PnL and st_loss_point on different sides of comparison operator
    n_leg_df_new = n_leg_df
    for a, b in n_leg_df_new.iterrows():
        if b['Total PnL'] <= b['st_loss_point']:
            exit_cond.append('Stop Loss Hit')
            #         print("Stop Loss Hit: ", b['Time'])
            b['Exit Condition'] = 'Stop Loss Hit'
        elif b['Total PnL'] >= b['book_profit_point']:
            exit_cond.append('Profit Booked')
            #         print("Stop Profit Hit: ", b['Time'])
            b['Exit Condition'] = 'Stop Profit Hit'
        elif b['Time'][0] == exit_time:
            exit_cond.append('Exit Time Hit')
        else:
            exit_cond.append('')

    #         print('Exit Time Hit')
    # print(len(exit_cond))
    # print(exit_time)
    n_leg_df_new['Exit Cond'] = exit_cond
    # print(n_leg_df_new[n_leg_df_new['Exit Cond'] != ''])
    # n_leg_df_new.to_csv(temp_file_name,index=False)
    return n_leg_df_new


def end_results_normalization(n_leg_df, atm_future_price, futures_data):
    trade_exit_time = n_leg_df[n_leg_df['Exit Cond'] != ''].iloc[0]['Time'].iloc[0]
    ATM_Exit = futures_data[futures_data['Time'] == trade_exit_time]['Close'].iloc[0]
    print('ATM Exit: ', ATM_Exit)
    n_leg_df['ATM Entry'] = atm_future_price
    n_leg_df['ATM_Exit'] = ATM_Exit
    print('Exit Condition: ', n_leg_df[n_leg_df['Exit Cond'] != ''].iloc[0]['Exit Cond'])
    final_exit_row = n_leg_df.loc[n_leg_df['Exit Cond'] != '']
    final_exit_row_columns = list(final_exit_row.columns)
    final_exit_row_columns = [x for x in final_exit_row_columns if not x.startswith('Unnamed:')]
    final_exit_row = n_leg_df[n_leg_df['Exit Cond'] != ''].iloc[0]
    final_exit_row = final_exit_row.dropna().tolist()
    # print(final_exit_row)
    return n_leg_df, final_exit_row, final_exit_row_columns


def execution_of_legs_atm_nw(legs):
    data_file_df = data_file_df_creation(legs)
    output_file_name_final = ''
    final_exit_row_columns=[]
    trade_output_dfs = []
    for index, row in data_file_df.iterrows():
        df = pd.read_csv(row['link'])
        current_file_date = row['data_date'].date()
        print('File Date: ', current_file_date)
        df = data_format_change(df)  # df for all use
        atm_future_price, current_nearest_expiry, futures_data = identify_atm_nwe(df, current_file_date)
        output_dfs = []
        for legs_items in legs.items():
            file_name_end = row['link'][-20:]
            leg_file_name, output_file_name = day_output_file_name_creation(legs_items, file_name_end)
            output_file_name_final += output_file_name
            df_leg = leg_df_creation(df, atm_future_price, legs_items, current_nearest_expiry)
            df_leg_new = legwise_output_file_generation(df_leg, entry_time, atm_future_price, legs_items, futures_data)
            output_dfs.append(df_leg_new)
            if len(output_dfs) == len(legs):
                output_file_name_final = output_file_name_final + '.csv'
                n_leg_df = n_leg_output_generation(output_dfs, atm_future_price, st_loss, st_profit, legs_items,
                                                   leg_file_name, output_file_name)
                n_leg_df = stop_conditions_applied(n_leg_df, current_file_date, exit_time)
                print(output_file_name_final)
                n_leg_df.to_csv(output_file_name_final, index=False)
                output_file_name_final = ''
                n_leg_df, final_exit_row, final_exit_row_columns = end_results_normalization(n_leg_df, atm_future_price,
                                                                                             futures_data)
                trade_output_dfs.append(final_exit_row)

    print(final_exit_row_columns)
    trade_output_df = pd.DataFrame(trade_output_dfs, columns=final_exit_row_columns)
    # trade_output_df.to_csv('Normalized_Code_Output_for_all_days_v2.csv')
    return trade_output_df
    # trade_output_dfs.append(final_exit_row)


def df_column_uniquify(df):
    df_columns = df.columns
    new_columns = []
    for item in df_columns:
        counter = 0
        newitem = item
        #         print(newitem,item)
        while newitem in new_columns:
            counter += 1
            # print(counter)
            newitem = "{}_{}".format(item, counter)
        new_columns.append(newitem)
    df.columns = new_columns
    return df


if __name__ == '__main__':
    # legs = {'Leg 1': {'instrument': 'NIFTY', 'action': 'SELL', 'option_type': 'CE', 'strike_given': 'ATM',
    #                   'expiry': 'Nearest Weekly', 'No_of_lots': 1,
    #                   'path': 'D:/Chistats Equity Analytics Local Files/Normalized_2021/NIFTY/'},
    #         'Leg 2': {'instrument': 'NIFTY', 'action': 'SELL', 'option_type': 'PE', 'strike_given': 'ATM',
    #                   'expiry': 'Nearest Weekly', 'No_of_lots': 1,
    #                   'path': 'D:/Chistats Equity Analytics Local Files/Normalized_2021/NIFTY/'}}

    # st.subheader('My sub')
    leg, trade = user_inputs()
    print(leg, trade)
    legs = leg
    entry_time = trade['entry_time']
    exit_time = trade['exit_time']
    ignore_time = trade['end_time']
    st_loss = trade['st_loss']
    st_profit = trade['st_profit']
    tol = 1e-3
    epsilon = 1
    r = 0.05
    q = 0.05
    N = 10000000
    start = time.perf_counter()
    if st.button('Add Position'):
        with st.spinner("Backtesting data..."):
            output = execution_of_legs_atm_nw(legs)
        df = df_column_uniquify(output)
        print(df)
        st.markdown("Resulted Output:")
        st.dataframe(df)
        st.write('Total Profit and Loss')
        df_PnL = df[['Date','Total PnL']]
        # Convert datetime from datetime64[ns] to string type
        df_PnL['Date'] = df_PnL['Date'].astype(str)
        df_PnL = df_PnL.set_index('Date')
        # st.line_chart(df1)
        st.line_chart(df_PnL['Total PnL'].cumsum(skipna = False))
        st.success('Done')

    # # Data of recent days
    # st.write('Recent data ')
    # st.dataframe(output.tail(10))
    # # execution_of_legs_atm_nw(legs)
    end = time.perf_counter()
    print(end - start)
