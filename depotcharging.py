'''
Title: Truck Power Draw Simulation
Author: Sameer Bajaj, Brennan Borlaug
Date: 2024-12-01
Description: Builds off of original script by @bborlaug (GitHub), adding user selection of bound (min, max, average);
plotting of all fleets as an aggregate plot and multi-line graph; scaling of data to simulate large systems, and
input of different fleet sizes for the three different fleets, allowing for fitting of data to real world proportions.
'''


import os
import math
import random
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
file_locations = []
directory = ''
def generate_load_profiles(fleet,
                           fleet_size,
                           charge_strat,
                           n_samples=30,
                           kwh_per_mile=1.8,
                           kw=100,
                           seed=0,
                           to_file=True,
                           to_plot = True,
                           ylim=None,
                           print_daily_energy_var=True,
                           agg_15min=True, data_type = 'all', directory = 'C:\\Users\\samee\\PycharmProjects\\EVtruck charging'):

    """
    Generates n_samples fleet load profiles for fleet ('fleet1-beverage-
    delivery', 'fleet2-warehouse-delivery', or 'fleet3-food-delivery')
    and fleet_size for a particular charge_strat ('immediate', 'delayed',
    'min_power') assuming kwh_per_mile average energy consumption rate. If
    charging_strategy is 'immediate' or 'delayed', EV is charged at kw constant
    charging power. If to_file==True, second-by-second average day, max peak
    load day, and min peak load day fleet charging load profiles are written to
    .csv at '../data/outputs/' and plots are written to .png at
    '../figures/'.

    Args:
        fleet (str): {'fleet1-beverage-delivery', 'fleet2-warehouse-delivery',
            'fleet3-food-delivery'}
        fleet_size (int): number of electric trucks operating per day
        charge_strat (str): {'immediate', 'delayed', 'min_power'}
        n_samples (int, optional): number of sample fleet-days to approx. the
            sampling distribution
        kwh_per_mile (float, optional): average fuel consumption rate
        kw (int, optional): if method in {'immediate', 'delayed'}, constant charging
            power level
        seed (int, optional): random seed for reproduceability
        to_file (bool, optional): if True: write load profiles to file at
            '../data/load_profiles/' and plot to file at '../figures/load_profiles/'
        y_lim (int, optional): fixes y-axis limit for plotting. If None, y-axis limit
            is set automatically
        prink_daily_energy_var (bool, optional): if True, prints peak load (kW) for min peak
            load day, average day, and max peak load day & min., mean., and max. daily energy
            requirements to STDOUT.
        agg_15min (bool, optional): if True, load profile is aggregated over
            15-min interval (avg.), else load profile second-by-second

    Returns:
        pd.DataFrame of concatenated vehicle charging schedules
    """

    global charge_profs_df, pk_load_fleet_prof_df, min_load_fleet_prof_df
    assert fleet in ['fleet1-beverage-delivery',
                     'fleet2-warehouse-delivery',
                     'fleet3-food-delivery'], "fleet not recognized!"

    assert charge_strat in ['immediate',
                            'delayed',
                            'min_power'], "charge_strat not recognized!"
    assert data_type in ['all', 'min', 'max', 'average'], "data type not recognized!"
    if agg_15min:
        res = '15min'
    else:
        res = '1s'

    # load fleet veh_op_day summaries, veh_scheds
    v_days_df = pd.read_csv(os.path.join(directory, 'data', 'fleet-schedules', f'{fleet}', 'veh_op_days.csv'))
    v_scheds_df = pd.read_csv(os.path.join(directory, 'data', 'fleet-schedules', f'{fleet}', 'veh_schedules.csv'))

    # produce random seeds
    random.seed(seed)
    rand_ints = [random.randint(0, 999) for i in range(n_samples)]

    # Color Pallette:
    if to_plot:
        if fleet == 'fleet1-beverage-delivery':  # red
            main_color = '#E77377'
            accent_color = '#f5d0d1'
        elif fleet == 'fleet2-warehouse-delivery':  # green
            main_color = '#8dccbe'
            accent_color = '#bfe3db'
        elif fleet == 'fleet3-food-delivery':  # blue
            main_color = '#355070'
            accent_color = '#a2bee0'

        # init:
    avg_veh_loads, max_veh_loads, min_veh_loads = [], [], []
    avg_fleet_loads, max_fleet_loads, min_fleet_loads = [], [], []
    total_load_all_samples = np.zeros(86399)
    max_peak_load_all_fleet_profs = 0
    max_daily_kwh_all_fleet_profs = 0
    min_peak_load_all_fleet_profs = np.inf
    min_daily_kwh_all_fleet_profs = np.inf
    if to_plot:
      fig, ax = plt.subplots(figsize=(2, 1.67))

    for rand_int in rand_ints:
        v_days_sample_df = v_days_df.sample(fleet_size,
                                            replace=True,
                                            random_state=rand_int)

        # Combine charging profiles
        charge_profs_df = pd.DataFrame()
        for i, vday in v_days_sample_df.iterrows():
            vday_sched_df = v_scheds_df[v_scheds_df.veh_op_day_id == vday.veh_op_day_id]

            if charge_strat == 'min_power':
                # Calculate daily energy consumption w/ kwh/mi assumption
                total_energy_kwh = kwh_per_mile * vday.vmt

                # Find min constant power to offset daily energy consumption
                day_off_shift_hrs = vday.time_off_shift_s / 3600
                min_power_kw = total_energy_kwh / day_off_shift_hrs

                # Extend on/off-shift reporting to sec-by-sec
                on_shift_s = []  # init
                for i, pattern in vday_sched_df.iterrows():
                    on_shift = pattern.on_shift
                    total_s = pattern.total_time_s
                    on_shift_s.extend([on_shift] * total_s)

                # Construct sec-by-sec charging profile
                charge_prof_df = pd.DataFrame({'veh_num': i,
                                               'rel_s': range(len(on_shift_s)),
                                               'on_shift': on_shift_s})

                inst_power_func = lambda x: min_power_kw if x == 0 else 0
                inst_pwr = charge_prof_df['on_shift'].apply(inst_power_func)
                charge_prof_df['kw'] = inst_pwr

            else:  # immediate or delayed charging strategies
                power_kw = kw
                three_vday_scheds_df = pd.concat([vday_sched_df] * 3).reset_index(drop=True)
                start_times, end_times, total_time_secs = [], [], []  # init
                on_shifts, vmts = [], []  # init

                i = 0
                while i < len(three_vday_scheds_df):
                    row = three_vday_scheds_df.iloc[i]
                    if i == len(three_vday_scheds_df) - 1:
                        start_times.append(row.start_time)
                        end_times.append(row.end_time)
                        total_time_secs.append(row.total_time_s)
                        on_shifts.append(row.on_shift)
                        vmts.append(row.vmt)
                        i += 1
                    else:
                        next_row = three_vday_scheds_df.iloc[i + 1]
                        if row.on_shift == next_row.on_shift:
                            start_times.append(row.start_time)
                            end_times.append(next_row.end_time)
                            time_s = row.total_time_s + next_row.total_time_s
                            total_time_secs.append(time_s)
                            on_shifts.append(row.on_shift)
                            vmts.append(row.vmt + next_row.vmt)
                            i += 2
                        else:
                            start_times.append(row.start_time)
                            end_times.append(row.end_time)
                            total_time_secs.append(row.total_time_s)
                            on_shifts.append(row.on_shift)
                            vmts.append(row.vmt)
                            i += 1

                three_vday_scheds_df = pd.DataFrame({'start_time': start_times,
                                                     'end_time': end_times,
                                                     'total_time_s': total_time_secs,
                                                     'on_shift': on_shifts,
                                                     'vmt': vmts})

                net_energy_consumed_kwh = 0  # init
                charging_power_kw, on_shift_s = [], []  # init
                for i, row in three_vday_scheds_df.iterrows():
                    if row.on_shift == 1:  # if: on-shift...
                        net_energy_consumed_kwh += (row.vmt * kwh_per_mile)  # add energy
                        charging_power_kw.extend([0] * row.total_time_s)
                        on_shift_s.extend([1] * row.total_time_s)
                    else:  # if: not on-shift...
                        req_charging_s = math.ceil(net_energy_consumed_kwh / power_kw * 3600)
                        dwell_s = row.total_time_s
                        on_shift_s.extend([0] * row.total_time_s)
                        if (req_charging_s > dwell_s) & (dwell_s > 0):  # if: required charging time > dwell time...
                            energy_charged_kwh = power_kw * dwell_s / 3600
                            net_energy_consumed_kwh -= energy_charged_kwh  # subtract energy
                            charging_power_kw.extend([power_kw] * dwell_s)
                        else:  # if: required charging time <= dwell_time...
                            net_energy_consumed_kwh = 0  # charge to full
                            if charge_strat == 'immediate':
                                charging_power_kw.extend([power_kw] * req_charging_s)
                                charging_power_kw.extend([0] * (dwell_s - req_charging_s))
                            elif charge_strat == 'delayed':
                                charging_power_kw.extend([0] * (dwell_s - req_charging_s))
                                charging_power_kw.extend([power_kw] * req_charging_s)

                charging_power_kw = charging_power_kw[86399: 172798]  # middle day
                on_shift_s = on_shift_s[86399: 172798]  # middle day

                # Construct charging profile
                charge_prof_df = pd.DataFrame({'veh_num': i,
                                               'rel_s': range(len(charging_power_kw)),
                                               'on_shift': on_shift_s,
                                               'kw': charging_power_kw})

            # Combine w/ other charging profiles
            charge_profs_df = pd.concat([charge_profs_df, charge_prof_df]).reset_index(drop=True)

        avg_veh_load_kw = charge_profs_df[charge_profs_df.on_shift == 0]['kw'].mean()
        avg_veh_loads.append(avg_veh_load_kw)
        max_veh_load_kw = max(charge_profs_df['kw'])
        max_veh_loads.append(max_veh_load_kw)

        fleet_prof_df = charge_profs_df.groupby('rel_s')['kw'].sum()
        fleet_prof_df = fleet_prof_df.reset_index()

        avg_fleet_load_kw = fleet_prof_df[fleet_prof_df.kw != 0]['kw'].mean()
        avg_fleet_loads.append(avg_fleet_load_kw)
        max_fleet_load_kw = max(fleet_prof_df['kw'])
        max_fleet_loads.append(max_fleet_load_kw)

        if max_fleet_load_kw > max_peak_load_all_fleet_profs:
            max_peak_load_all_fleet_profs = max_fleet_load_kw
            pk_load_fleet_prof_df = fleet_prof_df

        if max_fleet_load_kw < min_peak_load_all_fleet_profs:
            min_peak_load_all_fleet_profs = max_fleet_load_kw
            min_load_fleet_prof_df = fleet_prof_df

        daily_kwh = fleet_prof_df['kw'].sum() / 3600

        if daily_kwh > max_daily_kwh_all_fleet_profs:
            max_daily_kwh_all_fleet_profs = daily_kwh

        if daily_kwh < min_daily_kwh_all_fleet_profs:
            min_daily_kwh_all_fleet_profs = daily_kwh
        total_load_all_samples += np.array(fleet_prof_df['kw'])

        if to_plot:
        # Plot fleet daily load profile
            ax.plot(fleet_prof_df['rel_s'],
                    fleet_prof_df['kw'],
                    color=accent_color,
                    linewidth=0.5,
                    alpha=0.4)
        avg_load_all_samples = total_load_all_samples / n_samples
        if to_plot:
            ax.plot(range(len(avg_load_all_samples)),
                    avg_load_all_samples,
                    color=main_color)

            plt.xlim(0, len(fleet_prof_df['rel_s']))

            if ylim != None:
                plt.ylim(-0.1, ylim)
            else:
                plt.ylim(-0.1)

            ax.set_xticks(np.linspace(0, len(fleet_prof_df['rel_s']), 25)[::4])
            ax.set_xticklabels(range(0, 26)[::4], fontsize=8)
            ax.xaxis.set_minor_locator(AutoMinorLocator(4))
            plt.yticks(fontsize=8)
            plt.grid(axis='both', linestyle='--')

    if to_file == True:
        # Save plot as .png
        if to_plot == True:
            plot_fp = os.path.join(directory, 'figures', f'{fleet}_{fleet_size}vehs_{charge_strat}.png')
            plt.savefig(plot_fp, bbox_inches='tight', dpi=300)

        # Write average load profile to .csv
        times = []
        for hour in range(24):
            for minute in range(60):
                for second in range(60):
                    times.append(str(datetime.time(hour, minute, second)))
        if data_type in ['average', 'all']:
            avg_load_all_samples = list(avg_load_all_samples)
            avg_load_profile_df = pd.DataFrame({'time': times,
                                                'power_kw': [avg_load_all_samples[0]] + avg_load_all_samples})
            if agg_15min:  # aggregate load to avg. over 15-min. increments
                avg_load_profile_df = agg_15_min_load_profile(avg_load_profile_df)
            avg_lp_fp = os.path.join(directory, 'data', 'outputs', f'{fleet}_{fleet_size}vehs_avg-prof_{charge_strat}_{res}.csv')
            file_locations.append(avg_lp_fp)
            avg_load_profile_df.to_csv(avg_lp_fp, index=False)

        if data_type in ['all', 'max']:
            # Write max peak load profile to .csv
            max_pk_loads = list(pk_load_fleet_prof_df['kw'])
            pk_load_profile_df = pd.DataFrame({'time': times,
                                               'power_kw': [max_pk_loads[0]] + max_pk_loads})
            if agg_15min:  # aggregate load to avg. over 15-min. increments
                pk_load_profile_df = agg_15_min_load_profile(pk_load_profile_df)
            pk_lp_fp = os.path.join(directory, 'data', 'outputs', f'{fleet}_{fleet_size}vehs_peak-prof_{charge_strat}_{res}.csv')
            pk_load_profile_df.to_csv(pk_lp_fp, index=False)
            file_locations.append(pk_lp_fp)
        if data_type in ['min', 'all']:
        # Write min peak load profile to .csv
            min_pk_loads = list(min_load_fleet_prof_df['kw'])
            min_load_profile_df = pd.DataFrame({'time': times,
                                                'power_kw': [min_pk_loads[0]] + min_pk_loads})
            if agg_15min:  # aggregate load to avg. over 15-min. increments
                min_load_profile_df = agg_15_min_load_profile(min_load_profile_df)
            min_lp_fp = os.path.join(directory, 'data', 'outputs', f'{fleet}_{fleet_size}vehs_min-prof_{charge_strat}_{res}.csv')
            file_locations.append(min_lp_fp)
            min_load_profile_df.to_csv(min_lp_fp, index=False)
    if print_daily_energy_var:
        if data_type in ['all']:
            print('Low Bound Peak Demand (kW): {}'.format(round(min_peak_load_all_fleet_profs, 2)))
            print('Average Peak Demand (kW): {}'.format(round(np.array(avg_load_all_samples).max(), 2)))
            print('Upper Bound Peak Demand (kW): {}'.format(round(max_peak_load_all_fleet_profs, 2)))
            print()
            print('Low Bound kWh/operating day: {}'.format(round(min_daily_kwh_all_fleet_profs, 2)))
            print('Average kWh/operating day: {}'.format(round(np.array(avg_load_all_samples).sum() / 3600, 2)))
            print('Upper Bound kWh/operating day: {}'.format(round(max_daily_kwh_all_fleet_profs, 2)))
        elif data_type == 'min':
            print('Low Bound Peak Demand (kW): {}'.format(round(min_peak_load_all_fleet_profs, 2)))
            print('Low Bound kWh/operating day: {}'.format(round(min_daily_kwh_all_fleet_profs, 2)))
        elif data_type == 'max':
            print('Upper Bound Peak Demand (kW): {}'.format(round(max_peak_load_all_fleet_profs, 2)))
            print('Upper Bound kWh/operating day: {}'.format(round(max_daily_kwh_all_fleet_profs, 2)))
        else:
            print('Average Peak Demand (kW): {}'.format(round(np.array(avg_load_all_samples).max(), 2)))
            print('Average kWh/operating day: {}'.format(round(np.array(avg_load_all_samples).sum() / 3600, 2)))
    if to_plot:
        plt.show()

    return charge_profs_df


def agg_15_min_load_profile(load_profile_df):
    """
    Aggregates 1-Hz load profile by taking average demand over 15-min
    increments.
    """

    s_in_15min = 15 * 60

    # prepare idx slices
    start_idxs = np.arange(0, len(load_profile_df), s_in_15min)
    end_idxs = np.arange(s_in_15min, len(load_profile_df) + s_in_15min, s_in_15min)

    # generate list of avg kw over 15-min increments
    avg_15min_kw = []  # init
    for s_idx, e_idx in zip(start_idxs, end_idxs):
        avg_15min_kw.append(load_profile_df['power_kw'][s_idx:e_idx].mean())

    times = []  # init
    for hour in range(24):
        for minute in range(0, 60, 15):
            times.append(str(datetime.time(hour, minute, 0)))

    # create pd.DataFrame
    agg_15min_load_profile_df = pd.DataFrame({'time': times,
                                              'power_kw': avg_15min_kw})

    return agg_15min_load_profile_df
'''
def multiplot(fleet_sizes = [0], num_samples = 0, charge_strats = ['Q', 'L', 'M'], data_type = '',
                  directory = ''):
    while directory == '':
        directory = input("Enter your directory: \n"
                          "[Example: C:\\Users\\samee\\PycharmProjects\\EVtruck charging] ")
    if fleet_sizes == [0]:
        n = 3
        fleet_sizes = list(map(int,
                 input("Enter your fleet sizes [Example: 10 5 10]: ").strip().split()))[:n]
    while num_samples <= 0:
        num_samples = int(input("Enter number of samples: "))
    i=0
    while i < 3:
        if charge_strats[i] not in ['immediate', 'delayed', 'min_power']:
            charge_strats = list(map(str, input("Enter your charge strats [Example: immediate delayed immediate]: ").strip().split()))[:n]
            i = 0
        else:
            i+=1
    while data_type not in ['average', 'min', 'max']:
        data_type = input("Enter your data type [Either average, min or max]: ")
    assert len(fleet_sizes) in [1, 3], "Improper number of fleet sizes."
    assert len(charge_strats) in [1, 3], "Improper number of charge strats."
    size_dict = ['fleet1-beverage-delivery', 'fleet2-warehouse-delivery', 'fleet3-food-delivery']
    labels = ['Beverage', 'Warehouse', 'Food']
    colors = ['#E77377', '#8dccbe', '#355070']
    for p in range(3):
        generate_load_profiles(fleet = size_dict[p],
                               fleet_size = fleet_sizes[p],
                               charge_strat = charge_strats[p],
                               data_type = data_type,
                               n_samples = num_samples,
                               to_file = True,
                               to_plot = False,
                               print_daily_energy_var = False)
        print(charge_strats[p])
    fig, ax = plt.subplots(figsize=(3, 2))
    print(file_locations)
    for i, file_location in enumerate(file_locations):
        data = pd.read_csv(file_location)
        plt.plot(data['time'], data['power_kw'], label = labels[i], color = colors[i])
        plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Power (kW)')
    ax.set_xticks(np.linspace(0, len(data['time']), 25)[::4])
    ax.set_xticklabels(range(0, 26)[::4], fontsize=8)
    ax.xaxis.set_minor_locator(AutoMinorLocator(4))
    plot_fp = os.path.join(directory, 'figures',
                           'multiplot.png')
    plt.savefig(plot_fp, bbox_inches='tight', dpi=300)
    plt.show()
'''
def agg_plot(end, agg_data, scale_factor, wattage, fleet_size, charge_strat):
    fig, ax = plt.subplots(figsize=(2, 1.67))
    labels = ['Beverage', 'Warehouse', 'Food']
    colors = ['#E77377', '#8dccbe', '#355070']
    if end == 'multiplot':
        for i, file_location in enumerate(file_locations):
            data = pd.read_csv(file_location)
            plt.plot(data['time'], scale_factor*data['power_kw'], label=labels[i], color=colors[i])
        plt.legend(loc = 'upper left', fontsize = 8)
    else:
       plt.plot(agg_data['time'], agg_data['power_kw'], color=colors[0])
    plt.grid(axis='both', linestyle='--')
    plt.xlim(0, len(agg_data['time']))
    plt.xlabel('Time').set_fontsize(6)
    if wattage == 'unscaled':
        plt.ylabel('Power (kW)').set_fontsize(6)
    else:
        plt.ylabel('Power (MW)').set_fontsize(6)
    ax.set_xticks(np.linspace(0, len(agg_data['time']), 25)[::4])
    ax.set_xticklabels(range(0, 26)[::4], fontsize=8)
    ax.xaxis.set_minor_locator(AutoMinorLocator(4))
    plot_fp = os.path.join(directory, 'figures', f'{wattage}_{fleet_size}vehs_{charge_strat}_{end}.png')
    plt.savefig(plot_fp, bbox_inches='tight', dpi=300)
    plt.show()

def data_aggregate(directory = 'key',wattage = 0, fleet_sizes = [-1], charge_strats = ['key', 'Q', 'R']
             , data_type = '', res = '', num_samples = 0):
    fleets = ['fleet1-beverage-delivery', 'fleet2-warehouse-delivery', 'fleet3-food-delivery']
    while directory == 'key':
        directory = input("Enter your directory: \n"
                          "[Example: \"C:\\Users\\samee\\PycharmProjects\\EVtruck charging\" or press return to default to program location] ")
    if directory == '':
        directory = './'
    while wattage <= 0:
        wattage = input("Enter your system's max wattage (in MW): \n"
                            "(Press enter to receive unscaled data) ")
        if wattage == '':
            break
        else:
            wattage = int(wattage)
    i = 0; n = 3
    while i < 3:
        if fleet_sizes[i] < 0:
            fleet_sizes = list(map(int,
                 input("Enter your desired fleet sizes to simulate this wattage (\"x1 x2 x3\"): \n"
                       "x1 = # beverage trucks, x2 = # warehouse trucks, x3 = # food trucks. ").strip().split()))[:n]
            i = 0
        else:
            i += 1
    while num_samples <= 0:
        num_samples = int(input("Enter number of samples: "))
    i=0
    while i < 3:
        if charge_strats[i] not in ['immediate', 'delayed', 'min_power']:
            charge_strats = list(map(str, input("Enter your charge strategies (\"x1 x2 x3\") \n"
                                                "where each x is in [immediate, delayed, min_power] ").strip().split()))[:n]
            i = 0
        else:
            i+=1
    while data_type not in ['average', 'min', 'max']:
        data_type = input("Enter your data type [Either \"average\", \"min\" or \"max\"]: ")
    assert len(fleet_sizes) in [1, 3], "Improper number of fleet sizes."
    assert len(charge_strats) in [1, 3], "Improper number of charge strats."
    while res not in [True, False]:
        res = input("What resolution (\"15min\" or \"1s\") would you like to simulate this load case in? ")
        if res == '15min':
            res = True
        elif res == '1s':
            res = False
    for i in range(3):
        generate_load_profiles(fleet=fleets[i],
                               fleet_size=fleet_sizes[i],
                               charge_strat=charge_strats[i],
                               data_type=data_type,
                               n_samples=num_samples,
                               to_file=True,
                               to_plot=False,
                               print_daily_energy_var=False,
                               agg_15min = res)
    agg_data = pd.read_csv(file_locations[0])
    for file_location in file_locations[1:]:
        agg_data['power_kw'] += pd.read_csv(file_location)['power_kw']
    if wattage == '':
        scale_factor = 1
    else:
        scale_factor = wattage / agg_data['power_kw'].max()
    agg_data['power_kw']*= scale_factor
    if wattage == '':
        wattage = 'unscaled'
    else:
        wattage = str(wattage) + 'MW'
    end = 'multiplot'
    for i in range(2):
        agg_plot(end, agg_data, scale_factor, wattage, fleet_sizes, charge_strats)
        end = 'agg_plot'
    agg_data_fp = os.path.join(directory, 'data', 'agg_data', f'{wattage}_{fleet_sizes}vehs_{(str([x[0] for x in charge_strats])).upper()}_aggregate.csv')
    agg_data.to_csv(agg_data_fp, index = False)
    return agg_data
data_aggregate()
#generate_load_profiles(fleet='fleet1-beverage-delivery', n_samples = 50, to_plot = True, to_file = False, fleet_size = 10, charge_strat='delayed')
#data_aggregate(directory = './', charge_strats = ['immediate', 'delayed', 'immediate'], data_type = 'average', res = '15min',
               #num_samples = 20)