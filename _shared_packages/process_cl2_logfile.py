# Extracts data from RichardView for chlorine experiments

import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler


""" This file contains utilities for processing raw time-series logfiles and FTIR logfiles into intermediate results files (conversion files and 
bypass files) that can then be used to generate plots of CH4/Cl2 conversions, products, efficiencies, etc. The goal is to convert the raw logfiles 
into a more organized format where we can more easily extract and plot various figures of interest / merit for our system. We intended for these tools 
to only be used once for each log file, and not to need to be re-run if we change what information we wish to query from a given experiment or change 
our assumptions around some intrument's accuracy, the state of some hand-noted value during the experiment, etc. Therefore, certain functions (e.g., 
confidence interval generation) live in the cl2_utilities.py file instead.

Refer to one of the data processing notebooks to see how this package is used in practice. Basically, the user makes function calls and tweaks various 
parameters, while the package generates various diagnostic plots to help the user ensure that the data are being extracted correctly. The package is mainly 
meant for our own use, so the extraction scripts might be a bit confusing, but basically the script looks within user-specified bounds to identify when 
the UV light turned on and off, then samples the CH4, Cl2, and other gas concentrations at the appropriate times to learn how each changed with the light
cycle. The shaded regions in the top panel of the generated plots correspond to the windows in which different values are sampled. Blue corresponds to
light-off measurements, cyan corresponds to light-on measurements of Cl2 and CH4, and red corresponds to light-on measurements of CO, CH2O, and CO2, which
can only be seen when the caustic scrubbing bubbler is temporarily bypassed at a known time in each light cycle.
"""


# Extract the file data
def extract_data(info,hush=False): #Hush suppresses some diagnostic outputs
    paths = tuple(i[0] for i in info)
    labels = tuple(i[1] for i in info)
    for l in labels:
        if l[:2] not in ['01','02','03','04','05','06','07','08','09','10','11','12']:
            raise Exception(("Invalid experiment name "+str(l)+"; must begin with date in MM-DD-YY format."))
    # Extract all the data frames
    dfs = {label: pd.read_csv(path,parse_dates={'Timestamp':[0,1]}) for label,path in zip(labels,paths)}
    
    # Turn all numeric fields into floats
    for df in dfs.values():
        for k in df.keys():
            col = list(df[k])
            has_num = False
            for val in col:
                try:
                    float(val)
                    has_num=True
                    break
                except Exception:
                    pass
            if has_num:
                df[k] = pd.to_numeric(df[k],errors='coerce')
                df[k] = df[k].replace(9999, np.NaN)
        # If applicable, turn the UV light values into numbers
        if ('UV Light: Actual Status' in df.keys()):
            df['UV Light: Actual Status'] = list(1 if x=='On' else 0 for x in df['UV Light: Actual Status'])
        # Add columns of 'minutes elapsed' and 'hours elapsed', since datetimes can be annoying
        start = df['Timestamp'][0]
        df['Minutes']=list(((t-start).total_seconds()/60.0) for t in df['Timestamp'])
        df['Hours']=list(m/60.0 for m in df['Minutes'])
        
    # Print the available headers to help with writing further code
    if not hush:
        print("Successfully loaded "+str(len(paths))+" dataframes from RV2 logfiles.")
        print("Headers for the first dataframe are: ")
        keys=list(dfs[list(dfs.keys())[0]].keys())
        last=keys[len(keys)-1]
        for i, k in enumerate(keys):
            term=("." if k==last else ", ")
            print((k+term),end='')
            if i % 4 == 3: 
                print("")
        print("")
                
    # Return the dataframes
    return dfs

def apply_moving_average(dataframes,which_fields,window):
    for df in dataframes.values():
        for field in which_fields:
            label = field+" - Moving Avg"
            df[label]= df[field].rolling(window=window).mean()

# Align the plots, if desired. trimming_bounds is an array of tuples, in minutes, e.g. [(10,50),(30,90)],
def trim_dataframes(dataframes,trimming_params,plot=True,which_field='FTIR: CH4 (ppm)',which_df=0):
    trimmed_dataframes=dict()
    for key,(trimming_start,trimming_end) in zip(dataframes.keys(),trimming_params):
        if trimming_start is not None:
            df = dataframes[key].copy(deep=True)
            tdf=df[(df.Minutes>trimming_start) & (df.Minutes<(trimming_end))].copy(deep=True)
            start=list(tdf['Minutes'])[0]
            tdf['Minutes']=list((t-start) for t in tdf['Minutes'])
            tdf['Hours']=list(m/60.0 for m in tdf['Minutes'])
        else:
            tdf = dataframes[key].copy(deep=True)
        trimmed_dataframes[key]=tdf
    
    if not plot:
        return (None,None,trimmed_dataframes)
    
    # Make the plots objects
    fig, (ax1, ax2) = plt.subplots(figsize=(8,8),nrows=2, sharex=False,gridspec_kw={'height_ratios': [2, 2]})
    ax1.set_prop_cycle(None)
    ax2.set_prop_cycle(None)
    
    # Plot the aligned dataframes on the bottom axis
    color = None
    for key in list(trimmed_dataframes.keys())[:which_df+1]:
        tdf = trimmed_dataframes[key]
        lines = ax2.plot(tdf['Minutes'],tdf[which_field],label=key)
        color = lines[0].get_color()
    ax2.set_xlabel('Minutes Elapsed (Overlaid Adjusted Time Series)')
    ax2.legend(bbox_to_anchor=(1,1))
    
    # Plot the current set of data on the top axis, with markers for where it's trimmed
    key=list(dataframes.keys())[which_df]
    current_df = dataframes[key]
    ax1.plot(current_df['Minutes'],current_df[which_field],label=key,color=color) # Actual data
    for (trimming_start,trimming_end) in [trimming_params[which_df]]:
        if trimming_start is not None:
            x_vals = [trimming_start,trimming_end]
            y_vals = np.interp(x_vals,current_df['Minutes'],current_df[which_field])
            ax1.plot(x_vals,y_vals,color='m',marker='x',linestyle=' ',label='Trimming Bounds')
    ax1.set_xlabel('Minutes Elapsed (Single Time Series)')
    ax1.legend()
    
    # Return the axes
    return (trimmed_dataframes,ax1,ax2)

def find_UV_changes(UV_status,timestamps):
    on_indices = []
    off_indices = []

    for i in range(1, len(UV_status)):
        if UV_status.iloc[i] != UV_status.iloc[i-1]:  # Check if the current value is different from the previous value
            if UV_status.iloc[i] == 1:                # Transition from 0 to 1
                on_indices.append(UV_status.index[i])
            else:                                     # Transition from 1 to 0
                off_indices.append(UV_status.index[i])
    on_times = timestamps[on_indices]
    off_times = timestamps[off_indices]

    return (on_times, off_times)

class RemovalFinder:
    
    def __init__(self,dataframes,identifier,which_field=None):
        
        if identifier[:2] not in ['01','02','03','04','05','06','07','08','09','10','11','12']:
            raise Exception(("Invalid identifier for file exports "+str(identifier)+"; must begin with date in MM-YY format."))

        self.dataframes = dataframes
        self.identifier = identifier
        self.which_field = which_field
        self.out = pd.DataFrame(columns=['experiment_name','which_field','flow_rate','reaction_volume','tags','start_time','baseline','conversion','conversion_variance_due_to_noise'])
        self.tags='None'
        self.sample_points = dict()
        self.averaging_windows = dict()
        self.window_avg_values = dict()
        self.interp_baselines = dict()
        self.bypass_intervals = dict()
        self.rate_scale=1
        
    def set_setup_values(self,which_field,flow_rate=200,reaction_volume = 20*3.14159*0.25*5.7*5.7):#Assumes a 20cm long reaction zone
        self.flow_rate = flow_rate
        self.reaction_volume = reaction_volume
        self.which_field = which_field
        
    def set_tags(self,tags):
        self.tags=tags
        
    def set_rate_scale_factor(self,scale):
        # Scale the rate up by a constant
        self.rate_scale=scale

    def autofind_removal(self,which_dataframe,**kwargs):
        which_dataframe_and_suffix = which_dataframe+"; "+self.which_field
        start_time = kwargs['start_time'] if 'start_time' in kwargs.keys() else 0
        end_time = kwargs['end_time'] if 'end_time' in kwargs.keys() else 1e8
        offset = kwargs['offset'] if 'offset' in kwargs.keys() else 0
        correct_drift = kwargs['correct_drift'] if 'correct_drift' in kwargs.keys() else False
        include_last_no_correction = kwargs['include_last_no_correction'] if 'include_last_no_correction' in kwargs.keys() else False
        light_delay = kwargs['light_delay'] if 'light_delay' in kwargs.keys() else 1
        sample_duration = kwargs['sample_duration'] if 'sample_duration' in kwargs.keys() else 5
        record_no_removal = kwargs['record_no_removal'] if 'record_no_removal' in kwargs.keys() else False
         # Convert int index to string key, if needed
        if isinstance(which_dataframe, int):
            which_dataframe = list(self.dataframes.keys())[which_dataframe] # Can label either with string or integer
        # Do the calculation
        df = self.dataframes[which_dataframe]

        # Identify the times when the light turns on and off
        uv_light_data = list(df['UV Light: Actual Status'])
        toggle_indices = []
        if uv_light_data[0]==1:
            toggle_indices.append(0)
        last_change = 0
        for i in range(len(uv_light_data)-1):
            if uv_light_data[i]==0 and uv_light_data[i+1]==1: #Light turns on
                toggle_indices.append(i)
            if uv_light_data[i]==1 and uv_light_data[i+1]==0: #Light turns off
                toggle_indices.append(i)
        # Get rid of spurious rapid UV light cycles
        n = 0
        while n < len(toggle_indices)-1:
            if toggle_indices[n+1]-toggle_indices[n]<4:
                toggle_indices.pop(n)
                toggle_indices.pop(n)
            else:
                n+=1
        #toggle_indices = toggle_indices[14:16]
        
        # Turn this into windows for averaging
        averaging_windows = []
        for i in range(int(len(toggle_indices)/2)):
            light_on_time = list(df['Minutes'])[toggle_indices[2*i]]+offset
            light_off_time = list(df['Minutes'])[toggle_indices[2*i+1]]+offset
            if light_on_time<=0 or light_on_time-sample_duration-light_delay < start_time or light_off_time-light_delay+sample_duration > end_time:
                continue
            averaging_windows.append((light_on_time-sample_duration-light_delay,light_on_time-light_delay))
            averaging_windows.append((light_off_time-sample_duration-light_delay,light_off_time-light_delay))
        # We always sample for a 5-min window centered on when the light turns on, & a 10-minute window w/
        # the light turning off at the 3/4 point.

        # Do the actual averaging...
        window_avg_values = []
        window_stds = []
        for i in range(int(len(averaging_windows)/2)):
            # Get the indices of the dataframe rows corresponding to the different aveaging window limits.
            baseline_left = next(x[0] for x in enumerate(df['Minutes']) if x[1] > averaging_windows[2*i][0])
            baseline_right = next(x[0] for x in enumerate(df['Minutes']) if x[1] > averaging_windows[2*i][1])
            sample_left = next(x[0] for x in enumerate(df['Minutes']) if x[1] > averaging_windows[2*i+1][0])
            try:
                sample_right = next(x[0] for x in enumerate(df['Minutes']) if x[1] > averaging_windows[2*i+1][1])
            except StopIteration:
                sample_right = len(df['Minutes'])-1
            # Do the required averaging.
            baseline_vals = list(df[self.which_field])[baseline_left:baseline_right+1]
            baseline_conc = 1.0*sum(baseline_vals)/len(baseline_vals)
            sample_vals = list(df[self.which_field])[sample_left:sample_right+1]
            sample_avg = 1.0*sum(sample_vals)/len(sample_vals)
            avg_removal = baseline_conc - sample_avg
            # Store the averages calculated
            window_avg_values.append(baseline_conc)
            window_avg_values.append(sample_avg)
            num_unique_baseline = len(set(baseline_vals))#These lines get us to correctly record the GC readings
            num_unique_sample = len(set(sample_vals))
            window_stds.append(np.std(baseline_vals)/((num_unique_baseline**0.5)))
            window_stds.append(np.std(sample_vals)/((num_unique_sample**0.5)))

        # Cycle through and do the calculations
        interp_baselines = []
        first_at_this_concentration = True
        for i in range(int(len(window_avg_values)/2)):
            # Calculate the removal
            if not correct_drift:
                baseline_conc = window_avg_values[i*2]
                sample_avg = window_avg_values[i*2+1]
                avg_removal = baseline_conc-sample_avg
                variance_sum_contribution=(window_stds[2*i]**2 + window_stds[2*i+1]**2)
                interp_baselines.append([0.5*sum(averaging_windows[i*2+1]),sample_avg,baseline_conc])
            #elif 2*i+2>=len(window_avg_values):
            #    continue
            else:
                usually_2 = 2 if not 2*i+2>=len(window_avg_values) else 0
                next_baseline_left = averaging_windows[2*i+usually_2][0]
                baseline_conc = window_avg_values[2*i]
                baseline_right = averaging_windows[2*i][1]
                sample_avg = window_avg_values[i*2+1]
                next_baseline_conc = window_avg_values[2*i+usually_2]
                if abs((baseline_conc-next_baseline_conc)/baseline_conc)>0.3:
                    first_at_this_concentration = True
                    continue #Different concentration; can't interpolate.
                # Calculate the contribution of this data point to the total variance measured at this concentration
                if first_at_this_concentration:
                    variance_sum_contribution = ((1.0/9.0)*window_stds[2*i]**2 + window_stds[2*i+1]**2 +(4.0/9.0)*window_stds[2*i+usually_2]**2)
                else:
                    variance_sum_contribution = ((5.0/9.0)*window_stds[2*i]**2 + window_stds[2*i+1]**2 +(4.0/9.0)*window_stds[2*i+usually_2]**2)
                first_at_this_concentration=False #Teset the flag
                sample_center = 0.5*sum(averaging_windows[2*i+1])
                interp_baseline = np.interp(sample_center,[baseline_right,next_baseline_left],[baseline_conc,next_baseline_conc])
                avg_removal = interp_baseline-sample_avg
                interp_baselines.append([sample_center,sample_avg,interp_baseline])
            # Save the data
            ppm_conversion = avg_removal
            if record_no_removal:
                ppm_conversion=0
            self.out.loc[len(self.out.index)]=[which_dataframe, self.which_field, self.flow_rate, self.reaction_volume, self.tags, round(averaging_windows[2*i][0],2), baseline_conc, ppm_conversion, variance_sum_contribution]
        self.out.drop_duplicates(keep=False, inplace=True)

        # Record the averaging windows so we can plot them
        self.averaging_windows[which_dataframe_and_suffix]=averaging_windows # A list of tuples...
        self.window_avg_values[which_dataframe_and_suffix]=window_avg_values
        self.interp_baselines[which_dataframe_and_suffix]=interp_baselines
    
    def isolate_bypass_periods(self,which_dataframe,start_delay=3,ending_delay=0):
        df = self.dataframes[which_dataframe]
        which_field = "Bubbler Bypass Valve: Position Selection"
        bypass_starts = []
        bypass_ends = []
        for i in range(df.shape[0]-1):
            row = df.iloc[i]
            next_row = df.iloc[i+1]
            if row[which_field]=='Bubbler' and next_row[which_field]=='No Bubbler':
                bypass_starts.append(row["Minutes"])
            if row[which_field]=='No Bubbler' and next_row[which_field]=='Bubbler':
                bypass_ends.append(row["Minutes"])
        current_bypass_intervals = []
        for (s,e) in zip(bypass_starts,bypass_ends):
            add_me = (s+start_delay,e+ending_delay)
            if add_me[1]>add_me[0]:
                current_bypass_intervals.append(add_me)
        self.bypass_intervals[which_dataframe] = current_bypass_intervals

    def export_bypass_periods(self,folder_path,do_return=False,do_export=True):
        dfs_to_concat = []
        for which_dataframe in self.bypass_intervals.keys():
            # Get a dataframe containing only the bypass periods
            current_df = self.dataframes[which_dataframe].copy(deep=True)
            def to_keep(r):
                for (start,end) in self.bypass_intervals[which_dataframe]:
                    if (r['Minutes']>=start and r['Minutes']<=end):
                        return True
                return False
            m = current_df.apply(to_keep, axis=1)
            current_df = current_df[m]
            # Add a column for the start of the bypass sampling period
            def get_start(r):
                for (start,end) in self.bypass_intervals[which_dataframe]:
                    if (r>=start and r<=end):
                        return start
                return np.NaN
            current_df['bypass_start'] = [get_start(r) for r in current_df['Minutes']]
            # Add a column for the start of the next recent averaging window
            avg_windows = []
            for k in self.averaging_windows.keys():
                if which_dataframe in k:
                    avg_windows = self.averaging_windows[k]
                    break
            def get_last_window(b):
                prev_value=0
                for i in range(int(len(avg_windows)/2)):
                    value = round(avg_windows[2*i][0],2)
                    if value>b:
                        return prev_value
                    prev_value = value
                return prev_value
            current_df['closest_start_time'] = [get_last_window(b) for b in current_df['bypass_start']]
            current_df['experiment_name']= [which_dataframe for b in current_df['bypass_start']]
            dfs_to_concat.append(current_df)
        final_df = pd.concat(dfs_to_concat)
        final_df.reindex()
        if do_export:
            # Find the destination
            if folder_path[len(folder_path)-1]!='/':
                folder_path+="/"
            full_path = folder_path+"bypasses_"+self.identifier+".csv"
            final_df.to_csv(full_path)
        if do_return:
            return final_df

    def plot_baseline_and_samples(self,which_dataframe,which_field,color='C0', marker='^',ax=None,do_shading=True):
        # Setup plotting the underlying data
        if isinstance(which_dataframe, int) and which_dataframe>=0:
            which_dataframe = list(self.dataframes.keys())[which_dataframe] # Can label either with string or integer
        if not (which_dataframe in self.dataframes.keys()):
            return
        which_dataframe_and_suffix = which_dataframe+"; "+which_field
        if ax==None:
            (fig,ax)=plt.subplots(figsize=(14,4))
        df=self.dataframes[which_dataframe]
        ax.plot(df['Minutes'],df[self.which_field],label=self.which_field,color=color)
        # Plot the sampling locations
        if which_dataframe_and_suffix in self.sample_points.keys():
            baselines_and_samples = self.sample_points[which_dataframe_and_suffix]
            for baseline_x in baselines_and_samples.keys():
                baseline_y = np.interp(baseline_x,df['Minutes'],df[self.which_field])
                sample_xs = baselines_and_samples[baseline_x]
                sample_ys = list(np.interp(sample_xs,df['Minutes'],df[self.which_field]))
                ax.plot((baseline_x),(baseline_y),marker='D',linestyle='',color='r')
                ax.plot(sample_xs,sample_ys,marker='o',linestyle='',color='r')
                ax.plot(sorted(list(sample_xs)+[baseline_x]),[baseline_y]*(1+len(sample_xs)),linestyle=':',color='r')
                for sample_x, sample_y in zip(sample_xs,sample_ys):
                    ax.plot((sample_x,sample_x),(sample_y,baseline_y),linestyle=':',color='r')
        # Plot the averaging windows
        if (which_dataframe_and_suffix in self.averaging_windows.keys()) and do_shading:
            windows = self.averaging_windows[which_dataframe_and_suffix]
            vals = self.window_avg_values[which_dataframe_and_suffix]
            toggle = True
            for window,avg_val in zip(windows,vals):
                if do_shading:
                    ax.axvspan(window[0], window[1], color=('b' if toggle else 'c'), alpha=0.5)
                ax.plot([window[0],window[1]],[avg_val,avg_val],'k:')
                toggle = not toggle
        # Plot interpolated baselines, if drift correction used
        if which_dataframe_and_suffix in self.interp_baselines.keys():
            interp_baselines = self.interp_baselines[which_dataframe_and_suffix]
            for info in interp_baselines:
                ax.plot([info[0],info[0]],[info[1],info[2]],marker+'k')
        # Plot bypass intervals
        if (which_dataframe in self.bypass_intervals.keys()) and do_shading:
            for (bypass_start,bypass_end) in self.bypass_intervals[which_dataframe]:
                ax.axvspan(bypass_start, bypass_end, color='r', alpha=0.5)
        # Cosmetics
        ax.set_xlabel('Minutes')
        ax.xaxis.set_tick_params(labelbottom=True)
        ax.legend()
        return ax
    
    def plot_fields_during_bypass(self,which_dataframe,what_to_plot,colors,style,marker,ax=None):
        if isinstance(which_dataframe, int) and which_dataframe>=0:
            which_dataframe = list(self.dataframes.keys())[which_dataframe] # Can label either with string or integer
        if not (which_dataframe in self.dataframes.keys()):
            return
        current_df = self.dataframes[which_dataframe].copy(deep=True)
        def to_keep(r):
            for (start,end) in self.bypass_intervals[which_dataframe]:
                if (r['Minutes']>=start and r['Minutes']<=end):
                    return True
            return False
        m = current_df.apply(to_keep, axis=1)
        current_df = current_df[m]
        if ax is None:
            (fig,ax) = plt.subplots(figsize=(14,4))
        for w,c in zip(what_to_plot,colors):
            ax.plot(current_df['Minutes'],current_df[w],color=c,linestyle=style,marker=marker,label=w)
        ax.legend()
        return ax
    
    def get_dataframe(self):
        return self.out
    
    def dump_experiment(self,experiment):
        for thing in (self.sample_points,self.averaging_windows,self.window_avg_values,self.interp_baselines,self.bypass_intervals):
            for k in list(thing.keys()):
                if experiment in k:
                    thing.pop(k)
        self.out = self.out[self.out['experiment_name']!=experiment]

    def export_conversion_results_to_csv(self,folder_path,merge=True):
        # Find the destination
        if folder_path[len(folder_path)-1]!='/':
            folder_path+="/"
        full_path = folder_path+"conversions_"+self.identifier+".csv"
        # Quick export, if needed
        if not merge:
            self.out.to_csv(full_path)
            return
        # Get a list of all experiments for which we will have to poll data, and all the fields to include
        all_sample_points = []
        all_fields = []
        for i,row in self.out.iterrows():
            identifier = (row['experiment_name'],row['start_time'])
            field = row['which_field']
            if not (identifier in all_sample_points):
                all_sample_points.append(identifier)
            if not (field in all_fields):
                all_fields.append(field)
        # Make the dataframe and the header
        columns = ['experiment_name','start_time','flow_rate','reaction_volume','tags']
        for field in all_fields:
            columns+=([field+' baseline',field+' conversion',field+' conversion variance due to noise'])
        new_out = pd.DataFrame(columns=columns)
        # Add each row to the dataframe
        for (label,start_time) in all_sample_points:
            rows = self.out.loc[(self.out['experiment_name']==label) & (self.out['start_time']==start_time)]
            row = rows.iloc[0]
            line_to_add = [row['experiment_name'],row['start_time'],row['flow_rate'],row['reaction_volume'],row['tags']]
            for field in all_fields:
                row = rows[rows['which_field']==field]
                if row.empty:
                    line_to_add+=[np.NaN,np.NaN,np.NaN]
                else:
                    line_to_add+=[float(row['baseline']),float(row['conversion']),float(row['conversion_variance_due_to_noise'])]
            new_out.loc[len(new_out.index)] = line_to_add
        # Export it
        new_out.reindex()
        new_out.to_csv(full_path)

    def translate_tags(self,string):
        str_rep = ("dict("+string.replace(";",",")+")")
        return eval(str_rep)

# Rebind an FTIR logfile to a dataframe
import datetime
import numpy as np
epoch = datetime.datetime.utcfromtimestamp(0)

def rebind_ftir_prn(df,new_prn_path,fields_prn_and_df,hush=False):
    which_fields_prn = [i[0] for i in fields_prn_and_df]
    which_fields_df = [i[1] for i in fields_prn_and_df]
    prn = pd.read_csv(new_prn_path,sep='\t',parse_dates={'Timestamp':[1,2]},date_parser=lambda x: pd.to_datetime(x, errors="coerce"))
    prn = prn[pd.notnull(prn['Timestamp'])]
    def epoch_time_millis(dt):
        return (dt - epoch).total_seconds() * 1000.0
    prn_times = [epoch_time_millis(x) for x in prn['Timestamp']]
    df_times = [epoch_time_millis(x) for x in df['Timestamp']]
    for (df_field, prn_field) in zip(which_fields_df, which_fields_prn):
        prn_y_vals = list(prn[prn_field])
        prn_y_vals = [float(y) for y in prn_y_vals]
        df_y_vals = np.interp(df_times,prn_times,prn_y_vals)
        df[df_field]=df_y_vals
    if not hush:
        print("Successfully bound new PRN file to logfile dataframe.")

# Autoscale Y axis
import matplotlib.pyplot as plt
# A helper method; provided by Dan Hickstein from Stack Overflow
def autoscale_y(ax,margin=0.1):
    """This function rescales the y-axis based on the data that is visible given the current xlim of the axis.
    ax -- a matplotlib axes object
    margin -- the fraction of the total height of the y-data to pad the upper and lower ylims"""

    import numpy as np

    def get_bottom_top(line):
        xd = line.get_xdata()
        yd = line.get_ydata()
        lo,hi = ax.get_xlim()
        y_displayed = yd[((xd>lo) & (xd<hi))]
        h = np.max(y_displayed) - np.min(y_displayed)
        bot = np.min(y_displayed)-margin*h
        top = np.max(y_displayed)+margin*h
        return bot,top

    lines = ax.get_lines()
    bot,top = np.inf, -np.inf

    for line in lines:
        try:
            new_bot, new_top = get_bottom_top(line)
            if new_bot < bot: bot = new_bot
            if new_top > top: top = new_top
        except:
            pass

    ax.set_ylim(bot,top)