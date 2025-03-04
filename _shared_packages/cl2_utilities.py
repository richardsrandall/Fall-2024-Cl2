import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

"""This file contains a whole bunch of utilities for generating plots using a combination of intermediate results files and 
other user-specified inputs. It's meant to be called by a notebook that generates the final plots. Much of this package is machinery for calculating 
95% confidence intervals for various measured quantities based on recorded noise during the experiment and the known accuracies of various sensing and 
control devices."""

# Store some assumptions about our instruments so that changing them propagates through all analyses
class cl2_experiment_constants:
    def __init__(self):

        # 95% CI's for various sensed quantities
        self.ftir_absolute_accuracy_95 = 1 #ppm
        self.ftir_percent_accuracy_95 = 5 #percent
        self.picarro_absolute_accuracy_95 = 0.1 #ppm
        self.picarro_percent_accuracy_95 = 5 #percent
        self.cl2_mfc_sccm_accuracy_95 = 0.5 #standard cc's per minute; based on our experience working with the device and checking it with flow meters
        self.cl2_node_percent_accuracy_95 = 4 #percent; this accounts for the fact that Cl2 is sticky, so we always wait for sensor transients to die out to within 1 mV on readings of ~25mV.

        # Hard-coded values for cost modeling
        self.cost_cl2_usage_coefficient = 1050
        self.cost_cl_radical_usage_coefficient = 840

        # Assumptions for cost modeling
        self.LCOE = 0.04
        self.LED_cost_per_kwh = 0.07
        self.reflector_efficiency = 0.90 #Assume 90% of photons find a Cl2
        self.cost_per_ton_cl2 = 150 #dollars per ton

        # Go through a bunch of unit conversions to get the cost per mole of photons
        avogadro = 6.022e23
        ev_per_j = 6.24e18 #eV per Joule
        j_per_kwh = 3.6e6
        uv_photon_energy_ev = 3.5
        LED_efficiency = 0.72
        kwh_per_mole_photons = avogadro*uv_photon_energy_ev/(ev_per_j*j_per_kwh*LED_efficiency)
        self.LED_cost_per_mole_photons = (self.LCOE+self.LED_cost_per_kwh)*kwh_per_mole_photons



# Extract conversions and 95% CI's from the conversion dataframe for gases from optical spectrometry (i.e., CH4)
# We want to combine 1) the measured noise and 2) the warranted accuracy of the instruments to get an estimate of 95% confidence intervals for these readings
# E.g., the FTIR is generally accurate to +-1% and 1 ppm, and there's also some noise in the readings.
def extract_spectrometer_data_from_conversions(conversion_dataframe,fields,percent_accuracy_95=0,absolute_accuracy_95=0):
    percent_accuracy_95 = np.array(percent_accuracy_95) if hasattr(percent_accuracy_95, '__iter__') else np.ones(conversion_dataframe.shape[0])*percent_accuracy_95
    absolute_accuracy_95 = np.array(absolute_accuracy_95) if hasattr(absolute_accuracy_95, '__iter__') else np.ones(conversion_dataframe.shape[0])*absolute_accuracy_95
    out = []
    for f in fields:
        avg_f = np.array(conversion_dataframe[f+' conversion']) # Average as stored in the dataframe
        reading_std_f = np.array(conversion_dataframe[f+' conversion variance due to noise'])**0.5 # Stdev based on instrument noise as stored in the dataframe
        percent_accuracy_std_f = 0.01*0.5*percent_accuracy_95*avg_f #Stdev based on the percent accuracy of the instrument (e.g., +-5% of reading)
        absolute_accuracy_std_f = 0.5*absolute_accuracy_95 #Stdev based on the absolute accuracy of the instrument 
        total_std_f = np.sqrt(reading_std_f**2 + percent_accuracy_std_f**2 + absolute_accuracy_std_f**2) #Total stdev from the above 3 sources
        ci_95_f = 2*total_std_f #+- 2 STD's for a 95% CI
        out.append((avg_f,ci_95_f))
    return out

# Calculate 95% CI for chlorine readings, given nominal Cl2 and measured Cl2
# Due to the approach of frequently calibrating the Cl2 sensor, using the MFC bank's trusted Cl2 concentration as a standard, we need a slightly different CI tracking approach.
def extract_cl2_data_from_conversions(conversion_dataframe,bypass_dataframe,cl2_tank_ppm,cl2_mfc_sccm_accuracy_95,cl2_node_percent_accuracy_95,override_cl2_baseline = None):
    #cl2_baseline = np.array(conversion_dataframe['Cl2 LabJack: Cl2 reading minus zero (mV) baseline'])
    #cl2_variance = np.array(conversion_dataframe['Cl2 LabJack: Cl2 reading minus zero (mV) conversion variance due to noise'])
    #cl2_conversion = np.array(conversion_dataframe['Cl2 LabJack: Cl2 reading minus zero (mV) conversion'])
    start_times = conversion_dataframe['start_time']
    means = (bypass_dataframe.groupby('closest_start_time').mean(numeric_only=True).reset_index()) 
    flows = conversion_dataframe['flow_rate']
    if override_cl2_baseline is not None:
        cl2_baseline = override_cl2_baseline
    else:
        cl2_baseline = [(tank/flow)*float(means[means.closest_start_time==t]['Cl2 MFC: Setpoint Entry']) for t,tank,flow in zip(start_times,cl2_tank_ppm,flows)]
    cl2_conversion = [conv*(nom/baseline) for nom,conv,baseline in zip(cl2_baseline,conversion_dataframe['Cl2 LabJack: Cl2 reading minus zero (mV) conversion'],conversion_dataframe['Cl2 LabJack: Cl2 reading minus zero (mV) baseline'])]
    cl2_conversion_scale = np.array(cl2_conversion)/np.array(conversion_dataframe['Cl2 LabJack: Cl2 reading minus zero (mV) conversion'])
    cl2_variance = cl2_conversion_scale*cl2_conversion_scale*np.array(conversion_dataframe['Cl2 LabJack: Cl2 reading minus zero (mV) conversion variance due to noise'])

    total_flow = np.array(conversion_dataframe['flow_rate'])
    ppm_error = np.array(cl2_tank_ppm)*(cl2_mfc_sccm_accuracy_95/total_flow) # e.g., 1000 ppm tank with 0.5 sccm accuracy and 200 total flow gives 2.5 ppm error
    fractional_error_of_baseline = ppm_error / cl2_baseline # If we're using "30ppm" as a baseline to calibrate the Cl2 sensor, that could actually have been 27.5-32.5 ppm.
    conv_percent_accuracy_std = 0.5*fractional_error_of_baseline*np.array(cl2_conversion) # The Cl2 sensor reads relative to that baseline, so we develop the 95% CI for the Cl2 conversion accordingly.
    node_accuracy_std = 0.01*0.5*cl2_node_percent_accuracy_95*np.array(cl2_conversion)
    conv_total_std = np.sqrt(conv_percent_accuracy_std**2 + cl2_variance + (node_accuracy_std)**2) # Combine the variances due to calibration and due to measured noise, the latter of which is usually quite small.
    conversion_ci_95 = 2*conv_total_std
    return (cl2_baseline, cl2_conversion, conversion_ci_95)

# Extract averages and 95% CI's from the bypass dataframe (i.e., CO, CO2, CH2O)
# We want to combine 1) the measured noise and 2) the warranted accuracy of the instrument to get an estimate of 95% confidence intervals for these readings
# E.g., the FTIR is generally accurate to +-1% and 1 ppm, and there's also some noise in the readings.
def extract_spectrometer_data_from_bypass(conversion_dataframe,bypass_dataframe,fields,percent_accuracy_95=0,absolute_accuracy_95=0):
    start_times = conversion_dataframe['start_time']
    means = (bypass_dataframe.groupby('closest_start_time').mean(numeric_only=True).reset_index())
    stdevs = ((bypass_dataframe.groupby('closest_start_time')).std(numeric_only=True).reset_index())
    percent_accuracy_95 = np.array(percent_accuracy_95) if hasattr(percent_accuracy_95, '__iter__') else np.ones(len(start_times))*percent_accuracy_95
    absolute_accuracy_95 = np.array(absolute_accuracy_95) if hasattr(absolute_accuracy_95, '__iter__') else np.ones(len(start_times))*absolute_accuracy_95
    out = []
    for f in fields:
        avg_f = np.array([float(means[means.closest_start_time==t][f]) for t in start_times]) # Average of the data points in that bypass window
        reading_std_f = np.array([float(stdevs[stdevs.closest_start_time==t][f]) for t in start_times]) # Stdev based on instrument noise
        percent_accuracy_std_f = 0.01*0.5*percent_accuracy_95*avg_f #Stdev based on the percent accuracy of the instrument (e.g., +-5% of reading)
        absolute_accuracy_std_f = 0.5*absolute_accuracy_95 #Stdev based on the absolute accuracy of the instrument 
        total_std_f = np.sqrt(reading_std_f**2 + percent_accuracy_std_f**2 + absolute_accuracy_std_f**2) #Total stdev from the above 3 sources
        ci_95_f = 2*total_std_f #+- 2 STD's for a 95% CI
        out.append((avg_f,ci_95_f))
    return out

# Helper to approximate the 95% CI of the quotient of two random variables
# Uses a formula for an uncorrelated noncentral normal ratio from here:
# https://en.wikipedia.org/wiki/Ratio_distribution#Uncorrelated_noncentral_normal_ratio
def get_95_ci_of_ratio(num,num_ci_95,denom,denom_ci_95):
    mu_x = np.array(num)
    mu_y = np.array(denom)
    var_x = (0.5*num_ci_95)**2
    var_y = (0.5*denom_ci_95)**2
    var_z = ((mu_x**2) / (mu_y**2))*(var_x/(mu_x**2) + var_y/(mu_y**2))
    ci_95_z = np.sqrt(var_z)*2
    return ci_95_z

# Helper to plot lines faster
def do_error_bar(ax,props,name,x,y,e):
    ax.errorbar(x,y,e,label=props[name]['text'],marker=props[name]['marker'],linestyle=props[name]['linestyle'],linewidth=props[name]['linewidth'],color=props[name]['color'],capsize=3)

# Helper to label lines on an existing plot
def label_lines(ax,params,locs,transaxes=False):
    #print(lines[0].__dict__.keys())
    for (l,x,y,r) in locs:
        t = params[l]['text']
        c = params[l]['color']
        if not transaxes:
            ax.text(x,y,t,rotation=r,color=c,ha='left',va='bottom')
        else:
            ax.text(x,y,t,rotation=r,color=c,ha='left',va='bottom',transform=ax.transAxes)

# Helper to plot timeseries faster
def do_timeseries(ax,props,name,x,y):
    ax.plot(x,y,linewidth=props[name]['linewidth'],color=props[name]['color'])

def label_tslines(ax,params,locs,transaxes=False):
    for (l,x,y,r) in locs:
        t = params[l]['tstext']
        c = params[l]['color']
        if not transaxes:
            ax.text(x,y,t,rotation=r,color=c,ha='left',va='bottom')
        else:
            ax.text(x,y,t,rotation=r,color=c,ha='left',va='bottom',transform=ax.transAxes)

