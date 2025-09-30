#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Laser Contrast Imaging (LCI) Data Extraction Module

This module analyzes LCI data, computing perfusion metrics and generating plots.
It can be used as an imported module, run from command line, or standalone.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import logging
from pathlib import Path
from datetime import date

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("LCI_extract")


def read_data_file(filepath):
    """
    Read CSV or XLSX file and return DataFrame.
    
    Parameters:
    -----------
    filepath : Path or str
        Path to the data file.
        
    Returns:
    --------
    pandas.DataFrame
        Loaded data.
    """
    filepath = Path(filepath)
    if filepath.suffix.lower() == '.xlsx':
        return pd.read_excel(filepath, sheet_name=0)
    else:
        return pd.read_csv(filepath)


def analyze_lsci_data(data, segm, name, generate_plots=True):
    """
    Analyze LSCI data for a single file and calculate perfusion metrics.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Raw LSCI data.
    segm : pandas.Series
        Segmentation parameters for the current data.
    name : str
        Name identifier for the current data.
    generate_plots : bool, optional
        Whether to generate plots. Default is True.
        
    Returns:
    --------
    list
        List of dictionaries containing calculated perfusion metrics.
    """
    # Initialize return variables
    perfusion_metrics = []
    
    # Remove all unnecessary fields
    columns_to_drop = ['Entire image', 'Markers', 'Pauses', 'TOIs']
    existing_columns = [col for col in columns_to_drop if col in data.columns]
    data.drop(labels=existing_columns, inplace=True, axis=1)
    
    # Iterate through ROI fields
    for enum, field in enumerate(data.columns[1:]):
        # Initiate variables
        metrics = {}
        metrics['Name'] = segm['Filename']
        metrics['Iter'] = enum + 1
        
        data['Time ms'] = pd.to_numeric(data['Time ms'], errors='coerce')
        data = data.dropna(subset=['Time ms'])
        data = data.reset_index(drop=True)
        time = data['Time ms'].values
        perfusion = data[field].values
        
        # Analyze baseline
        baseline_mask = (time >= segm['BaselineStart']) & (time <= segm['BaselineEnd'])
        metrics['baseline_avg'] = np.mean(perfusion[baseline_mask])
        
        # Analyze compression
        compress_mask = (time >= segm['CompressionStart']) & (time <= segm['CompressionEnd'])
        metrics['compress_avg'] = np.mean(perfusion[compress_mask])
        
        # Analyze Deflation
        deflation_mask = (time >= segm['DeflationStart']) & (time <= segm['DeflationEnd'])
        deflation_time = time[deflation_mask]
        
        # Polynomial fit
        ## Extract data and normalize timescale for better fit
        deflation_perf = perfusion[deflation_mask]
        deflation_time_norm = deflation_time - deflation_time[0]
        ## Polynomial fit
        degree = 6
        poly_coeffs = np.polyfit(deflation_time_norm, deflation_perf, degree)
        poly_fit = np.poly1d(poly_coeffs)
        ## Generate smooth curve using the polynomial fit
        deflation_time_smooth = np.linspace(deflation_time_norm[0], deflation_time_norm[-1], 1000)
        deflation_poly = poly_fit(deflation_time_smooth)      
        ## Rescale time to original scale
        deflation_time_smooth += deflation_time[0]
        
        # Find peak in smoothed deflation data
        peak_index = np.argmax(deflation_poly)
        peak_time = deflation_time_smooth[peak_index]
        metrics['peak_value'] = deflation_poly[peak_index]
        metrics['time_to_peak'] = peak_time - segm['DeflationStart']
        
        metrics['peak_baseline_ratio'] = metrics['peak_value'] / metrics['baseline_avg']
        metrics['peak_compress_ratio'] = metrics['peak_value'] / metrics['compress_avg']
        
        # Improved algorithm for finding time to return to baseline 
        tolerance = 1.05  # 5% above baseline
        baseline_threshold = metrics['baseline_avg'] * tolerance
        
        # Only look at data after the peak
        post_peak_poly = deflation_poly[peak_index:]
        post_peak_time = deflation_time_smooth[peak_index:]
        
        # Check if we return to baseline in our observed data
        return_indices = np.where(post_peak_poly <= baseline_threshold)[0]
        
        if len(return_indices) > 0:
            # We found a return to baseline in our data
            first_return_idx = return_indices[0]
            time_to_baseline = post_peak_time[first_return_idx] - peak_time
            extrapolated = False
        else:
            # We need to extrapolate - try exponential decay model first
            # Use the last 50% of post-peak data
            half_len = len(post_peak_poly) // 2
            fit_y = post_peak_poly[-half_len:]
            fit_x = post_peak_time[-half_len:] - peak_time  # Time relative to peak
            
            # Ensure all values are positive for log transform
            offset = 0
            min_y = np.min(fit_y)
            if min_y <= 0:
                offset = abs(min_y) + 1e-3
                fit_y = fit_y + offset
                
            # Log transform for exponential fit: y = a*exp(-b*x) + c
            log_y = np.log(fit_y)
            
            # Linear fit on log-transformed data
            try:
                slope, intercept = np.polyfit(fit_x, log_y, 1)
                a = np.exp(intercept)
                b = -slope  # b should be positive for decay
                
                if b > 0:  # Valid decay model
                    # Solve: a*exp(-b*x) - offset = baseline_threshold
                    x_intercept = -np.log((baseline_threshold + offset)/a)/b
                    
                    if x_intercept > 0:  # Valid solution
                        time_to_baseline = x_intercept
                        intersection_time = peak_time + time_to_baseline
                        extrapolated = True
                        exponential_model = True
                    else:  # Invalid solution, fall back to linear
                        exponential_model = False
                else:
                    # Not a decay, use linear extrapolation
                    exponential_model = False
            except:
                # If exponential fit fails, use linear
                exponential_model = False
                
            # Fall back to linear if exponential didn't work
            if not exponential_model:
                # Linear regression on the last third
                last_third = len(post_peak_poly) // 3
                slope, intercept = np.polyfit(post_peak_time[-last_third:], 
                                              post_peak_poly[-last_third:], 1)
                
                # Find intersection with baseline
                if slope < 0:  # Downward trend
                    intersection_time = (baseline_threshold - intercept) / slope
                    if intersection_time > post_peak_time[-1]:
                        time_to_baseline = intersection_time - peak_time
                        extrapolated = True
                    else:
                        time_to_baseline = np.nan
                        extrapolated = False
                else:  # Upward trend or flat - will never return to baseline
                    time_to_baseline = np.nan
                    extrapolated = False
                    
        metrics['time_to_base'] = time_to_baseline
                
        # Calculate area under the curve (AUC)
        metrics['auc'] = np.trapezoid(deflation_poly - metrics['baseline_avg'], deflation_time_smooth)
        
        # Calculate half-life of perfusion decay
        half_peak = (metrics['peak_value'] + metrics['baseline_avg']) / 2
        half_life_indices = np.where(deflation_poly[peak_index:] <= half_peak)[0]
        
        if len(half_life_indices) > 0:
            half_life = deflation_time_smooth[peak_index + half_life_indices[0]] - peak_time
        else:
            half_life = np.nan
            
        metrics['half_life'] = half_life
        
        perfusion_metrics.append(metrics)
            
        # Plot original and smoothed graph for comparison along with metrics
        if generate_plots:
            plt.figure(figsize=(12, 6))
            plt.plot(time, perfusion, 'b-', linewidth=1, label='Raw Data')
            
            # Determine plot endpoint
            if np.isnan(time_to_baseline):
                plot_end = deflation_time_smooth[-1]
            else:
                plot_end = max(deflation_time_smooth[-1], peak_time + time_to_baseline)
            
            plt.axhline(y=metrics['baseline_avg'], 
                        xmin=0,
                        xmax=60000/plot_end, 
                        color='g', linewidth=2, label='Baseline Avg')
            plt.axhline(y=metrics['compress_avg'],
                        xmin=60001/plot_end,
                        xmax=segm['DeflationStart']/plot_end,
                        color='m', linewidth=2, label='Compression Avg')
            
            plt.plot(deflation_time_smooth,
                     deflation_poly,
                     'r-', linewidth=2, label='Fitted Deflation')
            plt.plot(peak_time, 
                     metrics['peak_value'],
                     'ro', markersize=10, markerfacecolor='r', label='Peak')
            
            # Plot return to baseline point if available
            if not np.isnan(metrics['time_to_base']):
                plt.plot(peak_time + time_to_baseline, 
                         baseline_threshold, 
                         'go', markersize=10, markerfacecolor='r', 
                         label='Return to Baseline')
                
                # If extrapolated, show the extrapolation
                if extrapolated:
                    if 'exponential_model' in locals() and exponential_model:
                        # Generate extrapolated curve
                        extra_times = np.linspace(post_peak_time[-1], intersection_time, 100)
                        extra_y = a * np.exp(-b * (extra_times - peak_time)) - offset
                        plt.plot(extra_times, extra_y, 'r--', linewidth=2, 
                                 label='Exponential Extrapolation')
                    else:  # Linear model
                        extended_time = np.linspace(post_peak_time[-1], intersection_time, 100)
                        extended_perfusion = slope * extended_time + intercept
                        plt.plot(extended_time, extended_perfusion, 'r--', linewidth=2, 
                                 label='Linear Extrapolation')
        
            plt.xlabel('Time (ms)')
            plt.ylabel('Perfusion')
            plt.title(f'LCI Analysis: {name}')
            plt.legend()
            plt.grid(True)
            
    return perfusion_metrics


def LCI_extract(raw_path, file_maps=None, output_path=None, generate_plots=False, segm_file=None):
    """
    Extract and analyze Laser Contrast Imaging (LCI) data.
    
    Parameters:
    -----------
    raw_path : str
        Path to the directory containing raw LCI data files.
    file_maps : pandas.DataFrame or None, optional
        DataFrame with 'ID' and 'LSC' columns mapping subject IDs to filenames.
        If None, all files in the directory will be processed.
    output_path : str or None, optional
        Path to save the extracted data and plots. If None, no files are saved.
    generate_plots : bool, optional
        Whether to generate and save plots. Default is False.
    segm_file : str or None, optional
        Path to the CSV file containing segmentation parameters.
        If None, looks for 'output_data.csv' in the '../extn' directory.
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing the extracted LCI data with calculated metrics.
    """
    logger.info(f"Starting LCI extraction from {raw_path}")
    
    # Convert paths to Path objects for easier manipulation
    raw_path = Path(raw_path)
    
    # Ensure raw_path exists
    if not raw_path.exists():
        logger.error(f"Raw data path {raw_path} does not exist")
        return pd.DataFrame()
    
    # Ensure output_path exists if provided
    if output_path is not None:
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output will be saved to {output_path}")
        
    # Find segmentation file if not provided
    if segm_file is None:
        segm_file = raw_path.parent / 'extn' / 'output_data.csv'
        if not segm_file.exists():
            logger.error(f"Segmentation file not provided and default not found at {segm_file}")
            return pd.DataFrame()
    else:
        segm_file = Path(segm_file)
        if not segm_file.exists():
            logger.error(f"Provided segmentation file {segm_file} does not exist")
            return pd.DataFrame()
    
    logger.info(f"Using segmentation file: {segm_file}")
        
    # Load segmentation data
    try:
        segments = pd.read_csv(segm_file)
        logger.info(f"Loaded segmentation data with {len(segments)} entries")
    except Exception as e:
        logger.error(f"Error loading segmentation file: {str(e)}")
        return pd.DataFrame()
    
    # Process file maps if provided
    if file_maps is not None:
        logger.info("Processing file mappings")
        if isinstance(file_maps, pd.DataFrame):
            # Assuming file_maps has 'ID' and 'LSC' columns
            file_maps = file_maps.set_index('ID')['LSC']
            
            # Filter segments based on file_maps
            available_files = []
            for index, file in file_maps.items():
                # Skip if file is NaN
                if pd.isna(file):
                    continue
                available_files.append(file)
            
            # Filter segments to only include files in file_maps
            segments = segments[segments['Filename'].isin(available_files)]
            
            if segments.empty:
                logger.warning("No matching files found in the file mapping")
                return pd.DataFrame()
        else:
            logger.error("file_maps must be a pandas DataFrame with 'ID' and 'LSC' columns")
            return pd.DataFrame()
    
    # Process each segment
    all_metrics = []
    for index, segment in segments.iterrows():
        # Load in data
        try:
            fn = segment['Filename']
            logger.info(f"Processing [{index + 1}/{len(segments)}]: {fn}")
            
            # Check if raws subdirectory exists
            raws_dir = raw_path / 'raws'
            if raws_dir.exists():
                data_path = raws_dir / fn
            else:
                data_path = raw_path / fn
                
            if not data_path.exists():
                logger.warning(f"File {data_path} does not exist, skipping")
                continue
                
            data = read_data_file(data_path)
        except Exception as e:
            logger.error(f"Error reading {segment['Filename']}: {str(e)}")
            continue
        
        # Run the analysis
        try:
            name = segment['Filename'].replace('/', '').replace('\\', '"')
            metrics = analyze_lsci_data(data, segment, name, generate_plots)
            all_metrics.extend(metrics)
            
            # Save the plot if requested
            if generate_plots and output_path:
                figs_path = output_path / f'lci_analysis_{date.today()}'
                figs_path.mkdir(parents=True, exist_ok=True)
                plt.savefig(figs_path / f"{name}_plot.png")
                plt.close()
        except Exception as e:
            logger.error(f"Error analyzing {segment['Filename']}: {str(e)}")
            continue

    # Create DataFrame and pivot for final results
    logger.info(f"Completed processing {len(all_metrics)} metrics entries")
    
    if not all_metrics:
        logger.warning("No data was processed successfully")
        return pd.DataFrame()
        
    # Create final DataFrame
    df = pd.DataFrame(all_metrics)
    
    try:
        pivoted_df = df.pivot(index='Name', columns='Iter', 
                          values=df.columns.drop(['Name', 'Iter']))
        pivoted_df.columns = [f'{col[1]}_{col[0]}' for col in pivoted_df.columns]
        pivoted_df.reset_index(inplace=True)
        sorted_columns = ['Name'] + sorted(pivoted_df.columns.drop('Name'))
        pivoted_df = pivoted_df[sorted_columns]
        
        # Save to CSV if requested
        if output_path:
            output_file = output_path / f'lci_analysis_{date.today()}.csv'
            pivoted_df.to_csv(output_file, index=False)
            logger.info(f"Results saved to {output_file}")
            
        return pivoted_df
    except Exception as e:
        logger.error(f"Error creating final DataFrame: {str(e)}")
        # Return unpivoted data as fallback
        return df


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Extract Laser Contrast Imaging (LCI) data')
    
    parser.add_argument('--raw_path', type=str, required=True,
                        help='Path to directory containing raw LCI files')
    
    parser.add_argument('--file_map', type=str,
                        help='Path to CSV file mapping IDs to filenames')
    
    parser.add_argument('--output_path', type=str,
                        help='Path to save extracted data and plots')
    
    parser.add_argument('--segm_file', type=str,
                        help='Path to segmentation file')
    
    parser.add_argument('--generate_plots', action='store_true',
                        help='Generate and save analysis plots')
    
    return parser.parse_args()


def main():
    """Main function for command line execution"""
    args = parse_arguments()
    
    # Load file_maps if provided
    file_maps = None
    if args.file_map:
        try:
            file_maps = pd.read_csv(args.file_map)
            if 'ID' not in file_maps.columns or 'LSC' not in file_maps.columns:
                logger.warning("file_map CSV must have 'ID' and 'LSC' columns")
        except Exception as e:
            logger.error(f"Error loading file mappings: {str(e)}")
    
    # Call extraction function
    result = LCI_extract(
        args.raw_path, 
        file_maps=file_maps, 
        output_path=args.output_path, 
        generate_plots=args.generate_plots,
        segm_file=args.segm_file
    )
    
    # Report summary
    if not result.empty:
        logger.info(f"Extraction complete. {len(result)} rows extracted.")
    else:
        logger.warning("No data was extracted.")
    
    return result


if __name__ == "__main__":
    main()