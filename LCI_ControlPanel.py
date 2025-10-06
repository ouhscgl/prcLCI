import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os
from datetime import date

class LSCIAnalyzer:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("LSCI Segment Tool")
        
        # Get screen dimensions
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        
        # Calculate window size and position
        window_width = screen_width - 100
        window_height = 600
        x_position = int((screen_width - window_width) / 2)
        y_position = int((screen_height - window_height) / 2)
        
        # Set window size and position
        self.root.geometry(f"{window_width}x{window_height}+{x_position}+{y_position}")

        # Create GUI elements
        self.setup_gui()
        
        # Initialize variables
        self.current_file = None
        self.points = []
        self.files_to_process = []
        self.output_data = []
        self.current_file_index = 0
        self.processed_files = set()
        self.current_directory = None
        self.last_saved_file = None
        
        # Set up key bindings for navigation
        self.root.bind('<Left>', self.previous_file)
        self.root.bind('<Right>', self.next_file)
        
    def setup_gui(self):
        # Create top control panel
        control_frame = tk.Frame(self.root)
        control_frame.pack(fill=tk.X, pady=5)
        
        # Directory selection
        tk.Button(control_frame, text="Select Directory", 
                  command=self.select_directory).pack(side=tk.LEFT, padx=5)
        
        # Input file selection
        tk.Button(control_frame, text="Load Previous Data", 
                  command=self.load_previous_data).pack(side=tk.LEFT, padx=5)
        
        # Add reset and save buttons
        tk.Button(control_frame, text="↻ Reset",
                  command=self.reset_markers).pack(side=tk.LEFT, padx=5)
        
        tk.Button(control_frame, text="↓ Save",
                  command=self.save_to_disk).pack(side=tk.LEFT, padx=5)
        
        # Add processing button
        tk.Button(control_frame, text="Run Processing", 
                  command=self.run_processing, bg='lightgreen').pack(side=tk.LEFT, padx=20)

        # Navigation buttons, labels
        tk.Button(control_frame, text=" → ", command=self.next_file).pack(side=tk.RIGHT, padx=5)
        self.nav_label = tk.Label(control_frame, text="0 / 0")
        self.nav_label.pack(side=tk.RIGHT, padx=20)
        tk.Button(control_frame, text=" ← ", command=self.previous_file).pack(side=tk.RIGHT, padx=5)

        
        # Create matplotlib figure
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Connect click event
        self.canvas.mpl_connect('button_press_event', self.on_click)
        
    def reset_markers(self):
        """Reset current markers and redraw plot"""
        self.points = []
        self.plot_data()
        
    def save_to_disk(self):
        """Save segmentation data to disk (only files with 6 markers)"""
        # Filter only complete entries (those with 6 points)
        complete_data = [entry for entry in self.output_data 
                        if all(key in entry for key in ['BaselineStart', 'BaselineEnd', 
                                                        'CompressionStart', 'CompressionEnd',
                                                        'DeflationStart', 'DeflationEnd'])]
        
        if not complete_data:
            messagebox.showwarning("No Data", "No files with complete segmentation (6 markers) to save.")
            return
            
        output_file = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            initialfile=f"lci_segment_{date.today()}.csv"
        )
        
        if output_file:
            df = pd.DataFrame(complete_data)
            df.to_csv(output_file, index=False)
            self.last_saved_file = output_file
            messagebox.showinfo("Save Complete", 
                              f"Saved {len(complete_data)} complete segmentations to {output_file}")
    
    def run_processing(self):
        """Run the LCI extraction processing"""
        try:
            from LCI_extract import LCI_extract
        except ImportError:
            messagebox.showerror("Error", "Could not import LCI_extract module. Make sure LCI_extract.py is in the same directory.")
            return
        
        # Check if we have segmentation file in memory
        segm_file = self.last_saved_file
        if not segm_file or not os.path.exists(segm_file):
            segm_file = filedialog.askopenfilename(
                title="Select Segmentation File",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
            )
            if not segm_file:
                return
        
        # Check if we have directory in memory
        raw_path = self.current_directory
        if not raw_path or not os.path.exists(raw_path):
            raw_path = filedialog.askdirectory(title="Select Data Directory")
            if not raw_path:
                return
        
        # Ask for output directory
        output_path = filedialog.askdirectory(title="Select Output Directory")
        if not output_path:
            return
        
        # Ask if plots should be generated
        generate_plots = messagebox.askyesno("Generate Plots", 
                                            "Generate analysis plots? (This may take longer)")
        
        try:
            messagebox.showinfo("Processing", "Processing started. This may take a while...")
            result = LCI_extract(
                raw_path=raw_path,
                output_path=output_path,
                generate_plots=generate_plots,
                segm_file=segm_file
            )
            
            if not result.empty:
                messagebox.showinfo("Success", 
                                   f"Processing complete! {len(result)} files processed.\nResults saved to {output_path}")
            else:
                messagebox.showwarning("Warning", "Processing completed but no data was extracted.")
                
        except Exception as e:
            messagebox.showerror("Error", f"Error during processing: {str(e)}")
    
    def load_previous_data(self):
        """Load previously processed file data to skip those files"""
        filepath = filedialog.askopenfilename(
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        
        if not filepath:
            return
            
        try:
            previous_data = pd.read_csv(filepath)
            
            # Add filenames to processed files set
            for filename in previous_data['Filename'].values:
                self.processed_files.add(filename)
                
            # Add data to output_data if not already there
            for _, row in previous_data.iterrows():
                filename = row['Filename']
                existing_index = next((i for i, d in enumerate(self.output_data) 
                                 if d['Filename'] == filename), None)
                
                if existing_index is None:
                    self.output_data.append(row.to_dict())
            
            messagebox.showinfo("Data Loaded", 
                               f"Loaded {len(previous_data)} entries. Files will be skipped during processing.")
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load file: {str(e)}")
    
    def select_directory(self):
        directory = filedialog.askdirectory()
        if directory:
            # Recursively find all CSV and XLSX files
            self.files_to_process = []
            for root, dirs, files in os.walk(directory):
                for file in files:
                    if file.endswith('.csv') or file.endswith('.xlsx'):
                        rel_path = os.path.relpath(os.path.join(root, file), 
                                                   directory)
                        
                        # Only add unprocessed files
                        if rel_path not in self.processed_files:
                            self.files_to_process.append(rel_path)
            
            self.current_directory = directory
            self.current_file_index = 0
            
            if self.files_to_process:
                self.update_nav_label()
                self.process_current_file()
            else:
                messagebox.showwarning("No files found",
                                       "No unprocessed CSV/XLSX files found in the selected directory.")
    
    def update_nav_label(self):
        total = len(self.files_to_process)
        current = self.current_file_index + 1 if self.files_to_process else 0
        self.nav_label.config(text=f"{current} / {total}")
    
    def previous_file(self, event=None):
        if not self.files_to_process:
            return
            
        # Save current points to memory if there are 6
        if len(self.points) == 6:
            self.save_current_points()
        self.points = []
        
        # Go to previous file
        self.current_file_index = max(0, self.current_file_index - 1)
        self.update_nav_label()
        self.process_current_file()
    
    def next_file(self, event=None):
        if not self.files_to_process:
            return
            
        # Save current points to memory if there are 6
        if len(self.points) == 6:
            self.save_current_points()
        self.points = []
        
        # Go to next file
        if self.current_file_index < len(self.files_to_process) - 1:
            self.current_file_index += 1
            self.update_nav_label()
            self.process_current_file()
    
    def process_current_file(self):
        if not self.files_to_process:
            return
            
        self.current_file = self.files_to_process[self.current_file_index]
        self.plot_data()
    
    def read_data_file(self, filepath):
        """Read CSV or XLSX file and return DataFrame"""
        if filepath.endswith('.xlsx'):
            # Read first sheet of Excel file
            df = pd.read_excel(filepath, sheet_name=0)
            return df
        else:
            return pd.read_csv(filepath)
    
    def plot_data(self):
        filepath = os.path.join(self.current_directory, self.current_file)
        try:
            data = self.read_data_file(filepath)
            self.current_data = data
            
            self.ax.clear()
            self.ax.plot(data['Time ms'], data['1. ROI'], label='ROI 1')
            self.ax.plot(data['Time ms'], data['2. ROI'], label='ROI 2')
            self.ax.set_title(f"Processing: {self.current_file}")
            self.ax.set_xlabel('Time (ms)')
            self.ax.set_ylabel('Value')
            self.ax.legend()
            
            # Add instruction text
            remaining_points = 6 - len(self.points)
            if remaining_points > 0:
                phase = "Baseline" if len(self.points) < 2 else "Compression" if len(self.points) < 4 else "Deflation"
                self.ax.text(0.5, 0.95, f"Click to mark {phase} points ({remaining_points} points remaining)",
                           transform=self.ax.transAxes, ha='center')
            
            # Add existing points if this file was previously marked
            existing_entry = next((entry for entry in self.output_data 
                                  if entry['Filename'] == self.current_file), None)
            
            if existing_entry and 'BaselineStart' in existing_entry:
                point_values = [
                    existing_entry['BaselineStart'],
                    existing_entry['BaselineEnd'],
                    existing_entry['CompressionStart'],
                    existing_entry['CompressionEnd'],
                    existing_entry['DeflationStart'],
                    existing_entry['DeflationEnd']
                ]
                
                for point in point_values:
                    self.ax.axvline(x=point, color='r', linestyle='--')
                
                self.ax.text(0.5, 0.9, "File has previously marked points",
                           transform=self.ax.transAxes, ha='center', color='red')
            
            self.canvas.draw()
            
        except Exception as e:
            messagebox.showerror("Error", f"Error loading file: {str(e)}")
            self.next_file()
    
    def on_click(self, event):
        if event.inaxes != self.ax:
            return
            
        self.points.append(event.xdata)
        
        # Draw vertical line at clicked point
        self.ax.axvline(x=event.xdata, color='r', linestyle='--')
        self.canvas.draw()
        
        if len(self.points) == 6:
            # Save to memory
            self.save_current_points()
            
            # Auto-advance to next file if it exists
            if self.current_file_index < len(self.files_to_process) - 1:
                self.points = []
                self.current_file_index += 1
                self.update_nav_label()
                self.process_current_file()
    
    def save_current_points(self):
        """Save current points to memory"""
        entry = {
            'Filename': self.current_file,
            'BaselineStart': self.points[0],
            'BaselineEnd': self.points[1],
            'CompressionStart': self.points[2],
            'CompressionEnd': self.points[3],
            'DeflationStart': self.points[4],
            'DeflationEnd': self.points[5]
        }
        
        # Check if we already have an entry for this file
        existing_index = next((i for i, d in enumerate(self.output_data) 
                             if d['Filename'] == self.current_file), None)
        
        if existing_index is not None:
            self.output_data[existing_index] = entry
        else:
            self.output_data.append(entry)
            
        # Mark this file as processed
        self.processed_files.add(self.current_file)
    
    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    app = LSCIAnalyzer()
    app.run()