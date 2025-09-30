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
        x_position = int((screen_width - window_width) / 2)  # Center horizontally
        y_position = int((screen_height - window_height) / 2)  # Center vertically
        
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
        self.processed_files = set()  # Track which files have already been processed
        
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
            # Recursively find all CSV files in directory and subdirectories
            self.files_to_process = []
            for root, dirs, files in os.walk(directory):
                for file in files:
                    if file.endswith('.csv'):
                        # Store full path relative to selected directory
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
                                       "No unprocessed CSV files found in the selected directory and its subdirectories.")
    
    def update_nav_label(self):
        total = len(self.files_to_process)
        current = self.current_file_index + 1 if self.files_to_process else 0
        self.nav_label.config(text=f"{current} / {total}")
    
    def previous_file(self, event=None):
        if not self.files_to_process:
            return
            
        # Save current points if there are any
        if self.points:
            response = messagebox.askyesno("Save Points", 
                                          "Save the current points before navigating?")
            if response and len(self.points) == 6:
                self.save_current_points()
            self.points = []
        
        # Go to previous file
        self.current_file_index = max(0, self.current_file_index - 1)
        self.update_nav_label()
        self.process_current_file()
    
    def next_file(self, event=None):
        if not self.files_to_process:
            return
            
        # Save current points if there are any
        if self.points:
            response = messagebox.askyesno("Save Points", 
                                          "Save the current points before navigating?")
            if response and len(self.points) == 6:
                self.save_current_points()
            self.points = []
        
        # Go to next file or finish
        if self.current_file_index < len(self.files_to_process) - 1:
            self.current_file_index += 1
            self.update_nav_label()
            self.process_current_file()
        else:
            response = messagebox.askyesno("Finished", 
                                          "No more files to process. Save results and exit?")
            if response:
                self.save_results()
    
    def process_current_file(self):
        if not self.files_to_process:
            return
            
        self.current_file = self.files_to_process[self.current_file_index]
        self.plot_data()
    
    def plot_data(self):
        filepath = os.path.join(self.current_directory, self.current_file)
        try:
            data = pd.read_csv(filepath)
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
            
            if existing_entry:
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
            self.save_current_points()
            
            # Create custom dialog with Yes as default
            self.root.focus_force()  # Force focus on main window
            response = messagebox.askyesnocancel(
                "Continue",
                "Process next file?",
                default=messagebox.YES,
                icon=messagebox.QUESTION)
            
            if response is True:
                self.next_file()
            elif response is False:  # No
                self.save_results()
            else:  # Cancel
                self.points = []  # Clear points
                self.plot_data()  # Redraw current file
    
    def save_current_points(self):
        # Update existing entry if it exists, otherwise append new entry
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
    
    def save_results(self):
        if self.output_data:
            output_file = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
                initialfile="lci_segment" + str(date.today()) + ".csv"
            )
            
            if output_file:
                df = pd.DataFrame(self.output_data)
                df.to_csv(output_file, index=False)
                messagebox.showinfo("Save Complete", f"Data saved to {output_file}")
        
        self.root.quit()
    
    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    app = LSCIAnalyzer()
    app.run()