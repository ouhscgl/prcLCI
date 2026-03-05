# Built-in libraries
import json, os, subprocess, sys
from pathlib import Path

def check_and_install_dependencies(CURRENT_VERSION):
    config_file = Path(__file__).parent / "lci_dependencies.json"
    
    required_packages = {
        "numpy"     : ">=1.19.0",
        "pandas"    : ">=1.3.0",
        "matplotlib": ">=3.3.0",
        "openpyxl"  : ">=3.0.0",
    }
    
    if config_file.exists():
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
                
            if config.get("version") == CURRENT_VERSION and config.get("dependencies_checked", False):
                return True
        except:
            pass
    
    print("\nChecking dependencies...")
    
    missing_packages = []
    for package, version_spec in required_packages.items():
        try:
            __import__(package)
        except ImportError:
            print(f"{package} is missing")
            missing_packages.append(package)
    
    if not missing_packages:
        config_data = {
            "version"               : CURRENT_VERSION,
            "dependencies_checked"  : True,
            "check_timestamp"       : os.path.getmtime(__file__),
            "python_version"        : sys.version
        }
        
        with open(config_file, 'w') as f:
            json.dump(config_data, f, indent=2)
        
        print("DONE")
        return True
    
    print(f"\nMissing packages: {', '.join(missing_packages)}")
    response = input("Would you like to install missing packages? ([y]/n): ").strip().lower()
    
    if response in ['y', 'yes', '']:
        try:
            install_cmd = [sys.executable, "-m", "pip", "install"]
            for package in missing_packages:
                install_cmd.append(package + required_packages[package])
            
            print(f"Running: {' '.join(install_cmd)}")
            result = subprocess.run(install_cmd, check=True, capture_output=True, text=True)
            print("DONE")
            
            config_data = {
                "version": CURRENT_VERSION,
                "dependencies_checked": True,
                "check_timestamp": os.path.getmtime(__file__),
                "python_version": sys.version,
                "installed_packages": missing_packages
            }
            with open(config_file, 'w') as f:
                json.dump(config_data, f, indent=2)

            return True
            
        except subprocess.CalledProcessError as e:
            print(f"Installation failed with error:\n{e.stderr}")
            print("\nPlease install the missing packages manually:")
            for package in missing_packages:
                print(f"  pip install {package}{required_packages[package]}")
            return False
    else:
        print("\nPlease install the missing packages manually:")
        for package in missing_packages:
            print(f"  pip install {package}{required_packages[package]}")
        return False


PROGRESS_FILENAME = ".lci_progress.json"

# Possible file states
STATE_UNPROCESSED = "unprocessed"
STATE_MARKED      = "marked"
STATE_REJECTED    = "rejected"

# Colors for file list
COLORS = {
    STATE_UNPROCESSED : "#FFFFFF",
    STATE_MARKED      : "#C8F7C8",
    STATE_REJECTED    : "#F7C8C8",
}
SELECTED_COLORS = {
    STATE_UNPROCESSED : "#D0D0FF",
    STATE_MARKED      : "#A0D8A0",
    STATE_REJECTED    : "#D8A0A0",
}


class LSCIAnalyzer:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("LSCI Segment Tool")
        
        # Get screen dimensions
        screen_width  = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        
        window_width  = screen_width - 100
        window_height = 600
        x_position = int((screen_width - window_width) / 2)
        y_position = int((screen_height - window_height) / 2)
        self.root.geometry(f"{window_width}x{window_height}+{x_position}+{y_position}")

        # Initialize state
        self.current_file = None
        self.current_directory = None
        self.points = []
        self.files_to_process = []   # ordered list of relative paths
        self.current_file_index = 0
        self.output_data = {}        # filename -> dict of segment points
        self.rejected_files = set()  # filenames explicitly rejected

        # Build UI
        self.setup_gui()
        
        # Key bindings
        self.root.bind('<Left>',   self.previous_file)
        self.root.bind('<Right>',  self.next_file)
        self.root.bind('<Delete>', lambda e: self.reject_file())
        self.root.bind('<BackSpace>', lambda e: self.reject_file())

    # ── State helpers ───────────────────────────────────────────────

    @property
    def processed_files(self):
        return set(self.output_data.keys())

    def file_state(self, filename):
        if filename in self.rejected_files:
            return STATE_REJECTED
        if filename in self.output_data:
            return STATE_MARKED
        return STATE_UNPROCESSED

    # ── Progress persistence ────────────────────────────────────────

    def _progress_path(self):
        if self.current_directory:
            return Path(self.current_directory) / PROGRESS_FILENAME
        return None

    def _save_progress(self):
        path = self._progress_path()
        if path is None:
            return
        data = {
            "segments": self.output_data,
            "rejected": sorted(self.rejected_files),
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    def _load_progress(self):
        path = self._progress_path()
        if path is None or not path.exists():
            return
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            self.output_data = data.get("segments", {})
            self.rejected_files = set(data.get("rejected", []))
        except Exception as e:
            messagebox.showwarning("Progress", f"Could not load progress file:\n{e}")

    # ── GUI setup ───────────────────────────────────────────────────

    def setup_gui(self):
        # Top control bar
        control_frame = tk.Frame(self.root)
        control_frame.pack(fill=tk.X, pady=5)
        
        tk.Button(control_frame, text="Select Directory", 
                  command=self.select_directory).pack(side=tk.LEFT, padx=5)
        
        tk.Button(control_frame, text="↻ Reset Markers",
                  command=self.reset_markers).pack(side=tk.LEFT, padx=5)
        
        tk.Button(control_frame, text="✕ Reject File",
                  command=self.reject_file).pack(side=tk.LEFT, padx=5)
        
        tk.Button(control_frame, text="Export CSV",
                  command=self.export_csv).pack(side=tk.LEFT, padx=5)

        tk.Button(control_frame, text="Run Processing", 
                  command=self.run_processing, bg='lightgreen').pack(side=tk.LEFT, padx=20)

        # Current filename display
        self.file_label = tk.Label(control_frame, text="", font=("Courier", 10), anchor='w')
        self.file_label.pack(side=tk.LEFT, padx=10)

        # Main content area: side panel + plot
        content_frame = tk.Frame(self.root)
        content_frame.pack(fill=tk.BOTH, expand=True)

        # Side panel
        side_frame = tk.Frame(content_frame, width=320)
        side_frame.pack(side=tk.LEFT, fill=tk.Y)
        side_frame.pack_propagate(False)

        self.files_label = tk.Label(side_frame, text="Files", font=("Courier", 10, "bold"))
        self.files_label.pack(pady=(5, 2))

        list_container = tk.Frame(side_frame)
        list_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        scrollbar = tk.Scrollbar(list_container, orient=tk.VERTICAL)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.file_listbox = tk.Listbox(
            list_container,
            yscrollcommand=scrollbar.set,
            activestyle='none',
            font=("Courier", 12),
            selectmode=tk.SINGLE,
            fg='black',
            selectforeground='black',
        )
        self.file_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.file_listbox.yview)

        self.file_listbox.bind('<<ListboxSelect>>', self._on_listbox_select)

        # Plot area
        plot_frame = tk.Frame(content_frame)
        plot_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.canvas.mpl_connect('button_press_event', self.on_click)
        self.canvas.mpl_connect('motion_notify_event', self._on_mouse_move)
        self.canvas.mpl_connect('draw_event', self._on_draw)
        self._bg = None
        self._cursor_line = None
        self._cursor_text = None

    # ── Side panel management ───────────────────────────────────────

    def _populate_file_list(self):
        self.file_listbox.delete(0, tk.END)
        for fn in self.files_to_process:
            # Show just the basename for readability, full relative path as tooltip-ish
            display = fn if len(fn) < 45 else "…" + fn[-42:]
            self.file_listbox.insert(tk.END, display)
        self._refresh_file_colors()

    def _refresh_file_colors(self):
        for i, fn in enumerate(self.files_to_process):
            state = self.file_state(fn)
            is_selected = (i == self.current_file_index)
            bg = SELECTED_COLORS[state] if is_selected else COLORS[state]
            self.file_listbox.itemconfig(i, bg=bg)

    def _on_listbox_select(self, event):
        sel = self.file_listbox.curselection()
        if not sel:
            return
        index = sel[0]
        if index == self.current_file_index:
            return
        # Save current work before navigating
        if len(self.points) == 6:
            self.save_current_points()
        self.points = []
        self.current_file_index = index
        self.update_nav_label()
        self.process_current_file()

    # ── Directory / file loading ────────────────────────────────────

    def select_directory(self):
        directory = filedialog.askdirectory()
        if not directory:
            return

        self.current_directory = directory
        self.output_data = {}
        self.rejected_files = set()

        # Load any saved progress first
        self._load_progress()

        # Discover files
        self.files_to_process = []
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith('.csv') or file.endswith('.xlsx'):
                    rel_path = os.path.relpath(os.path.join(root, file), directory)
                    self.files_to_process.append(rel_path)

        self.files_to_process.sort()
        self.current_file_index = 0
        self._populate_file_list()

        if self.files_to_process:
            self.update_nav_label()
            self.process_current_file()
        else:
            messagebox.showwarning("No files found",
                                   "No CSV/XLSX files found in the selected directory.")
    
    def update_nav_label(self):
        total = len(self.files_to_process)
        done = len(self.output_data) + len(self.rejected_files)
        width = len(str(total))
        self.files_label.config(text=f"Files [{done:0{width}d}/{total}]")
        if self.current_file:
            self.file_label.config(text=self.current_file)

    # ── Navigation ──────────────────────────────────────────────────

    def previous_file(self, event=None):
        if not self.files_to_process:
            return
        if len(self.points) == 6:
            self.save_current_points()
        self.points = []
        self.current_file_index = max(0, self.current_file_index - 1)
        self.update_nav_label()
        self.process_current_file()
    
    def next_file(self, event=None):
        if not self.files_to_process:
            return
        if len(self.points) == 6:
            self.save_current_points()
        self.points = []
        if self.current_file_index < len(self.files_to_process) - 1:
            self.current_file_index += 1
            self.update_nav_label()
            self.process_current_file()

    # ── Core actions ────────────────────────────────────────────────

    def process_current_file(self):
        if not self.files_to_process:
            return
        self.current_file = self.files_to_process[self.current_file_index]
        self.file_label.config(text=self.current_file)
        self.plot_data()
        self._refresh_file_colors()
        # Keep listbox selection in sync
        self.file_listbox.selection_clear(0, tk.END)
        self.file_listbox.selection_set(self.current_file_index)
        self.file_listbox.see(self.current_file_index)

    def reset_markers(self):
        self.points = []
        # Also remove from saved data if present
        if self.current_file and self.current_file in self.output_data:
            del self.output_data[self.current_file]
            self._save_progress()
        # Un-reject if it was rejected
        self.rejected_files.discard(self.current_file)
        self.plot_data()
        self._refresh_file_colors()
        self.update_nav_label()

    def reject_file(self):
        if not self.current_file:
            return
        self.points = []
        # Remove segment data if any
        self.output_data.pop(self.current_file, None)
        self.rejected_files.add(self.current_file)
        self._save_progress()
        self._refresh_file_colors()
        # Auto-advance
        if self.current_file_index < len(self.files_to_process) - 1:
            self.current_file_index += 1
            self.update_nav_label()
            self.process_current_file()
        else:
            self.plot_data()
            self.update_nav_label()

    def save_current_points(self):
        self.output_data[self.current_file] = {
            'Filename'        : self.current_file,
            'BaselineStart'   : self.points[0],
            'BaselineEnd'     : self.points[1],
            'CompressionStart': self.points[2],
            'CompressionEnd'  : self.points[3],
            'DeflationStart'  : self.points[4],
            'DeflationEnd'    : self.points[5],
        }
        self.rejected_files.discard(self.current_file)
        self._save_progress()
        self._refresh_file_colors()
        self.update_nav_label()

    # ── Plot ────────────────────────────────────────────────────────

    def read_data_file(self, filepath):
        if filepath.endswith('.xlsx'):
            return pd.read_excel(filepath, sheet_name=0)
        return pd.read_csv(filepath)

    def plot_data(self):
        filepath = os.path.join(self.current_directory, self.current_file)
        try:
            data = self.read_data_file(filepath)
            self.current_data = data
            
            time_s = data['Time ms'] / 1000.0
            
            self.ax.clear()
            self.ax.plot(time_s, data['1. ROI'], label='ROI 1')
            self.ax.plot(time_s, data['2. ROI'], label='ROI 2')
            
            # Small axis annotation instead of labels
            self.ax.text(0.005, 0.01, 'time (s) / value',
                        transform=self.ax.transAxes, fontsize=6, color='gray',
                        va='bottom', ha='left')
            
            # Shrink tick labels and tighten margins
            self.ax.tick_params(axis='both', labelsize=7, pad=2)
            self.ax.legend(fontsize=7, loc='upper right')
            self.fig.subplots_adjust(left=0.05, right=0.995, top=0.995, bottom=0.06)
            
            # Clamp x-axis to data range
            self.ax.set_xlim(time_s.iloc[0], time_s.iloc[-1])
            
            state = self.file_state(self.current_file)

            if state == STATE_REJECTED:
                self.ax.text(0.005, 0.97, "REJECTED (reset to re-enable)",
                           transform=self.ax.transAxes, ha='left', va='top',
                           color='red', fontsize=7, fontweight='bold')

            elif state == STATE_MARKED:
                entry = self.output_data[self.current_file]
                for key in ['BaselineStart','BaselineEnd','CompressionStart',
                            'CompressionEnd','DeflationStart','DeflationEnd']:
                    self.ax.axvline(x=entry[key] / 1000.0, color='r', linestyle='--', linewidth=0.8)
                self.ax.text(0.005, 0.97, "Segmented (reset to redo)",
                           transform=self.ax.transAxes, ha='left', va='top',
                           color='green', fontsize=7)

            else:
                for pt in self.points:
                    self.ax.axvline(x=pt / 1000.0, color='r', linestyle='--', linewidth=0.8)
            
            # Set up animated cursor elements (not drawn by canvas.draw)
            from matplotlib.lines import Line2D
            from matplotlib.transforms import blended_transform_factory
            trans = blended_transform_factory(self.ax.transData, self.ax.transAxes)
            
            self._cursor_line = Line2D([0, 0], [0, 1], transform=trans,
                                       color='gray', linewidth=0.7, alpha=0.5,
                                       animated=True, visible=False)
            self.ax.add_line(self._cursor_line)
            
            self._cursor_text = self.ax.text(0, 0.97, '', transform=trans,
                                             fontsize=6, ha='center', va='top',
                                             animated=True, visible=False)
            
            self.canvas.draw()
            self._bg = self.canvas.copy_from_bbox(self.fig.bbox)
            
        except Exception as e:
            messagebox.showerror("Error", f"Error loading file: {str(e)}")

    def _on_draw(self, event):
        """Recapture background after any full redraw (e.g. resize)."""
        self._bg = self.canvas.copy_from_bbox(self.fig.bbox)

    PHASE_COLORS = {
        'Baseline'   : '#4477CC',
        'Compression': '#CC7722',
        'Deflation'  : '#22AA55',
    }

    def _on_mouse_move(self, event):
        if self._bg is None or self._cursor_line is None:
            return

        # Hide cursor when outside axes or file not clickable
        if (event.inaxes != self.ax
                or not self.current_file
                or self.file_state(self.current_file) != STATE_UNPROCESSED):
            if self._cursor_line.get_visible():
                self._cursor_line.set_visible(False)
                self._cursor_text.set_visible(False)
                self.canvas.restore_region(self._bg)
                self.canvas.blit(self.fig.bbox)
            return

        # Clamp x to axis data range
        xlim = self.ax.get_xlim()
        x = max(xlim[0], min(xlim[1], event.xdata))

        # Phase info
        n = len(self.points)
        PHASE_NAMES = [
            'Baseline start', 'Baseline end',
            'Compression start', 'Compression end',
            'Deflation start', 'Deflation end',
        ]
        phase = PHASE_NAMES[n]
        remaining = 6 - n
        color = self.PHASE_COLORS[phase.split()[0]]

        self._cursor_line.set_xdata([x, x])
        self._cursor_line.set_color(color)
        self._cursor_line.set_visible(True)

        self._cursor_text.set_position((x, 0.97))
        self._cursor_text.set_text(f"{phase} · {remaining}")
        self._cursor_text.set_color(color)
        self._cursor_text.set_visible(True)

        self.canvas.restore_region(self._bg)
        self.ax.draw_artist(self._cursor_line)
        self.ax.draw_artist(self._cursor_text)
        self.canvas.blit(self.fig.bbox)

    def on_click(self, event):
        if event.inaxes != self.ax:
            return
        if self.file_state(self.current_file) != STATE_UNPROCESSED:
            return
        
        # Store in ms, display in seconds
        ms_value = event.xdata * 1000.0
        self.points.append(ms_value)
        self.ax.axvline(x=event.xdata, color='r', linestyle='--', linewidth=0.8)
        
        # Redraw static content and recapture background for cursor blitting
        self.canvas.draw()
        self._bg = self.canvas.copy_from_bbox(self.fig.bbox)
        
        if len(self.points) == 6:
            self.save_current_points()
            # Auto-advance
            if self.current_file_index < len(self.files_to_process) - 1:
                self.points = []
                self.current_file_index += 1
                self.update_nav_label()
                self.process_current_file()

    # ── Export ──────────────────────────────────────────────────────

    def export_csv(self):
        complete = list(self.output_data.values())
        if not complete:
            messagebox.showwarning("No Data", "No segmented files to export.")
            return
            
        output_file = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            initialfile=f"lci_segment_{date.today()}.csv"
        )
        if output_file:
            df = pd.DataFrame(complete)
            df.to_csv(output_file, index=False)
            messagebox.showinfo("Export Complete", 
                              f"Exported {len(complete)} segmentations to {output_file}")

    # ── Processing ──────────────────────────────────────────────────

    def run_processing(self):
        try:
            from LCI_extract import LCI_extract
        except ImportError:
            messagebox.showerror("Error", 
                "Could not import LCI_extract module.\n"
                "Make sure LCI_extract.py is in the same directory.")
            return
        
        # Export a temp CSV for LCI_extract to consume
        complete = list(self.output_data.values())
        if not complete:
            messagebox.showwarning("No Data", "No segmented files to process.")
            return

        raw_path = self.current_directory
        if not raw_path or not os.path.exists(raw_path):
            raw_path = filedialog.askdirectory(title="Select Data Directory")
            if not raw_path:
                return
        
        output_path = filedialog.askdirectory(title="Select Output Directory")
        if not output_path:
            return
        
        generate_plots = messagebox.askyesno("Generate Plots", 
                                            "Generate analysis plots? (This may take longer)")
        
        # Write temp segmentation file
        segm_file = Path(output_path) / f"_segmentation_{date.today()}.csv"
        pd.DataFrame(complete).to_csv(segm_file, index=False)

        try:
            messagebox.showinfo("Processing", "Processing started. This may take a while...")
            result = LCI_extract(
                raw_path=raw_path,
                output_path=output_path,
                generate_plots=generate_plots,
                segm_file=str(segm_file)
            )
            
            if not result.empty:
                messagebox.showinfo("Success", 
                    f"Processing complete! {len(result)} files processed.\n"
                    f"Results saved to {output_path}")
            else:
                messagebox.showwarning("Warning", "Processing completed but no data was extracted.")
                
        except Exception as e:
            messagebox.showerror("Error", f"Error during processing: {str(e)}")
    
    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    CURRENT_VERSION = "4.0"
    if not check_and_install_dependencies(CURRENT_VERSION):
        print("\nDependency check failed. Exiting...")
        sys.exit(1)
    
    print("\nOpening application...")
    import tkinter as tk
    from tkinter import filedialog, messagebox
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from datetime import date
    app = LSCIAnalyzer()
    print("DONE")
    app.run()
