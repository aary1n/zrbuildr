#!/usr/bin/env python3

# Simple Tkinter GUI for Zr/ZrO₂ Interface Generation.
# Initialise .venv: Type '.venv\Scripts\activate' in terminal.
# Type 'python gui.py' in the terminal to run the GUI.


import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
import numpy as np
from interface_builder import ZrZrO2InterfaceBuilder, visualize_interface
import os

class InterfaceGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Zr/ZrO₂ Interface Builder")
        self.root.geometry("600x500")
        
        # Initialize builder
        self.builder = ZrZrO2InterfaceBuilder()
        self.current_interface = None
        
        self.create_widgets()
        
    def create_widgets(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Title
        title_label = ttk.Label(main_frame, text="Zr/ZrO₂ Interface Builder", 
                               font=('Arial', 16, 'bold'))
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))
        
        # Parameter input frame
        param_frame = ttk.LabelFrame(main_frame, text="Interface Parameters", padding="10")
        param_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Zr surface
        ttk.Label(param_frame, text="Zr Surface:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.zr_surface_var = tk.StringVar(value="1010")  # Changed to better default
        self.zr_combo = ttk.Combobox(param_frame, textvariable=self.zr_surface_var, 
                               values=["0001", "1010", "1120"], width=15, state="readonly")
        self.zr_combo.grid(row=0, column=1, sticky=tk.W, padx=(10, 0), pady=5)
        
        # ZrO2 surface  
        ttk.Label(param_frame, text="ZrO₂ Surface:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.zro2_surface_var = tk.StringVar(value="100")  # Changed to better default
        self.zro2_combo = ttk.Combobox(param_frame, textvariable=self.zro2_surface_var,
                                 values=["111", "100", "110", "001"], width=15, state="readonly")
        self.zro2_combo.grid(row=1, column=1, sticky=tk.W, padx=(10, 0), pady=5)
        
        # ZrO2 phase
        ttk.Label(param_frame, text="ZrO₂ Phase:").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.zro2_phase_var = tk.StringVar(value="tetragonal")
        self.phase_combo = ttk.Combobox(param_frame, textvariable=self.zro2_phase_var,
                                  values=["tetragonal", "monoclinic", "cubic"], width=15, state="readonly")
        self.phase_combo.grid(row=2, column=1, sticky=tk.W, padx=(10, 0), pady=5)
        
        # Zr layers
        ttk.Label(param_frame, text="Zr Layers:").grid(row=3, column=0, sticky=tk.W, pady=5)
        self.zr_layers_var = tk.StringVar(value="6")
        zr_layers_spin = ttk.Spinbox(param_frame, from_=3, to=15, textvariable=self.zr_layers_var, width=15)
        zr_layers_spin.grid(row=3, column=1, sticky=tk.W, padx=(10, 0), pady=5)
        
        # ZrO2 layers
        ttk.Label(param_frame, text="ZrO₂ Layers:").grid(row=4, column=0, sticky=tk.W, pady=5)
        self.zro2_layers_var = tk.StringVar(value="4")
        zro2_layers_spin = ttk.Spinbox(param_frame, from_=3, to=15, textvariable=self.zro2_layers_var, width=15)
        zro2_layers_spin.grid(row=4, column=1, sticky=tk.W, padx=(10, 0), pady=5)
        
        # Gap
        ttk.Label(param_frame, text="Interface Gap (Å):").grid(row=5, column=0, sticky=tk.W, pady=5)
        self.gap_var = tk.StringVar(value="2.5")
        gap_spin = ttk.Spinbox(param_frame, from_=1.0, to=5.0, increment=0.1, 
                              textvariable=self.gap_var, width=15)
        gap_spin.grid(row=5, column=1, sticky=tk.W, padx=(10, 0), pady=5)
        
        # Sandwich mode
        ttk.Label(param_frame, text="Structure Type:").grid(row=6, column=0, sticky=tk.W, pady=5)
        self.sandwich_var = tk.BooleanVar(value=False)
        # Advanced options
        adv_frame = ttk.LabelFrame(main_frame, text="Advanced Options", padding="10")
        adv_frame.grid(row=7, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        self.true_tet_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(adv_frame, text="Use true tetragonal ZrO₂ surfaces (more realistic)", variable=self.true_tet_var).grid(row=0, column=0, sticky=tk.W)
        self.remove_overlaps_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(adv_frame, text="Remove unphysical overlaps (can alter edge rows)", variable=self.remove_overlaps_var).grid(row=1, column=0, sticky=tk.W)
        self.compact_wrap_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(adv_frame, text="Compact in-plane wrap (fill cell more)", variable=self.compact_wrap_var).grid(row=2, column=0, sticky=tk.W)
        sandwich_frame = ttk.Frame(param_frame)
        sandwich_frame.grid(row=6, column=1, sticky=tk.W, padx=(10, 0), pady=5)
        
        self.normal_radio = ttk.Radiobutton(sandwich_frame, text="Normal (Zr|ZrO₂)", 
                                          variable=self.sandwich_var, value=False)
        self.normal_radio.grid(row=0, column=0, sticky=tk.W)
        
        self.sandwich_radio = ttk.Radiobutton(sandwich_frame, text="Sandwich (ZrO₂|Zr|ZrO₂)", 
                                            variable=self.sandwich_var, value=True)
        self.sandwich_radio.grid(row=1, column=0, sticky=tk.W)
        
        # Buttons frame
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=2, column=0, columnspan=2, pady=20)
        
        # Generate button
        self.generate_btn = ttk.Button(button_frame, text="Generate Interface", 
                                      command=self.generate_interface, width=20)
        self.generate_btn.grid(row=0, column=0, padx=(0, 10))
        
        # Visualize button
        self.visualize_btn = ttk.Button(button_frame, text="Visualize", 
                                       command=self.visualize_interface_gui, 
                                       width=15, state="disabled")
        self.visualize_btn.grid(row=0, column=1, padx=(0, 10))
        
        # Save button
        self.save_btn = ttk.Button(button_frame, text="Save", 
                                  command=self.save_interface, 
                                  width=15, state="disabled")
        self.save_btn.grid(row=0, column=2)

        # Save components button
        self.save_comp_btn = ttk.Button(button_frame, text="Save Parts", 
                                      command=self.save_components, 
                                      width=15, state="disabled")
        self.save_comp_btn.grid(row=0, column=3, padx=(10, 0))
        
        # Progress bar
        self.progress = ttk.Progressbar(main_frame, mode='indeterminate')
        self.progress.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Status/results frame
        result_frame = ttk.LabelFrame(main_frame, text="Results", padding="10")
        result_frame.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        
        # Results text
        self.result_text = tk.Text(result_frame, height=8, width=60, wrap=tk.WORD)
        scrollbar = ttk.Scrollbar(result_frame, orient=tk.VERTICAL, command=self.result_text.yview)
        self.result_text.configure(yscrollcommand=scrollbar.set)
        
        self.result_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(4, weight=1)
        result_frame.columnconfigure(0, weight=1)
        result_frame.rowconfigure(0, weight=1)
        
        # Initial message
        self.result_text.insert(tk.END, "Welcome to the Zr/ZrO₂ Interface Builder!\n\n")
        self.result_text.insert(tk.END, "1. Set your desired parameters above\n")
        self.result_text.insert(tk.END, "2. Choose Normal or Sandwich structure type\n")
        self.result_text.insert(tk.END, "3. Click 'Generate Interface' to create the structure\n")
        self.result_text.insert(tk.END, "4. Use 'Visualize' to view the interface\n")
        self.result_text.insert(tk.END, "5. Use 'Save' to export as CIF, POSCAR, or XYZ\n\n")
        self.result_text.insert(tk.END, "Structure Types:\n")
        self.result_text.insert(tk.END, "• Normal: Zr | ZrO₂ interface\n")
        self.result_text.insert(tk.END, "• Sandwich: ZrO₂ | Zr | ZrO₂ structure\n\n")
        
    def log_message(self, message):
        # Add message to results text box
        self.result_text.insert(tk.END, f"{message}\n")
        self.result_text.see(tk.END)
        self.root.update()
        
    def validate_parameters(self):
        # Validate and return current GUI parameters
        try:
            # Get all parameters
            zr_surface = self.zr_surface_var.get().strip()
            zro2_surface = self.zro2_surface_var.get().strip()
            zro2_phase = self.zro2_phase_var.get().strip()
            zr_layers = int(self.zr_layers_var.get())
            zro2_layers = int(self.zro2_layers_var.get())
            gap = float(self.gap_var.get())
            sandwich_mode = self.sandwich_var.get()
            
            # Validate ranges
            if zr_layers < 3 or zr_layers > 15:
                raise ValueError(f"Zr layers must be between 3-15, got {zr_layers}")
            if zro2_layers < 3 or zro2_layers > 15:
                raise ValueError(f"ZrO₂ layers must be between 3-15, got {zro2_layers}")
            if gap < 1.0 or gap > 5.0:
                raise ValueError(f"Gap must be between 1.0-5.0 Å, got {gap}")
                
            # Validate surface orientations
            valid_zr_surfaces = ["0001", "1010", "1120"]
            valid_zro2_surfaces = ["111", "100", "110", "001"]
            valid_phases = ["tetragonal", "monoclinic", "cubic"]
            
            if zr_surface not in valid_zr_surfaces:
                raise ValueError(f"Invalid Zr surface: {zr_surface}")
            if zro2_surface not in valid_zro2_surfaces:
                raise ValueError(f"Invalid ZrO₂ surface: {zro2_surface}")
            if zro2_phase not in valid_phases:
                raise ValueError(f"Invalid ZrO₂ phase: {zro2_phase}")
            
            return {
                'zr_surface': zr_surface,
                'zro2_surface': zro2_surface,
                'zro2_phase': zro2_phase,
                'zr_layers': zr_layers,
                'zro2_layers': zro2_layers,
                'gap': gap,
                'sandwich_mode': sandwich_mode
            }
            
        except Exception as e:
            raise ValueError(f"Parameter validation failed: {str(e)}")
        
    def generate_interface(self):
        # Generate interface in a separate thread
        def worker():
            try:
                self.progress.start()
                self.generate_btn.config(state="disabled")
                
                # Validate and get parameters
                params = self.validate_parameters()
                
                # Debug: Show what parameters we're actually using
                self.log_message(f"DEBUG - Parameters read from GUI:")
                self.log_message(f"  Zr Surface: '{params['zr_surface']}'")
                self.log_message(f"  ZrO₂ Surface: '{params['zro2_surface']}'")
                self.log_message(f"  ZrO₂ Phase: '{params['zro2_phase']}'")
                self.log_message(f"  Zr Layers: {params['zr_layers']}")
                self.log_message(f"  ZrO₂ Layers: {params['zro2_layers']}")
                self.log_message(f"  Gap: {params['gap']}Å")
                self.log_message(f"  Sandwich Mode: {params['sandwich_mode']}")
                self.log_message("")
                
                structure_type = "sandwich" if params['sandwich_mode'] else "normal"
                self.log_message(f"Generating {structure_type} Zr({params['zr_surface']})/ZrO₂({params['zro2_surface']}) interface...")
                self.log_message(f"Parameters: {params['zr_layers']} Zr layers, {params['zro2_layers']} ZrO₂ layers, {params['gap']}Å gap, Mode: {structure_type}")
                
                # Set builder options
                try:
                    self.builder.set_options(
                        use_true_tetragonal_surfaces=self.true_tet_var.get(),
                        remove_overlaps=self.remove_overlaps_var.get(),
                        compact_wrap=self.compact_wrap_var.get(),
                    )
                except AttributeError:
                    # Backward compatibility: older builder without set_options
                    pass

                # Generate interface with validated parameters
                if params['sandwich_mode']:
                    result = self.builder.build_sandwich_interface(
                        zr_surface=params['zr_surface'],
                        zro2_surface=params['zro2_surface'],
                        zro2_phase=params['zro2_phase'],
                        zr_layers=params['zr_layers'],
                        zro2_layers=params['zro2_layers'],
                        gap=params['gap']
                    )
                else:
                    result = self.builder.build_interface(
                        zr_surface=params['zr_surface'],
                        zro2_surface=params['zro2_surface'],
                        zro2_phase=params['zro2_phase'],
                        zr_layers=params['zr_layers'],
                        zro2_layers=params['zro2_layers'],
                        gap=params['gap']
                    )
                
                self.current_interface = result
                
                # Display results
                metadata = result['metadata']
                self.log_message(f"✓ Interface generated successfully!")
                self.log_message(f"  Formula: {metadata['formula']}")
                self.log_message(f"  Total atoms: {metadata['total_atoms']}")
                self.log_message(f"  Strain: {metadata['strain']*100:.2f}%")
                self.log_message(f"  Interface area: {metadata['interface_area']:.2f} Å²")
                self.log_message(f"  Cell dimensions: {result['structure'].cell.lengths()}")
                self.log_message("")
                
                # Enable buttons
                self.visualize_btn.config(state="normal")
                self.save_btn.config(state="normal")
                self.save_comp_btn.config(state="normal")
                
            except Exception as e:
                self.log_message(f"❌ Error: {str(e)}")
                messagebox.showerror("Error", f"Failed to generate interface:\n{str(e)}")
            finally:
                self.progress.stop()
                self.generate_btn.config(state="normal")
        
        # Run in thread to prevent GUI freezing
        thread = threading.Thread(target=worker)
        thread.daemon = True
        thread.start()
        
    def visualize_interface_gui(self):
        # Visualize the current interface
        if self.current_interface is None:
            messagebox.showwarning("Warning", "No interface generated yet. Please generate an interface first.")
            return
            
        try:
            # Get current parameters for title
            params = self.validate_parameters()
            sandwich_mode_text = "Sandwich" if params['sandwich_mode'] else "Normal"
            title = f"{sandwich_mode_text} Zr({params['zr_surface']})/ZrO₂({params['zro2_surface']}) Interface"
            
            self.log_message("Opening visualization...")
            
            # Get structure and check for issues
            structure = self.current_interface['structure']
            positions = structure.get_positions()
            
            # Check for NaN values and warn user
            if hasattr(positions, 'any') and np.isnan(positions).any():
                self.log_message("⚠️  Structure contains invalid values - attempting to clean and visualize...")
                
            visualize_interface(structure, title)
            self.log_message("✓ Visualization opened successfully")
            
        except Exception as e:
            error_msg = str(e)
            self.log_message(f"❌ Visualization error: {error_msg}")
            
            # Provide helpful error message based on error type
            if "NaN" in error_msg or "invalid" in error_msg:
                detailed_msg = ("The interface structure contains invalid values (NaN).\n\n"
                               "This usually happens when the lattice matching fails.\n"
                               "Try using different surface orientations or layer numbers.")
            elif "view" in error_msg or "GUI" in error_msg:
                detailed_msg = ("ASE GUI visualization failed.\n\n"
                               "This may be due to missing GUI dependencies.\n"
                               "The tool will try to use matplotlib instead.")
            else:
                detailed_msg = f"Visualization failed: {error_msg}\n\nCheck the console output for more details."
                
            messagebox.showwarning("Visualization Issue", detailed_msg)
            
    def save_interface(self):
        # Save the current interface to file
        if self.current_interface is None:
            messagebox.showwarning("Warning", "No interface generated yet. Please generate an interface first.")
            return
            
        try:
            # Ask for filename and format
            file_path = filedialog.asksaveasfilename(
                defaultextension=".cif",
                filetypes=[
                    ("CIF files", "*.cif"),
                    ("POSCAR files", "*.poscar"),
                    ("Extended XYZ", "*.extxyz"),
                    ("XYZ files", "*.xyz"),
                    ("All files", "*.*")
                ],
                title="Save Interface As..."
            )
            
            if file_path:
                self.current_interface['structure'].write(file_path)
                self.log_message(f"✓ Interface saved as: {os.path.basename(file_path)}")
                
        except Exception as e:
            self.log_message(f"❌ Save error: {str(e)}")
            messagebox.showerror("Error", f"Failed to save interface:\n{str(e)}")

    def save_components(self):
        # Save Zr and ZrO₂ component slabs to files
        if self.current_interface is None:
            messagebox.showwarning("Warning", "No interface generated yet. Please generate an interface first.")
            return
        try:
            comps = self.current_interface.get('components') or {}
            zr = comps.get('zr')
            ox = comps.get('zro2')
            if zr is None or ox is None:
                messagebox.showwarning("Warning", "Component slabs are not available for this build.")
                return

            # Choose directory
            outdir = filedialog.askdirectory(title="Select folder to save Zr/ZrO₂ parts")
            if not outdir:
                return

            # Base name hint
            base = f"zr{self.zr_surface_var.get()}_zro2{self.zro2_surface_var.get()}"

            # Save in multiple formats
            zr.write(os.path.join(outdir, f"{base}_Zr.cif"))
            zr.write(os.path.join(outdir, f"{base}_Zr.xyz"))
            zr.write(os.path.join(outdir, f"{base}_Zr.extxyz"))
            zr.write(os.path.join(outdir, f"{base}_Zr.POSCAR"))

            ox.write(os.path.join(outdir, f"{base}_ZrO2.cif"))
            ox.write(os.path.join(outdir, f"{base}_ZrO2.xyz"))
            ox.write(os.path.join(outdir, f"{base}_ZrO2.extxyz"))
            ox.write(os.path.join(outdir, f"{base}_ZrO2.POSCAR"))

            self.log_message(f"✓ Saved components to: {outdir}")
        except Exception as e:
            self.log_message(f"❌ Save components error: {str(e)}")
            messagebox.showerror("Error", f"Failed to save components:\n{str(e)}")

def main():
    #Launch the GUI application
    root = tk.Tk()
    app = InterfaceGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main() 