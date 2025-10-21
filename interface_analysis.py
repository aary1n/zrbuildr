#!/usr/bin/env python3

# disclaimer: this file includes segments written with gen-AI assistance (ChatGPT/GPT-5).
# reviewed + adapted manually.

# Interface Energy Analysis

# Simple tool for analyzing interface energies and finding optimal configurations.
# Focuses on the core scientific question: which interfaces have lowest energy?


import json
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

class InterfaceEnergyAnalyzer:
    """
    Analyze interface energies from DFT calculations.
    
    Purpose: Find optimal interface configurations for experimental comparison
    
    1. Read DFT energies from VASP calculations
    2. Calculate interface formation energies
    3. Identify energy minima
    4. Compare with experimental data
    5. Analyze trends (strain, layers, gap effects)
    """
    
    def __init__(self, interfaces_dir):
        self.interfaces_dir = Path(interfaces_dir)
        self.energies = {}
        self.reference_energies = {}
        
    def load_vasp_energies(self):
        # Load final energies from VASP OUTCAR files.
        for interface_dir in self.interfaces_dir.iterdir():
            if not interface_dir.is_dir():
                continue
                
            outcar = interface_dir / "OUTCAR"
            if not outcar.exists():
                continue
                
            try:
                energy = self._extract_vasp_energy(outcar)
                
                # Load metadata
                metadata_file = interface_dir / "metadata.json"
                if metadata_file.exists():
                    with open(metadata_file) as f:
                        metadata = json.load(f)
                    
                    self.energies[interface_dir.name] = {
                        'total_energy': energy,
                        'parameters': metadata['parameters'],
                        'metadata': metadata['metadata']
                    }
                    
            except Exception as e:
                print(f"Warning: Could not read energy for {interface_dir.name}: {e}")
        
        print(f"Loaded energies for {len(self.energies)} interfaces")
    
    def _extract_vasp_energy(self, outcar_path):
        # Extract final energy from VASP OUTCAR.
        with open(outcar_path) as f:
            for line in f:
                if "free  energy   TOTEN" in line:
                    energy = float(line.split()[-2])
        return energy
    
    def calculate_interface_energies(
        self,
        mu_zr_bulk_eV_per_atom=-8.52,
        mu_zro2_bulk_eV_per_fu=-22.42,
        mu_o2_eV_per_molecule=-9.86,
        reference_scheme="bulk_split",
    ):
        """
        save me vibecoding!!!

        Calculate interface and formation energies with clear references.

        Parameters:
        - mu_zr_bulk_eV_per_atom: Zr metal bulk energy per atom (same settings as interfaces)
        - mu_zro2_bulk_eV_per_fu: ZrO2 bulk energy per formula unit (ZrO2)
        - mu_o2_eV_per_molecule: O2 molecule energy (same code/pseudopotentials if used)
        - reference_scheme: 'bulk_split' (default, for interface energy) or 'elemental'

        Returns per interface:
        - gamma_eV_per_A2 and gamma_J_per_m2 (only meaningful for 'bulk_split')
        - deltaH_eV_total and deltaH_eV_per_fu (elemental, Zr + 1/2 O2 -> ZrO2)
        """
        EV_PER_A2_TO_J_PER_M2 = 16.021766

        interface_energies = {}

        for name, data in self.energies.items():
            composition = data['metadata']['composition']
            total_energy = data['total_energy']

            n_zr = int(composition.get('Zr', 0))
            n_o = int(composition.get('O', 0))

            # Estimate number of ZrO2 units in the oxide region by oxygen count
            n_fu_zro2 = n_o // 2
            n_zr_in_oxide = n_fu_zro2
            n_zr_in_metal = max(n_zr - n_zr_in_oxide, 0)

            area_A2 = float(data['metadata']['interface_area'])  # Å^2
            num_interfaces = int(data['metadata'].get('num_interfaces', 2))

            # Interface energy using split-bulk references
            gamma_eV_total = None
            gamma_eV_per_A2 = None
            gamma_J_per_m2 = None
            if reference_scheme == "bulk_split":
                ref = n_zr_in_metal * mu_zr_bulk_eV_per_atom + n_fu_zro2 * mu_zro2_bulk_eV_per_fu
                excess = total_energy - ref
                # Two interfaces in a symmetric slab
                gamma_eV_per_A2 = excess / (num_interfaces * area_A2)
                gamma_J_per_m2 = gamma_eV_per_A2 * EV_PER_A2_TO_J_PER_M2

            # Global formation energy relative to elemental references
            deltaH_eV_total = total_energy - n_zr * mu_zr_bulk_eV_per_atom - (n_o / 2.0) * mu_o2_eV_per_molecule
            deltaH_eV_per_fu = deltaH_eV_total / n_fu_zro2 if n_fu_zro2 > 0 else None

            interface_energies[name] = {
                'strain': data['metadata']['strain'],
                'area_A2': area_A2,
                'parameters': data['parameters'],
                'gamma_eV_per_A2': gamma_eV_per_A2,
                'gamma_J_per_m2': gamma_J_per_m2,
                'deltaH_eV_total': deltaH_eV_total,
                'deltaH_eV_per_fu': deltaH_eV_per_fu,
                'reference_scheme': reference_scheme,
            }

        self.interface_energies = interface_energies
        return interface_energies
    
    def find_optimal_interfaces(self, max_strain=0.05):
        # Find interfaces with lowest formation energy.
        # Filter by strain
        valid_interfaces = {name: data for name, data in self.interface_energies.items() 
                          if data['strain'] <= max_strain}
        
        # Sort by appropriate metric
        def score(item):
            data = item[1]
            # Prefer gamma if available; fallback to deltaH per fu
            if data.get('gamma_eV_per_A2') is not None:
                return data['gamma_eV_per_A2']
            return data.get('deltaH_eV_per_fu', np.inf)

        sorted_interfaces = sorted(valid_interfaces.items(), key=score)
        
        print(f"\n=== OPTIMAL INTERFACES (strain ≤ {max_strain*100:.1f}%) ===")
        for i, (name, data) in enumerate(sorted_interfaces[:10]):
            params = data['parameters']
            print(f"{i+1:2d}. {name}")
            if data.get('gamma_eV_per_A2') is not None:
                print(f"    Interface energy: {data['gamma_eV_per_A2']:.4f} eV/Å²  ({data['gamma_J_per_m2']:.2f} J/m²)")
            if data.get('deltaH_eV_per_fu') is not None:
                print(f"    Formation (elemental): {data['deltaH_eV_per_fu']:.3f} eV per ZrO₂ fu")
            print(f"    Strain: {data['strain']*100:.2f}%")
            print(f"    Configuration: Zr({params['zr_surface']})/ZrO₂({params['zro2_surface']})")
            print(f"    Layers: Zr={params['zr_layers']}, ZrO₂={params['zro2_layers']}")
            print(f"    Gap: {params['gap']} Å")
            print()
        
        return sorted_interfaces
    
    def analyze_trends(self):
        # Analyze how interface energy depends on parameters.
        # Group by different parameters
        by_surfaces = {}
        by_layers = {}
        by_gaps = {}
        
        for name, data in self.interface_energies.items():
            params = data['parameters']
            
            # By surface combination
            surface_key = f"Zr({params['zr_surface']})/ZrO₂({params['zro2_surface']})"
            if surface_key not in by_surfaces:
                by_surfaces[surface_key] = []
            metric = data.get('gamma_eV_per_A2') if data.get('gamma_eV_per_A2') is not None else data.get('deltaH_eV_per_fu')
            by_surfaces.setdefault(surface_key, []).append(metric)
            
            # By layer thickness
            layer_key = f"Zr{params['zr_layers']}_ZrO2{params['zro2_layers']}"
            if layer_key not in by_layers:
                by_layers[layer_key] = []
            by_layers.setdefault(layer_key, []).append(metric)
            
            # By gap
            gap_key = params['gap']
            if gap_key not in by_gaps:
                by_gaps[gap_key] = []
            by_gaps.setdefault(gap_key, []).append(metric)
        
        # Print trends
        print("\n=== ENERGY TRENDS ===")
        
        print("\nBy Surface Orientation (lower is better):")
        for surface, energies in by_surfaces.items():
            mean_energy = np.mean(energies)
            std_energy = np.std(energies)
            print(f"  {surface:<28}: {mean_energy:8.4f} ± {std_energy:.4f}")
        
        print("\nBy Layer Thickness:")
        for layers, energies in sorted(by_layers.items()):
            mean_energy = np.mean(energies)
            print(f"  {layers:<15}: {mean_energy:6.3f} eV")
        
        print("\nBy Interface Gap:")
        for gap, energies in sorted(by_gaps.items()):
            mean_energy = np.mean(energies)
            print(f"  {gap:4.1f} Å: {mean_energy:6.3f} eV")
    
    def plot_energy_landscape(self, output_file='energy_landscape.png'):
        # Create visualization of energy landscape.
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        
        # Extract data for plotting
        strains = [data['strain'] * 100 for data in self.interface_energies.values()]
        # Prefer interface energy per area; fallback to formation per fu
        energies = [
            (data['gamma_eV_per_A2'] if data.get('gamma_eV_per_A2') is not None else data.get('deltaH_eV_per_fu'))
            for data in self.interface_energies.values()
        ]
        areas = [data['area_A2'] for data in self.interface_energies.values()]
        
        # Energy vs strain
        ax1.scatter(strains, energies, alpha=0.7)
        ax1.set_xlabel('Strain (%)')
        ax1.set_ylabel('Energy metric (eV/Å² or eV/fu)')
        ax1.set_title('Energy vs Strain')
        ax1.grid(True, alpha=0.3)
        
        # Energy vs area
        ax2.scatter(areas, energies, alpha=0.7)
        ax2.set_xlabel('Interface Area (Å²)')
        ax2.set_ylabel('Energy metric (eV/Å² or eV/fu)')
        ax2.set_title('Energy vs Area')
        ax2.grid(True, alpha=0.3)
        
        # Energy distribution
        ax3.hist(energies, bins=20, alpha=0.7, edgecolor='black')
        ax3.set_xlabel('Energy metric (eV/Å² or eV/fu)')
        ax3.set_ylabel('Count')
        ax3.set_title('Energy Metric Distribution')
        ax3.grid(True, alpha=0.3)
        
        # Strain distribution
        ax4.hist(strains, bins=20, alpha=0.7, edgecolor='black')
        ax4.set_xlabel('Strain (%)')
        ax4.set_ylabel('Count')
        ax4.set_title('Strain Distribution')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Energy landscape plot saved to {output_file}")
    
    def export_results(self, output_file='interface_analysis_results.json'):
        # Export analysis results for further processing.
        results = {
            'interface_energies': self.interface_energies,
            'summary': {
                'total_interfaces': len(self.interface_energies),
                'interface_energy_range_eV_per_A2': {
                    'min': min(
                        data['gamma_eV_per_A2'] for data in self.interface_energies.values()
                        if data.get('gamma_eV_per_A2') is not None
                    ) if any(data.get('gamma_eV_per_A2') is not None for data in self.interface_energies.values()) else None,
                    'max': max(
                        data['gamma_eV_per_A2'] for data in self.interface_energies.values()
                        if data.get('gamma_eV_per_A2') is not None
                    ) if any(data.get('gamma_eV_per_A2') is not None for data in self.interface_energies.values()) else None,
                    'mean': (
                        np.mean([data['gamma_eV_per_A2'] for data in self.interface_energies.values() if data.get('gamma_eV_per_A2') is not None])
                        if any(data.get('gamma_eV_per_A2') is not None for data in self.interface_energies.values()) else None
                    ),
                },
                'formation_energy_range_eV_per_fu': {
                    'min': min(
                        data['deltaH_eV_per_fu'] for data in self.interface_energies.values()
                        if data.get('deltaH_eV_per_fu') is not None
                    ) if any(data.get('deltaH_eV_per_fu') is not None for data in self.interface_energies.values()) else None,
                    'max': max(
                        data['deltaH_eV_per_fu'] for data in self.interface_energies.values()
                        if data.get('deltaH_eV_per_fu') is not None
                    ) if any(data.get('deltaH_eV_per_fu') is not None for data in self.interface_energies.values()) else None,
                    'mean': (
                        np.mean([data['deltaH_eV_per_fu'] for data in self.interface_energies.values() if data.get('deltaH_eV_per_fu') is not None])
                        if any(data.get('deltaH_eV_per_fu') is not None for data in self.interface_energies.values()) else None
                    ),
                },
                'strain_range': {
                    'min': min(data['strain'] for data in self.interface_energies.values()),
                    'max': max(data['strain'] for data in self.interface_energies.values()),
                    'mean': np.mean([data['strain'] for data in self.interface_energies.values()])
                }
            }
        }
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"Analysis results exported to {output_file}")


# Simple command-line interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze interface energies from DFT calculations")
    parser.add_argument("interfaces_dir", help="Directory containing interface calculations")
    parser.add_argument("--plot", action="store_true", help="Create energy landscape plots")
    parser.add_argument("--max-strain", type=float, default=0.05, help="Maximum strain for optimization")
    
    args = parser.parse_args()
    
    # Run analysis
    analyzer = InterfaceEnergyAnalyzer(args.interfaces_dir)
    
    print("Loading DFT energies from VASP calculations...")
    analyzer.load_vasp_energies()
    
    print("Calculating interface formation energies...")
    analyzer.calculate_interface_energies()
    
    print("Finding optimal interface configurations...")
    optimal = analyzer.find_optimal_interfaces(args.max_strain)
    
    print("Analyzing energy trends...")
    analyzer.analyze_trends()
    
    if args.plot:
        analyzer.plot_energy_landscape()
    
    analyzer.export_results()
    
    print(f"\n=== SUMMARY ===")
    print(f"Analyzed {len(analyzer.interface_energies)} interface configurations")
    print(f"Best interface: {optimal[0][0]}")
    best = optimal[0][1]
    if best.get('gamma_eV_per_A2') is not None:
        print(f"Interface energy: {best['gamma_eV_per_A2']:.4f} eV/Å²  ({best['gamma_J_per_m2']:.2f} J/m²)")
    if best.get('deltaH_eV_per_fu') is not None:
        print(f"Formation (elemental): {best['deltaH_eV_per_fu']:.3f} eV per ZrO₂ fu")
    print(f"Ready for experimental comparison!") 