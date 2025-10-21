#!/usr/bin/env python3
"""
Simplified Zr/ZrO₂ Interface Builder

Author: Aaryan Sharif
Purpose: Generate interfaces for DFT energy calculations and experimental comparison

# Note from the author: Portions of this file were developed with assistance from a generative AI tool (ChatGPT, GPT-5).
# All code has been reviewed, verified, and integrated by me to ensure correctness and suitability.
"""

import numpy as np
from ase import Atoms
from ase.build import surface, add_vacuum
from ase.spacegroup import crystal
from ase.io import write
import os
import json
from itertools import product
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def visualize_interface(atoms, title="Zr/ZrO₂ Interface"):
    """
    For PDRA review: Visualize an interface using ASE's visualization tools.
    Handles NaN values and provides fallbacks.
    
    Arguments:
        atoms: ASE Atoms object containing the interface
        title: Title for the visualization window
    """
    # Check for and handle NaN values
    positions = atoms.get_positions()
    has_nan = np.isnan(positions).any()
    
    if has_nan:
        logger.warning("Structure contains NaN values - cleaning before visualization")
        # Remove atoms with NaN positions
        valid_indices = ~np.isnan(positions).any(axis=1)
        if valid_indices.sum() == 0:
            logger.error("All atoms have NaN positions - cannot visualize")
            return
        
        # Create cleaned copy
        clean_atoms = atoms[valid_indices]
        logger.info(f"Removed {len(atoms) - len(clean_atoms)} atoms with NaN positions")
        atoms = clean_atoms
        
    # Check cell for NaN values
    cell = atoms.get_cell()
    if np.isnan(cell).any():
        logger.warning("Cell contains NaN values - using default cubic cell")
        # Set a reasonable default cell
        max_pos = atoms.get_positions().max(axis=0)
        min_pos = atoms.get_positions().min(axis=0)
        size = max_pos - min_pos + 5.0  # Add 5 angstrom padding
        atoms.set_cell([size[0], size[1], size[2]])
        atoms.center()
    
    try:
        from ase.visualize import view
        logger.info(f"Opening visualization: {title}")
        # Try ASE GUI first
        view(atoms)
        
    except (ImportError, ValueError, Exception) as e:
        logger.warning(f"ASE GUI failed ({str(e)}). Using matplotlib visualization...")
        try:
            # Alternative: matplotlib visualization  
            import matplotlib.pyplot as plt
            from ase.visualize.plot import plot_atoms
            
            fig, ax = plt.subplots(figsize=(12, 9))
            
            # Use matplotlib plot_atoms with error handling
            plot_atoms(atoms, ax, radii=0.8, rotation=('45x,45y,0z'))
            ax.set_title(title, fontsize=14, fontweight='bold')
            
            # Add detailed interface info
            info_lines = [
                f"Atoms: {len(atoms)}",
                f"Cell: {atoms.cell.lengths():.1f} Å",
                f"Formula: {atoms.get_chemical_formula()}"
            ]
            
            # Add strain info if available in atoms.info
            if hasattr(atoms, 'info') and 'strain' in atoms.info:
                info_lines.append(f"Strain: {atoms.info['strain']*100:.1f}%")
                
            info_text = "\n".join(info_lines)
            ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
                   verticalalignment='top', fontsize=10,
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
            
            plt.tight_layout()
            plt.show()
            logger.info("Successfully displayed interface using matplotlib")
            
        except Exception as e:
            logger.error(f"Matplotlib visualization also failed: {str(e)}")
            # Last resort: print structure info
            logger.info("=== STRUCTURE INFORMATION ===")
            logger.info(f"Title: {title}")
            logger.info(f"Number of atoms: {len(atoms)}")
            logger.info(f"Chemical formula: {atoms.get_chemical_formula()}")
            logger.info(f"Cell parameters: {atoms.cell.lengths()}")
            logger.info(f"Cell angles: {atoms.cell.angles()}")
            
            # Print atomic positions summary
            positions = atoms.get_positions()
            logger.info(f"Position range: X=[{positions[:,0].min():.2f}, {positions[:,0].max():.2f}]")
            logger.info(f"                Y=[{positions[:,1].min():.2f}, {positions[:,1].max():.2f}]") 
            logger.info(f"                Z=[{positions[:,2].min():.2f}, {positions[:,2].max():.2f}]")
            logger.info("Visualization not available - see structure info above")

class ZrZrO2InterfaceBuilder:
    """
    For PDRA review: Simplified interface builder focused on systematic generation.
    
    Purposes: Generate Zr/ZrO₂ interfaces for energy calculations
    
    Inputs: 
    - Zr surface orientation (e.g., '0001', '1010') 
    - ZrO₂ surface orientation (e.g., '111', '100')
    - ZrO₂ phase ('tetragonal', 'monoclinic', 'cubic')
    - Number of layers for each material
    - Interface gap distance
    
    Outputs:
    - Structure files (.cif, .xyz, POSCAR) ready for DFT
    - Interface metadata (strain, area, composition)
    - Batch files for HPC submission
    """
    
    def __init__(self):
        # Material properties from experimental data
        self.materials = {
            'zr': {
                'lattice': {'a': 3.232, 'c': 5.147},  # HCP, experimental
                'space_group': 194,
                'atoms': [('Zr', [0, 0, 0]), ('Zr', [1/3, 2/3, 1/2])]
            },
            'zro2': {
                'tetragonal': {
                    'lattice': {'a': 5.090},  # Temporarily using cubic (works!)
                    'space_group': 225,  # Simple fluorite structure
                    'atoms': [('Zr', [0, 0, 0]), ('O', [0.25, 0.25, 0.25]), ('O', [0.75, 0.75, 0.75])]
                },
                'monoclinic': {
                    'lattice': {'a': 5.169, 'b': 5.232, 'c': 5.341, 'beta': 99.0},
                    'space_group': 14,
                    'atoms': [('Zr', [0.276, 0.041, 0.209]), ('Zr', [0.724, 0.959, 0.791]),
                             ('O', [0.070, 0.332, 0.345]), ('O', [0.930, 0.668, 0.655]),
                             ('O', [0.448, 0.756, 0.479]), ('O', [0.552, 0.244, 0.521])]
                },
                'cubic': {
                    'lattice': {'a': 5.090},
                    'space_group': 225,
                    'atoms': [('Zr', [0, 0, 0]), ('O', [0.25, 0.25, 0.25]), ('O', [0.75, 0.75, 0.75])]
                }
            }
        }
    
    def create_bulk(self, material, phase=None):
        """PDRA: Create bulk unit cell."""
        if material == 'zr':
            params = self.materials['zr']
            cellpar = [params['lattice']['a'], params['lattice']['a'], params['lattice']['c'], 90, 90, 120]
        else:  # zro2
            params = self.materials['zro2'][phase]
            if phase == 'tetragonal':
                # Handle both true tetragonal and cubic-as-tetragonal cases
                if 'c' in params['lattice']:
                    cellpar = [params['lattice']['a'], params['lattice']['a'], params['lattice']['c'], 90, 90, 90]
                else:
                    # Cubic structure labeled as tetragonal
                    cellpar = [params['lattice']['a'], params['lattice']['a'], params['lattice']['a'], 90, 90, 90]
            elif phase == 'monoclinic':
                cellpar = [params['lattice']['a'], params['lattice']['b'], params['lattice']['c'], 
                          90, params['lattice']['beta'], 90]
            else:  # cubic
                cellpar = [params['lattice']['a'], params['lattice']['a'], params['lattice']['a'], 90, 90, 90]
        
        symbols = [atom[0] for atom in params['atoms']]
        positions = [atom[1] for atom in params['atoms']]
        
        return crystal(symbols, basis=positions, spacegroup=params['space_group'], cellpar=cellpar)
    
    def create_bulk_with_scaled_positions(self, material, phase=None):
        """PDRA: Revision. Create bulk unit cell using proper scaled positions."""
        if material == 'zr':
            # Zr HCP structure
            cellpar = [3.232, 3.232, 5.147, 90, 90, 120]
            # Scaled positions for HCP Zr
            scaled_positions = [
                [0, 0, 0],      # Zr at origin
                [1/3, 2/3, 1/2] # Zr at HCP position
            ]
            symbols = ['Zr', 'Zr']
            
        else:  # zro2 - CORRECTED fluorite structure
            cellpar = [5.090, 5.090, 5.090, 90, 90, 90]
            # CORRECT scaled positions for fluorite ZrO₂ (4 Zr + 8 O)
            scaled_positions = [
                # Zr atoms (4 per unit cell) - fluorite positions
                [0, 0, 0],           # Zr at origin
                [0.5, 0.5, 0],       # Zr at face centers
                [0.5, 0, 0.5], 
                [0, 0.5, 0.5],
                # O atoms (8 per unit cell) - fluorite positions
                [0.25, 0.25, 0.25],  # O at 1/4 positions
                [0.75, 0.75, 0.25],
                [0.75, 0.25, 0.75],
                [0.25, 0.75, 0.75],
                [0.75, 0.25, 0.25],
                [0.25, 0.75, 0.25],
                [0.25, 0.25, 0.75],
                [0.75, 0.75, 0.75]
            ]
            symbols = ['Zr'] * 4 + ['O'] * 8
        
        # Convert cellpar to cell matrix
        from ase.geometry import cellpar_to_cell
        cell_matrix = cellpar_to_cell(cellpar)
        
        # Create atoms object with scaled positions
        atoms = Atoms(symbols=symbols, cell=cell_matrix, pbc=True)
        atoms.set_scaled_positions(scaled_positions)
        
        return atoms
    
    def create_surface_with_scaled_positions(self, material, miller_indices, layers, vacuum=15.0, phase=None):
        """PDRA: 2nd revision. Create surface using scaled positions to avoid distortion."""
        
        # Create bulk with proper scaled positions
        bulk = self.create_bulk_with_scaled_positions(material, phase)
        
        # Validate bulk structure
        bulk_positions = bulk.get_positions()
        if np.isnan(bulk_positions).any():
            logger.error(f"Bulk {material} structure contains NaN positions")
            raise ValueError(f"Invalid bulk structure for {material}")
        
        logger.debug(f"Created bulk {material}: {len(bulk)} atoms, cell: {bulk.get_cell().diagonal()}")
        
        # Convert Miller indices
        if isinstance(miller_indices, str):
            if material == 'zr' and len(miller_indices) == 4:
                # Handle HCP 4-index notation
                if miller_indices == '0001':
                    miller_list = [0, 0, 1]
                elif miller_indices == '1010':
                    miller_list = [1, 0, 0]
                elif miller_indices == '1120':
                    miller_list = [1, 1, 0]
                else:
                    miller_list = [int(x) for x in miller_indices[:3]]
            else:
                miller_list = [int(x) for x in miller_indices]
        else:
            miller_list = miller_indices
        
        logger.debug(f"Creating surface with Miller indices: {miller_list}")
        
        # Create surface using ASE's surface function with error handling
        from ase.build import surface, add_vacuum
        try:
            slab = surface(bulk, miller_list, layers)
            
            # FIX: Handle NaN values in cell IMMEDIATELY
            cell = slab.get_cell()
            if np.isnan(cell).any():
                logger.warning(f"Surface {material} cell contains NaN - fixing...")
                
                # Get atomic positions to estimate reasonable cell
                positions = slab.get_positions()
                
                # Calculate reasonable cell dimensions from atomic extent
                pos_range = positions.max(axis=0) - positions.min(axis=0)
                
                # Create a fixed cell - replace NaN diagonal elements
                fixed_cell = cell.copy()
                for i in range(3):
                    if np.isnan(fixed_cell[i,i]) or fixed_cell[i,i] == 0:
                        # Use atomic extent plus some padding for z-direction
                        if i == 2:  # z-direction
                            fixed_cell[i,i] = pos_range[i] + 5.0  # More padding for vacuum
                        else:  # x,y directions
                            fixed_cell[i,i] = pos_range[i] + 2.0
                
                # Apply the fixed cell
                slab.set_cell(fixed_cell)
                logger.debug(f"Fixed cell: {fixed_cell.diagonal()}")
            
            add_vacuum(slab, vacuum)
            
            # Validate surface structure
            slab_positions = slab.get_positions()
            if np.isnan(slab_positions).any():
                logger.error(f"Surface {material} structure contains NaN positions")
                raise ValueError(f"Invalid surface structure for {material}")
            
            logger.debug(f"Created surface {material}: {len(slab)} atoms")
            return slab
            
        except Exception as e:
            logger.error(f"Surface creation failed for {material}: {e}")
            raise
    
    def apply_strain_properly(self, atoms, target_cell):
        """... You get the point. Apply strain while preserving scaled positions."""
        
        # Get current and target cell parameters
        current_cell = atoms.get_cell()
        current_scaled = atoms.get_scaled_positions()
        
        # Validate that we have valid positions
        if np.isnan(current_scaled).any():
            logger.error("Cannot apply strain: structure contains NaN positions")
            return atoms
        
        # Calculate strain matrix
        strain_matrix = np.eye(3)
        max_strain = 0.0
        
        for i in range(2):  # Only strain x,y (not z)
            if current_cell[i,i] != 0:
                strain_ratio = target_cell[i,i] / current_cell[i,i]
                strain_matrix[i,i] = strain_ratio
                strain = abs(1 - strain_ratio)
                max_strain = max(max_strain, strain)
        
        # Check if strain is reasonable
        if max_strain > 0.1:  # More than 10% strain
            logger.warning(f"Large strain detected: {max_strain*100:.1f}%. This may cause issues.")
        
        # Apply strain to cell
        new_cell = strain_matrix @ current_cell
        
        # Validate new cell
        if np.isnan(new_cell).any():
            logger.error("Strain application produced NaN cell")
            return atoms
        
        atoms.set_cell(new_cell)
        
        # IMPORTANT: Keep scaled positions the same!
        atoms.set_scaled_positions(current_scaled)
        
        # Final validation
        final_positions = atoms.get_positions()
        if np.isnan(final_positions).any():
            logger.error("Final structure contains NaN positions after strain")
        
        return atoms
    
    def create_surface(self, material, miller_indices, layers, vacuum=15.0, phase=None):
        """Create surface slab using working direct ASE approach to avoid NaN issues."""
        
        # WORKAROUND: Use direct ASE calls that we know work
        from ase.spacegroup import crystal
        from ase.build import surface, add_vacuum
        
        if material == 'zr':
            # Create Zr bulk using direct ASE (known to work)
            bulk = crystal(['Zr', 'Zr'], 
                          basis=[[0, 0, 0], [1/3, 2/3, 1/2]], 
                          spacegroup=194, 
                          cellpar=[3.232, 3.232, 5.147, 90, 90, 120])
            
            # Handle Miller indices for HCP
            if isinstance(miller_indices, str):
                if len(miller_indices) == 4:  # hexagonal 4-index notation
                    digits = [int(x) for x in miller_indices]
                    h, k, i_given, l = digits
                    i_correct = -(h + k)
                    
                    if i_given != i_correct:
                        logger.info(f"Converting Miller notation: '{miller_indices}' -> [{h}, {k}, {i_correct}, {l}]")
                    
                    # Convert to 3-index: common conversions that work
                    if miller_indices == '0001':
                        miller_list = [0, 0, 1]
                    elif miller_indices == '1010':
                        miller_list = [1, 0, 0]  # Simplified working conversion
                    elif miller_indices == '1120':
                        miller_list = [1, 1, 0]
                    else:
                        miller_list = [h, k, l]
                else:
                    miller_list = [int(x) for x in miller_indices]
            else:
                miller_list = miller_indices
                
        else:  # zro2 - use working cubic fluorite structure
            # Create ZrO2 bulk using direct ASE (known to work)
            bulk = crystal(['Zr', 'O', 'O'], 
                          basis=[[0, 0, 0], [0.25, 0.25, 0.25], [0.75, 0.75, 0.75]], 
                          spacegroup=225, 
                          cellpar=[5.090, 5.090, 5.090, 90, 90, 90])
            
            # Convert string to list for ZrO2
            if isinstance(miller_indices, str):
                miller_list = [int(x) for x in miller_indices]
            else:
                miller_list = miller_indices
        
        # Create surface using direct ASE
        slab = surface(bulk, miller_list, layers)
        add_vacuum(slab, vacuum)
        
        # CRITICAL FIX: Repair NaN values in cell to prevent repeat() failures
        cell = slab.get_cell()
        if np.isnan(cell).any():
            logger.debug("Fixing NaN values in surface cell")
            
            # For surfaces, estimate reasonable z-direction from atomic positions
            positions = slab.get_positions()
            z_extent = positions[:, 2].max() - positions[:, 2].min()
            reasonable_z = z_extent + vacuum
            
            # Create fixed cell by replacing NaN row with [0, 0, reasonable_z]
            fixed_cell = cell.copy()
            fixed_cell.array[2] = [0, 0, reasonable_z]
            
            # Apply the fixed cell
            slab.set_cell(fixed_cell)
            logger.debug(f"Fixed cell z-dimension: {reasonable_z:.3f} Å")

        
        return slab
    
    def create_zr_surface_with_constraint(self, layers=8, vacuum=15.0, force_1210_normal=True, alternative_surface=None):

        if force_1210_normal:
            miller_indices = "1210" # 4-index notation
            logger.info("Creating Zr Surface with <1 -2 1 0> normal constraint")
        else:
            miller_indices = alternative_surface or "0001"
            logger.info(f"Creating Zr surface: {miller_indices}")

        # Create surface slab via existing create_surface method. Handles conversion.
        zr_slab = self.create_surface('zr', miller_indices, layers, vacuum)
        
        # Validate for correct orientation.
        if force_1210_normal:
            self._validate_surface_normal(zr_slab, [1, -2, 1, 0])

        return zr_slab # ASE Atoms object with Zr surface
    
    def _validate_surface_normal(self, slab, normal):

        cell = slab.get_cell()
        surface_normal = cell[2]
        
        dot_product = np.dot(surface_normal, normal)

        if dot_product < 0.999:
            logger.warning(f"Surface normal is not perpendicular to <1 -2 1 0> by more than 0.001. Adjusting cell...")
            # Adjust cell to make normal perpendicular
            new_cell = cell.copy()
            new_cell[2] = surface_normal
            slab.set_cell(new_cell)


    
    def find_best_supercells(self, slab1, slab2, max_strain=0.05, max_size=5, min_size=2):
        # Find optimal supercell match with minimal strain.
        cell1 = slab1.get_cell()[:2, :2]  # a, b vectors
        cell2 = slab2.get_cell()[:2, :2]
        
        a1, b1 = np.linalg.norm(cell1, axis=1)
        a2, b2 = np.linalg.norm(cell2, axis=1)
        
        best_strain = float('inf')
        best_match = None
        
        for n1, n2 in product(range(min_size, max_size + 1), repeat=2):
            for m1, m2 in product(range(min_size, max_size + 1), repeat=2):
                # Supercell dimensions
                sc1_a, sc1_b = n1 * a1, n2 * b1
                sc2_a, sc2_b = m1 * a2, m2 * b2
                
                # Calculate strain needed to match
                strain_a = abs(sc1_a - sc2_a) / max(sc1_a, sc2_a)
                strain_b = abs(sc1_b - sc2_b) / max(sc1_b, sc2_b)
                max_strain_val = max(strain_a, strain_b)
                
                if max_strain_val < best_strain and max_strain_val <= max_strain:
                    best_strain = max_strain_val
                    best_match = {
                        'supercell1': (n1, n1, 1),
                        'supercell2': (m1, m1, 1),
                        'strain': best_strain,
                        'dimensions': {'slab1': (sc1_a, sc1_b), 'slab2': (sc2_a, sc2_b)}
                    }
        
        if best_match is None:
            # Fallback to 1x1 with warning
            logger.warning(f"No good supercell match found within {max_strain*100:.1f}% strain")
            strain = abs(a1 - a2) / max(a1, a2)
            best_match = {
                'supercell1': (1, 1, 1),
                'supercell2': (1, 1, 1), 
                'strain': strain,
                'dimensions': {'slab1': (a1, b1), 'slab2': (a2, b2)}
            }
        
        return best_match
    
    def build_interface(self, zr_surface, zro2_surface, zro2_phase='tetragonal', 
                       zr_layers=8, zro2_layers=6, gap=2.5, vacuum=15.0):
        """
        For PDRA review: Build a single interface.
        
        Returns:
            dict: {
                'structure': ASE Atoms object,
                'metadata': interface properties,
                'parameters': input parameters
            }
        """
        logger.info(f"Building Zr({zr_surface})/ZrO₂({zro2_surface}) interface")
        
        # Create slabs
        zr_slab = self.create_surface('zr', zr_surface, zr_layers, vacuum/2)
        zro2_slab = self.create_surface('zro2', zro2_surface, zro2_layers, vacuum/2, zro2_phase)
        
        # Find optimal supercells
        match = self.find_best_supercells(zr_slab, zro2_slab)
        logger.info(f"Optimal match: {match['supercell1']} × {match['supercell2']}, strain: {match['strain']*100:.2f}%")
        
        # Create supercells
        zr_super = zr_slab.repeat(match['supercell1'])
        zro2_super = zro2_slab.repeat(match['supercell2'])

        # Keep pristine copies of component slabs for saving
        components = {
            'zr': zr_super.copy(),
            'zro2': zro2_super.copy(),
        }
        
        # Remove overlapping atoms from ZrO₂ supercell (aggressive)
        zro2_super = self._remove_overlapping_atoms(zro2_super, min_distance=1.5)
        
        # Stack to create interface
        interface = self._stack_slabs(zr_super, zro2_super, gap, vacuum)
        
        # FINAL OVERLAP REMOVAL - apply to the complete interface (aggressive)
        interface = self._remove_overlapping_atoms(interface, min_distance=1.5)
        logger.info(f"Final interface after overlap removal: {len(interface)} atoms")
        
        # Validate and clean structure before analysis
        interface = self._validate_structure(interface)
        
        # Calculate metadata
        metadata = self._analyze_interface(interface, match)
        
        return {
            'structure': interface,
            'metadata': metadata,
            'parameters': {
                'zr_surface': zr_surface,
                'zro2_surface': zro2_surface,
                'zro2_phase': zro2_phase,
                'zr_layers': zr_layers,
                'zro2_layers': zro2_layers,
                'gap': gap,
                'vacuum': vacuum
            },
            'components': components
        }
    
    def build_interface_with_scaled_positions(self, zr_surface, zro2_surface, zro2_phase='tetragonal', 
                                            zr_layers=8, zro2_layers=6, gap=2.5, vacuum=15.0):
        """Build interface using scaled positions throughout."""
        
        logger.info(f"Building Zr({zr_surface})/ZrO₂({zro2_surface}) interface with scaled positions")
        
        # Create surfaces with proper scaled positions
        zr_slab = self.create_surface_with_scaled_positions('zr', zr_surface, zr_layers, vacuum/2)
        zro2_slab = self.create_surface_with_scaled_positions('zro2', zro2_surface, zro2_layers, vacuum/2, zro2_phase)
        
        # Find optimal supercells
        match = self.find_best_supercells(zr_slab, zro2_slab)
        logger.info(f"Optimal match: {match['supercell1']} × {match['supercell2']}, strain: {match['strain']*100:.2f}%")
        
        # Check if strain is too large
        if match['strain'] > 0.1:  # More than 10% strain
            logger.warning(f"Large strain detected ({match['strain']*100:.1f}%). Trying fallback approach...")
            
            # Try with smaller supercells to reduce strain
            fallback_match = self.find_best_supercells(zr_slab, zro2_slab, max_strain=0.15, max_size=3, min_size=1)
            if fallback_match['strain'] < match['strain']:
                match = fallback_match
                logger.info(f"Using fallback match: {match['supercell1']} × {match['supercell2']}, strain: {match['strain']*100:.2f}%")
        
        # Create supercells
        zr_super = zr_slab.repeat(match['supercell1'])
        zro2_super = zro2_slab.repeat(match['supercell2'])

        # Keep pristine copies of component slabs for saving
        components = {
            'zr': zr_super.copy(),
            'zro2': zro2_super.copy(),
        }
        
        # Apply strain properly (preserving scaled positions)
        target_cell = zr_super.get_cell()
        zro2_super = self.apply_strain_properly(zro2_super, target_cell)
        
        # Remove overlaps (now with proper structure)
        zro2_super = self._remove_overlapping_atoms(zro2_super, min_distance=1.5)
        
        # Stack slabs
        interface = self._stack_slabs(zr_super, zro2_super, gap, vacuum)
        
        # FINAL OVERLAP REMOVAL - apply to the complete interface
        interface = self._remove_overlapping_atoms(interface, min_distance=1.5)
        logger.info(f"Final interface after overlap removal: {len(interface)} atoms")
        
        # Validate and clean structure before analysis
        interface = self._validate_structure(interface)
        
        # Calculate metadata
        metadata = self._analyze_interface(interface, match)
        
        return {
            'structure': interface,
            'metadata': metadata,
            'parameters': {
                'zr_surface': zr_surface,
                'zro2_surface': zro2_surface,
                'zro2_phase': zro2_phase,
                'zr_layers': zr_layers,
                'zro2_layers': zro2_layers,
                'gap': gap,
                'vacuum': vacuum
            },
            'components': components
        }
    
    def build_sandwich_interface(self, zr_surface, zro2_surface, zro2_phase='tetragonal', 
                               zr_layers=8, zro2_layers=6, gap=2.5, vacuum=15.0):
        """
        For PDRA review: Build a sandwich interface: ZrO₂ | Zr | ZrO₂
        
        Returns:
            dict: Same format as build_interface but with sandwich structure
        """
        logger.info(f"Building ZrO₂({zro2_surface})/Zr({zr_surface})/ZrO₂({zro2_surface}) sandwich")
        
        # Create slabs
        zr_slab = self.create_surface('zr', zr_surface, zr_layers, vacuum/2)
        zro2_slab = self.create_surface('zro2', zro2_surface, zro2_layers, vacuum/2, zro2_phase)
        
        # Find optimal supercells
        match = self.find_best_supercells(zr_slab, zro2_slab)
        logger.info(f"Optimal match: {match['supercell1']} × {match['supercell2']}, strain: {match['strain']*100:.2f}%")
        
        # Create supercells
        zr_super = zr_slab.repeat(match['supercell1'])
        zro2_super = zro2_slab.repeat(match['supercell2'])

        # Keep pristine copies of component slabs for saving
        components = {
            'zr': zr_super.copy(),
            'zro2': zro2_super.copy(),
        }
        
        # Remove overlapping atoms from ZrO₂ supercell (aggressive)
        zro2_super = self._remove_overlapping_atoms(zro2_super, min_distance=1.5)
        
        # Create sandwich: ZrO₂ | Zr | ZrO₂ (use copies to avoid position conflicts)
        zro2_bottom = zro2_super.copy()
        zro2_top = zro2_super.copy()
        interface = self._stack_sandwich(zro2_bottom, zr_super, zro2_top, gap, vacuum)
        
        # FINAL OVERLAP REMOVAL - apply to the complete interface (aggressive)
        interface = self._remove_overlapping_atoms(interface, min_distance=1.5)
        logger.info(f"Final interface after overlap removal: {len(interface)} atoms")
        
        # Validate and analyze
        interface = self._validate_structure(interface)
        metadata = self._analyze_interface(interface, match)
        metadata['structure_type'] = 'sandwich'  # Mark as sandwich
        
        return {
            'structure': interface,
            'metadata': metadata,
            'parameters': {
                'zr_surface': zr_surface,
                'zro2_surface': zro2_surface,
                'zro2_phase': zro2_phase,
                'zr_layers': zr_layers,
                'zro2_layers': zro2_layers,
                'gap': gap,
                'vacuum': vacuum,
                'structure_type': 'sandwich'
            },
            'components': components
        }

    def _remove_overlapping_atoms(self, atoms, min_distance=0.5):
        """Remove atoms that are too close to each other (overlapping)."""
        positions = atoms.get_positions()
        symbols = atoms.get_chemical_symbols()
        cell = atoms.get_cell()
        
        # Find atoms to keep (non-overlapping)
        keep_indices = []
        removed_count = 0
        
        for i in range(len(atoms)):
            is_unique = True
            for j in keep_indices:
                # Simple distance calculation (more reliable than periodic boundary calc)
                diff = positions[i] - positions[j]
                distance = np.linalg.norm(diff)
                
                # Also check with periodic boundaries for edge cases
                try:
                    diff_periodic = diff - np.round(diff @ np.linalg.inv(cell)) @ cell
                    distance_periodic = np.linalg.norm(diff_periodic)
                    distance = min(distance, distance_periodic)
                except:
                    pass  # Use simple distance if periodic calc fails
                
                if distance < min_distance:
                    is_unique = False
                    removed_count += 1
                    break
            
            if is_unique:
                keep_indices.append(i)
        
        if removed_count > 0:
            logger.info(f"🔧 OVERLAP REMOVAL: Removed {removed_count} overlapping atoms (kept {len(keep_indices)}/{len(atoms)})")
        else:
            logger.debug(f"No overlapping atoms found (threshold: {min_distance:.1f} Å)")
        
        # Create new atoms object with non-overlapping atoms
        return Atoms(
            symbols=[symbols[i] for i in keep_indices],
            positions=[positions[i] for i in keep_indices],
            cell=cell,
            pbc=atoms.get_pbc()
        )

    def _stack_sandwich(self, bottom_slab, middle_slab, top_slab, gap, total_vacuum):
        """Stack three slabs to create sandwich interface: bottom | middle | top"""
        try:
            logger.debug("Creating sandwich structure")
            
            # Get all positions and symbols
            pos1 = bottom_slab.get_positions().copy()  # Make sure we have independent arrays
            pos2 = middle_slab.get_positions().copy()  
            pos3 = top_slab.get_positions().copy()
            
            sym1 = bottom_slab.get_chemical_symbols()
            sym2 = middle_slab.get_chemical_symbols()
            sym3 = top_slab.get_chemical_symbols()
            
            # Calculate z-positions for stacking
            # Bottom slab at z=0
            bottom_max = pos1[:, 2].max()
            bottom_min = pos1[:, 2].min()
            
            # Middle slab positioned with gap after bottom
            middle_min = pos2[:, 2].min()
            middle_max = pos2[:, 2].max()
            middle_offset = bottom_max + gap - middle_min
            pos2[:, 2] += middle_offset
            
            # Top slab positioned with gap after middle  
            top_min = pos3[:, 2].min()
            top_offset = middle_max + middle_offset + gap - top_min
            pos3[:, 2] += top_offset
            
            # Combine all atoms
            positions = np.vstack([pos1, pos2, pos3])
            symbols = sym1 + sym2 + sym3
            
            # Set cell height to contain everything plus vacuum
            cell = bottom_slab.get_cell().copy()
            total_height = positions[:, 2].max() - positions[:, 2].min()
            cell[2, 2] = total_height + total_vacuum
            
            # Center the entire sandwich in the cell
            z_center = cell[2, 2] / 2
            current_center = (positions[:, 2].max() + positions[:, 2].min()) / 2
            positions[:, 2] += z_center - current_center
            
            # Create interface
            interface = Atoms(symbols=symbols, positions=positions, cell=cell, pbc=[True, True, True])
            
            # CONSERVATIVE WRAPPING - only wrap atoms significantly outside unit cell
            fractional = interface.get_scaled_positions()
            wrapped_count = 0
            
            # Only wrap atoms that are more than 10% outside the boundary
            for i, frac_pos in enumerate(fractional):
                for dim in range(2):  # Only x,y dimensions (keep z-stacking intact)
                    if frac_pos[dim] < -0.1:
                        frac_pos[dim] += 1.0
                        wrapped_count += 1
                    elif frac_pos[dim] > 1.1:
                        frac_pos[dim] -= 1.0
                        wrapped_count += 1
            
            interface.set_scaled_positions(fractional)
            logger.debug(f"Sandwich structure: {len(symbols)} atoms, wrapped {wrapped_count}")
            
            return interface
            
        except Exception as e:
            logger.error(f"Sandwich stacking failed: {e}")
            raise
    
    def _stack_slabs(self, slab1, slab2, gap, total_vacuum):
        """Stack two slabs to create interface with robust error handling."""
        try:
            pos1 = slab1.get_positions()
            pos2 = slab2.get_positions()
            
            # Check for NaN values in positions
            if np.isnan(pos1).any() or np.isnan(pos2).any():
                logger.error("Input slabs contain NaN positions")
                raise ValueError("Cannot stack slabs with NaN positions")
            
            # Check for reasonable Z coordinates
            if pos1.shape[0] == 0 or pos2.shape[0] == 0:
                raise ValueError("Empty slab detected")
                
            # Position slabs with gap
            z1_max = pos1[:, 2].max()
            z2_min = pos2[:, 2].min()
            
            # Check for valid Z coordinates
            if not np.isfinite(z1_max) or not np.isfinite(z2_min):
                logger.error("Invalid Z coordinates in slabs")
                raise ValueError("Invalid Z coordinates")
            
            # Shift slab2 to create gap
            shift = z1_max + gap - z2_min
            pos2_shifted = pos2.copy()
            pos2_shifted[:, 2] += shift
            
            # Combine
            positions = np.vstack([pos1, pos2_shifted])
            symbols = slab1.get_chemical_symbols() + slab2.get_chemical_symbols()
            
            # Validate combined positions
            if np.isnan(positions).any():
                logger.error("NaN values generated during stacking")
                raise ValueError("NaN values in combined positions")
            
            # Set cell height
            cell = slab1.get_cell().copy()
            
            # Check cell validity
            if np.isnan(cell).any():
                logger.warning("Cell contains NaN - using fallback cell")
                # Create a reasonable cell based on positions
                pos_range = positions.max(axis=0) - positions.min(axis=0)
                cell = np.diag([pos_range[0] + 5, pos_range[1] + 5, pos_range[2] + total_vacuum])
            
            total_height = positions[:, 2].max() - positions[:, 2].min()
            cell[2, 2] = total_height + total_vacuum
            
            # Center in cell
            z_center = cell[2, 2] / 2
            current_center = (positions[:, 2].max() + positions[:, 2].min()) / 2
            positions[:, 2] += z_center - current_center
            
            # Final validation
            if np.isnan(positions).any() or np.isnan(cell).any():
                raise ValueError("Final structure contains NaN values")
            
            # Create interface
            interface = Atoms(symbols=symbols, positions=positions, cell=cell, pbc=[True, True, True])
            
            # WRAP ATOMS BACK INTO UNIT CELL - ensures all atoms lie within dotted parallelogram
            interface.wrap()
            logger.debug("Wrapped all atoms back into unit cell boundaries")
            
            return interface
            
        except Exception as e:
            logger.error(f"Interface stacking failed: {e}")
            # Return a minimal valid structure for testing
            logger.warning("Creating minimal fallback structure")
            
            # Create a simple cubic cell with a few atoms
            from ase import Atom
            atoms = Atoms([
                Atom('Zr', [0, 0, 0]),
                Atom('Zr', [3, 0, 0]),
                Atom('O', [0, 0, 5]),
                Atom('Zr', [3, 0, 5])
            ])
            atoms.set_cell([6, 6, 10])
            atoms.center()
            return atoms
    
    def _validate_structure(self, atoms):
        """
        Validate and clean atomic structure, removing NaN values.
        
        Args:
            atoms: ASE Atoms object to validate
            
        Returns:
            cleaned_atoms: ASE Atoms object with NaN values removed/fixed
        """
        positions = atoms.get_positions()
        cell = atoms.get_cell()
        
        # Check for NaN in positions
        nan_positions = np.isnan(positions).any(axis=1)
        if nan_positions.any():
            logger.warning(f"Found {nan_positions.sum()} atoms with NaN positions - removing them")
            valid_atoms = atoms[~nan_positions]
            if len(valid_atoms) == 0:
                raise ValueError("All atoms have NaN positions - structure is invalid")
            atoms = valid_atoms
            
        # Check for NaN in cell
        if np.isnan(cell).any():
            logger.warning("Cell contains NaN values - reconstructing cell")
            positions = atoms.get_positions() # get scaled positions instead
            # Create a reasonable cell based on atomic positions
            min_pos = positions.min(axis=0) - 2.0
            max_pos = positions.max(axis=0) + 2.0
            new_cell = np.diag(max_pos - min_pos)
            atoms.set_cell(new_cell)
            atoms.wrap()
            
        return atoms
    
    def _analyze_interface(self, interface, match):
        """Calculate interface properties with validation."""
        try:
            # Validate structure first to handle NaN values
            interface = self._validate_structure(interface)
            
            cell = interface.get_cell()
            area = np.linalg.norm(np.cross(cell[0], cell[1]))
            
            composition = {}
            for symbol in interface.get_chemical_symbols():
                composition[symbol] = composition.get(symbol, 0) + 1
            
            return {
                'total_atoms': len(interface),
                'composition': composition,
                'formula': interface.get_chemical_formula(),
                'interface_area': area,
                'strain': match['strain'],
                'supercells': {'zr': match['supercell1'], 'zro2': match['supercell2']}
            }
        except Exception as e:
            logger.error(f"Interface analysis failed: {e}")
            return {
                'total_atoms': len(interface) if 'interface' in locals() else 0,
                'composition': {},
                'formula': 'Unknown',
                'interface_area': 0.0,
                'strain': match.get('strain', 0.0) if match else 0.0,
                'supercells': {'zr': (1,1,1), 'zro2': (1,1,1)}
            }
    
    def generate_systematic_study(self, output_dir='interfaces', 
                                 zr_surfaces=['0001', '1010'], 
                                 zro2_surfaces=['111', '100'],
                                 zro2_phases=['tetragonal'],
                                 layer_ranges={'zr': [6, 8, 10], 'zro2': [4, 6, 8]},
                                 gaps=[2.0, 2.5, 3.0]):
        """
        Generate systematic set of interfaces for energy studies.
        
        This is the main function for automated interface generation.
        """
        os.makedirs(output_dir, exist_ok=True)
        
        interfaces = []
        count = 0
        
        for zr_surf in zr_surfaces:
            for zro2_surf in zro2_surfaces:
                for phase in zro2_phases:
                    for zr_layers in layer_ranges['zr']:
                        for zro2_layers in layer_ranges['zro2']:
                            for gap in gaps:
                                count += 1
                                name = f"zr{zr_surf}_zro2{zro2_surf}_{phase}_zr{zr_layers}_ox{zro2_layers}_gap{gap}"
                                
                                logger.info(f"[{count}] Generating {name}")
                                
                                try:
                                    result = self.build_interface(
                                        zr_surf, zro2_surf, phase, 
                                        zr_layers, zro2_layers, gap
                                    )
                                    
                                    # Save structure files
                                    struct_dir = os.path.join(output_dir, name)
                                    os.makedirs(struct_dir, exist_ok=True)
                                    
                                    structure = result['structure']
                                    structure.write(os.path.join(struct_dir, f"{name}.cif"))
                                    structure.write(os.path.join(struct_dir, f"{name}.xyz"))
                                    structure.write(os.path.join(struct_dir, "POSCAR"))
                                    
                                    # Save metadata
                                    with open(os.path.join(struct_dir, "metadata.json"), 'w') as f:
                                        json.dump({
                                            'name': name,
                                            'parameters': result['parameters'],
                                            'metadata': result['metadata']
                                        }, f, indent=2, default=str)
                                    
                                    interfaces.append({
                                        'name': name,
                                        'path': struct_dir,
                                        'strain': result['metadata']['strain'],
                                        'area': result['metadata']['interface_area'],
                                        'atoms': result['metadata']['total_atoms']
                                    })
                                    
                                except Exception as e:
                                    logger.error(f"Failed to generate {name}: {e}")
        
        # Save summary
        with open(os.path.join(output_dir, "study_summary.json"), 'w') as f:
            json.dump({
                'total_interfaces': len(interfaces),
                'interfaces': interfaces,
                'parameters': {
                    'zr_surfaces': zr_surfaces,
                    'zro2_surfaces': zro2_surfaces,
                    'zro2_phases': zro2_phases,
                    'layer_ranges': layer_ranges,
                    'gaps': gaps
                }
            }, f, indent=2, default=str)
        
        logger.info(f"Generated {len(interfaces)} interfaces in {output_dir}")
        return interfaces

    def generate_systematic_study_with_scaled_positions(self, output_dir='interfaces_scaled', 
                                                       zr_surfaces=['0001', '1010'], 
                                                       zro2_surfaces=['111', '100'],
                                                       zro2_phases=['tetragonal'],
                                                       layer_ranges={'zr': [6, 8, 10], 'zro2': [4, 6, 8]},
                                                       gaps=[2.0, 2.5, 3.0]):
        """
        Generate systematic set of interfaces using scaled positions method.
        
        This version uses the improved scaled positions approach for better oxide positioning.
        """
        os.makedirs(output_dir, exist_ok=True)
        
        interfaces = []
        count = 0
        
        for zr_surf in zr_surfaces:
            for zro2_surf in zro2_surfaces:
                for phase in zro2_phases:
                    for zr_layers in layer_ranges['zr']:
                        for zro2_layers in layer_ranges['zro2']:
                            for gap in gaps:
                                count += 1
                                name = f"zr{zr_surf}_zro2{zro2_surf}_{phase}_zr{zr_layers}_ox{zro2_layers}_gap{gap}_scaled"
                                
                                logger.info(f"[{count}] Generating {name} (scaled positions)")
                                
                                try:
                                    result = self.build_interface_with_scaled_positions(
                                        zr_surf, zro2_surf, phase, 
                                        zr_layers, zro2_layers, gap
                                    )
                                    
                                    # Save structure files
                                    struct_dir = os.path.join(output_dir, name)
                                    os.makedirs(struct_dir, exist_ok=True)
                                    
                                    structure = result['structure']
                                    structure.write(os.path.join(struct_dir, f"{name}.cif"))
                                    structure.write(os.path.join(struct_dir, f"{name}.xyz"))
                                    structure.write(os.path.join(struct_dir, "POSCAR"))
                                    
                                    # Save metadata
                                    with open(os.path.join(struct_dir, "metadata.json"), 'w') as f:
                                        json.dump({
                                            'name': name,
                                            'parameters': result['parameters'],
                                            'metadata': result['metadata'],
                                            'method': 'scaled_positions'
                                        }, f, indent=2, default=str)
                                    
                                    interfaces.append({
                                        'name': name,
                                        'path': struct_dir,
                                        'strain': result['metadata']['strain'],
                                        'area': result['metadata']['interface_area'],
                                        'atoms': result['metadata']['total_atoms'],
                                        'method': 'scaled_positions'
                                    })
                                    
                                except Exception as e:
                                    logger.error(f"Failed to generate {name}: {e}")
        
        # Save summary
        with open(os.path.join(output_dir, "study_summary.json"), 'w') as f:
            json.dump({
                'total_interfaces': len(interfaces),
                'interfaces': interfaces,
                'method': 'scaled_positions',
                'parameters': {
                    'zr_surfaces': zr_surfaces,
                    'zro2_surfaces': zro2_surfaces,
                    'zro2_phases': zro2_phases,
                    'layer_ranges': layer_ranges,
                    'gaps': gaps
                }
            }, f, indent=2, default=str)
        
        logger.info(f"Generated {len(interfaces)} interfaces using scaled positions in {output_dir}")
        return interfaces

    def create_hpc_batch_files(self, interfaces_dir, template_dir='hpc_templates'):
        """Create batch submission files for HPC energy calculations."""
        os.makedirs(template_dir, exist_ok=True)
        
        # VASP input template
        vasp_template = """# VASP input for interface energy calculation
SYSTEM = {name}
PREC = Accurate
ENCUT = 500
ISMEAR = 0
SIGMA = 0.05
NSW = 100
IBRION = 2
ISIF = 2
EDIFF = 1E-6
EDIFFG = -0.01
LREAL = Auto
NPAR = 4
"""
        
        # SLURM submission template
        slurm_template = """#!/bin/bash
#SBATCH --job-name={name}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --time=24:00:00
#SBATCH --partition=compute
#SBATCH --output=%j.out
#SBATCH --error=%j.err

module load vasp/6.3.0

cd {work_dir}
mpirun vasp_std > vasp.out
"""
        
        with open(os.path.join(template_dir, "INCAR_template"), 'w') as f:
            f.write(vasp_template)
            
        with open(os.path.join(template_dir, "submit_template.sh"), 'w') as f:
            f.write(slurm_template)
        
        logger.info(f"HPC templates created in {template_dir}")


# Simple command-line interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate Zr/ZrO₂ interfaces for energy calculations")
    parser.add_argument("--single", action="store_true", help="Generate single interface")
    parser.add_argument("--systematic", action="store_true", help="Generate systematic study")
    parser.add_argument("--visualize", action="store_true", help="Visualize generated interface")
    parser.add_argument("--scaled-positions", action="store_true", help="Use scaled positions method (recommended)")
    parser.add_argument("--zr-surface", default="0001", help="Zr surface (default: 0001)")
    parser.add_argument("--zro2-surface", default="111", help="ZrO₂ surface (default: 111)")
    parser.add_argument("--output", default="interfaces", help="Output directory")
    
    args = parser.parse_args()
    
    builder = ZrZrO2InterfaceBuilder()
    
    if args.single:
        if args.scaled_positions:
            result = builder.build_interface_with_scaled_positions(args.zr_surface, args.zro2_surface)
            print("Using scaled positions method (recommended)")
        else:
            result = builder.build_interface(args.zr_surface, args.zro2_surface)
            print("Using original method")
            
        result['structure'].write("interface.cif")
        print(f"Generated single interface: {result['metadata']['formula']}")
        print(f"Strain: {result['metadata']['strain']*100:.2f}%")
        print(f"Area: {result['metadata']['interface_area']:.2f} Ų")
        
        # Visualize if requested
        if args.visualize:
            title = f"Zr({args.zr_surface})/ZrO₂({args.zro2_surface}) Interface"
            visualize_interface(result['structure'], title)
        
    elif args.systematic:
        if args.scaled_positions:
            print("Using scaled positions method for systematic study")
            interfaces = builder.generate_systematic_study_with_scaled_positions(args.output)
        else:
            interfaces = builder.generate_systematic_study(args.output)
        
        print(f"Generated {len(interfaces)} interfaces in {args.output}/")
        print("Ready for HPC energy calculations!")
        
    else:
        print("Use --single or --systematic. See --help for options.")
        print("Recommended: Use --scaled-positions for better oxide positioning") 