#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
import os

input_dir = "interaction_details_csv_cif"
output_dir = "binary_interactions_csv"
os.makedirs(output_dir, exist_ok=True)


for csv_file in os.listdir(input_dir):
    if csv_file.endswith("_interactions.csv.gz"):
        csv_path = os.path.join(input_dir, csv_file)
        df = pd.read_csv(csv_path, compression='gzip')
        
        df_heavy = df.dropna(subset=['Atom_Residue', 'Atom_Ligand'])

        # Convert 'Classification' column directly to binary interaction
        df_heavy['Interacting'] = (df_heavy['Classification'] == 'Interacting').astype(int)

        # Summarize interactions at residue-level
        interaction_summary = df_heavy.groupby(
            ['PDB ID', 'PDB Chain', 'LIG ID', 'Chain_Id_Ligand', 'Position', 'Residue_1L', 'Residue_3L'],
            as_index=False
        ).agg({
            'Interacting': 'max',
            'Ligand_Type': 'first',
            'Organism': 'first',
            'Resolution': 'first',
            'Experimental_Method': 'first',
            'Canonical_Smile': 'first',
            'Inchikey': 'first'
        })
        
        # Save binary CSV if at least one positive
        if interaction_summary['Interacting'].max() > 0:
            output_csv = os.path.join(output_dir, csv_file.replace("_interactions.csv", "_binary.csv"))
            interaction_summary.to_csv(output_csv, index=False, compression='gzip')
            print(f"Saved binary interactions: {output_csv}")
        else:
            print(f"No interactions found in {csv_file}; binary file not saved.")

print("All files processed successfully!")