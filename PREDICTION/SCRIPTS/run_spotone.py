#!/usr/bin/env python

"""
Extract sequence based features on a protocol including window size
"""

__author__ = "C. Marques-Pereira, A.J. Preto"
__email__ = "amarques@cnc.uc.pt"
__group__ = "Data-Driven Molecular Design"
__group_leader__ = "Irina S. Moreira"
__project__ = "ViralBindPredict: Empowering Viral Protein-Ligand Binding Sites through Deep Learning and Protein Sequence-Derived Insights"

from ViralBindPredict_variables import HEADER_COLUMNS, ONE_TO_THRE_CODE_CONVERTER, \
                                INTERMEDIATE_SEP, SYSTEM_SEP, SEP, COMPLEX_NAME_COL, \
                                CHAIN_NAME_COL, RES_NUMBER_COL, ENCODING_FILE, \
                                ENCODING_RESIDUE_NAME, RES_NAME_COL, THREE_TO_ONE_CODE_CONVERTER, \
                                AMINO_PROPERTIES_FILE, AMINO_PROPERTIES_THREE_CODE_COL, \
                                FEATURES_HOLDER_COLS, CENTRED_FEATURES_WINDOWS, \
                                SECTIONS_FEATURES_SPLITS, PROCESSED_TERMINATION, FEATURES_TERMINATION, SPOTONE_FOLDER, H5_FOLDER
import pandas as pd
import string
import sys
import os
import h5py

def generate_encoded(input_sequence):

    output_table = []
    encoded_table = utilities().table_opener(ENCODING_FILE)
    for residue_number in input_sequence.keys():
        residue_letter = utilities().converter[input_sequence[residue_number]]
        encoded_residue = encoded_table.loc[encoded_table[class_id_name] == residue_letter].iloc[:,1:]
        proper_row = [residue_number] + list(encoded_residue.values[0])
        output_table.append(proper_row)
    header = [class_id_output] + list(encoded_residue)
    return pd.DataFrame(output_table, columns = header)

def transform_file(input_file):

    opened_file = open(input_file,"r").readlines()
    error_file = SYSTEM_SEP.join(input_file.split(SYSTEM_SEP)[0:-1]) + SYSTEM_SEP + "error"
    record_chains, chains_dict, default_chains_list = {}, {}, string.ascii_uppercase
    count = 0
    amino_list = list(THREE_TO_ONE_CODE_CONVERTER.values())
    found_irregular = False
    for row in opened_file:
        row = row.replace("\n","")
        if row[0] == ">":
            try:
                chain_name = row[1:].split(":")[0].lower() + \
                INTERMEDIATE_SEP + row[1:].split(":")[1][0]
                chains_dict[chain_name] = ""
                record_chains[chain_name] = row[1:]
            except:
                chain_name = row[1:]
                chains_dict[chain_name] = ""
                try:
                    record_chains[chain_name] = row.split(INTERMEDIATE_SEP)[1]
                except:
                    record_chains[chain_name] = row
            count += 1
        else:
            chains_dict[chain_name] += row
            for amino_acid in row:
                if amino_acid not in amino_list:
                    found_irregular = True
    if found_irregular == True:
        with open(error_file, "w") as error_check:
            error_check.write("found error")
    return chains_dict, record_chains

def generate_table_file(input_dict, output_file):

    """
    Generate a table with the most basic of features
    """
    output_list = []
    for entry in input_dict.keys():
        count = 1
        for residue in input_dict[entry]:
            residue_name = ONE_TO_THRE_CODE_CONVERTER[residue]
            current_row = [entry.split(INTERMEDIATE_SEP)[0],entry.split(INTERMEDIATE_SEP)[1],str(count),residue_name]
            output_list.append(current_row)
            count += 1

    output_table = pd.DataFrame(output_list, columns = HEADER_COLUMNS)
    output_table.to_csv(output_file, sep = SEP, index = False)

def location_features(input_sequence, target_residue):

    full_size = len(input_sequence)
    relative_position = float(target_residue/full_size)
    if relative_position < 0.25: return [1.0]
    if (relative_position >= 0.25) and (relative_position < 0.5): return [2.0]
    if (relative_position >= 0.5) and (relative_position < 0.75): return [3.0]
    if relative_position >= 0.75: return [4.0]

def extract_amino_properties(input_res, retrieved_header = False):

    opened_holder = pd.read_csv(AMINO_PROPERTIES_FILE, sep = SEP, header = 0)
    res_row = opened_holder.loc[opened_holder[AMINO_PROPERTIES_THREE_CODE_COL] == input_res]
    if retrieved_header == False: return list(res_row)[3:], list(res_row.values[0])[3:]
    elif retrieved_header == True: return list(res_row.values[0])[3:]

def extract_subsections(res_number, input_dict):

    all_sections = []
    for current_range in CENTRED_FEATURES_WINDOWS:
        if (int(res_number) - current_range) < 1: start_range = 1
        else: start_range = int(res_number) - current_range
        if (int(res_number) + current_range) > int(list(input_dict.keys())[-1]): end_range = int(list(input_dict.keys())[-1])
        else: end_range = int(res_number) + current_range
        holder_list = []
        for amino_acid in range(start_range, end_range + 1):
            current_residue = input_dict[str(amino_acid)]
            holder_list.append(current_residue)
        all_sections += [holder_list]
    return all_sections

def extract_centred(input_dict, res_number, features_table):

    output_features = []
    calculated_sections = extract_subsections(res_number, input_dict)
    id_column = features_table[AMINO_PROPERTIES_THREE_CODE_COL]
    for current_feature in FEATURES_HOLDER_COLS:
        current_table = pd.concat([id_column, features_table[current_feature]], axis = 1)
        current_table = current_table.set_index(AMINO_PROPERTIES_THREE_CODE_COL).transpose()
        for sequence_calc in calculated_sections:
            holder_list = []
            for amino_acid in sequence_calc:
                try:
                    holder_list += [float(current_table[amino_acid])]
                except: continue
            try: features_average = sum(holder_list)/len(holder_list)
            except: features_average = 0
            output_features += [features_average]
    return output_features

def extract_sections(input_dict, features_table):

    output_features = []
    protein_length = len(list(input_dict.keys()))
    for number_of_sections in SECTIONS_FEATURES_SPLITS:
        section_size = int(protein_length/number_of_sections)
        section_start = int(list(input_dict.keys())[0])
        for current_section in range(1, number_of_sections + 1):
            end_range = section_start + section_size
            holder_list = []
            if end_range > protein_length: end_range = protein_length
            for amino_acid in range(section_start, end_range):
                try:
                    current_residue = input_dict[str(amino_acid)]
                    holder_list.append(float(features_table.loc[features_table[AMINO_PROPERTIES_THREE_CODE_COL] == current_residue][FEATURES_HOLDER_COLS[-1]]))
                except: continue
            try: features_average = sum(holder_list)/len(holder_list)
            except: features_average =  0
            output_features += [features_average]
            section_start += section_size
    return output_features

def render_discriminated_sequence_dict(input_sequence, chain, first_input = True):

    if first_input == True:
        output_dict, output_dict[chain], count = {}, {}, 1
        for residue in input_sequence:
            output_dict[chain][str(count)] = ONE_TO_THRE_CODE_CONVERTER[residue]
            count += 1
        return output_dict
    elif first_input == False:
        output_dict, count = {}, 1
        for residue in input_sequence:
            output_dict[str(count)] = ONE_TO_THRE_CODE_CONVERTER[residue]
            count += 1
        return output_dict

def generate_simple_features(input_dict, input_file, chains_original, input_raw_file):

    encoding_table = pd.read_csv(ENCODING_FILE, sep = SEP, header = 0)
    template_features_table = pd.read_csv(AMINO_PROPERTIES_FILE, sep = SEP, header = 0)
    opened_file = pd.read_csv(input_file, sep = SEP, header = 0)
    output_table, written_header = [], False
    discriminated_dictionary = {}
    row_count = 1
    for index, row in opened_file.iterrows():
        current_chain = row[COMPLEX_NAME_COL] + INTERMEDIATE_SEP + row[CHAIN_NAME_COL]
        current_res = row[RES_NAME_COL]
        one_hot_encoding = list(encoding_table.loc[encoding_table[ENCODING_RESIDUE_NAME] == THREE_TO_ONE_CODE_CONVERTER[current_res]].drop([ENCODING_RESIDUE_NAME], axis = 1).values[0])
        relative_position = location_features(input_dict[current_chain], row[RES_NUMBER_COL])
        if (row[COMPLEX_NAME_COL] in list(discriminated_dictionary.keys())) and (row[CHAIN_NAME_COL] not in list(discriminated_dictionary[row[COMPLEX_NAME_COL]].keys())):
            discriminated_dictionary[row[COMPLEX_NAME_COL]][row[CHAIN_NAME_COL]] = render_discriminated_sequence_dict(input_dict[current_chain], row[CHAIN_NAME_COL], first_input = False)
        elif row[COMPLEX_NAME_COL] not in list(discriminated_dictionary.keys()):
            discriminated_dictionary[row[COMPLEX_NAME_COL]] = render_discriminated_sequence_dict(input_dict[current_chain], row[CHAIN_NAME_COL], first_input = True)
        window_centred = extract_centred(discriminated_dictionary[row[COMPLEX_NAME_COL]][row[CHAIN_NAME_COL]], str(row[RES_NUMBER_COL]), template_features_table)
        if written_header == True:
            table_properties = extract_amino_properties(row[RES_NAME_COL], retrieved_header = True)
        if written_header == False:
            header_properties, table_properties = extract_amino_properties(row[RES_NAME_COL], retrieved_header = False)
            header = HEADER_COLUMNS  + ["Binary " + str(i) for i in range(1, 21)] + ["Relative distance"] + \
                        header_properties + ["Position " + str(i) for i in range(1, len(window_centred) + 1)]
            written_header = True
        #print("Current row:", row_count, "/", opened_file.shape[0])
        row_count += 1
        proper_row = list(row.values) + one_hot_encoding  + relative_position + \
                        table_properties + window_centred
        output_table.append(proper_row)
    output_dataframe = pd.DataFrame(output_table, columns = header)
    final_name = input_raw_file + FEATURES_TERMINATION
    output_dataframe.to_csv(final_name, sep = SEP, index = False)
    return header


path = SPOTONE_FOLDER

file = str(sys.argv[1])
test_file = path+file

output_file_name  = test_file + PROCESSED_TERMINATION
processed_chains, original_chains_names = transform_file(test_file)

generate_table_file(processed_chains, output_file_name)
header = generate_simple_features(processed_chains, output_file_name, original_chains_names, test_file)

with open(H5_FOLDER + "spotone_descriptors.txt", "w") as descriptors:
    descriptors.write("\n".join(map(str, header[4:])))
