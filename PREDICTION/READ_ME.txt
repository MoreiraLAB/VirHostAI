ViralBindPredict: Empowering Viral Protein-Ligand Binding Sites through Deep Learning and Protein Sequence-Derived Insights

New prediction:
   -> mordred_features.py
        Uses as input files the file ./ViralBindPredict/NEW_PREDICTION/ligand_predict.smi
        Creates a folder ./ViralBindPredict/FEATURES/MORDRED/ to store Mordred descriptors
        Creates mordred.hdf5, a hdf5 file in ./ViralBindPredict/H5_FILES/ folder, mordred_keys.txt with h5 file keys, mordred_descriptors.txt with a list of descriptors and mordred_error.csv in case of errors

   -> spotone_features.py
        Uses as input files the file ./ViralBindPredict/NEW_PREDICTION/protein_predict.fasta
        Creates a folder ./ViralBindPredict/FEATURES/SPOTONE/ to store SPOTONE descriptors and individual fasta files
        Creates spotone.hdf5, a hdf5 file in ./ViralBindPredict/H5_FILES/ folder, spotone_keys.txt with h5 file keys, and spotone_descriptors.txt with a list of descriptors

   -> pssm_features.py
        Uses as input files the file ./ViralBindPredict/FEATURES/SPOTONE/ individual fasta files
        Creates a folder ./ViralBindPredict/FEATURES/PSSM/ to store PSSM descriptors
        Creates pssm.hdf5, a hdf5 file in ./ViralBindPredict/H5_FILES/ folder, pssm_keys.txt with h5 file keys, and pssm_descriptors.txt with a list of descriptors

Output file: The new predictions will be stored, one csv file per interaction at ./ViralBindPredict/NEW_PREDICTION/
