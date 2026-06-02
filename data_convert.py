import os
import pandas as pd
import json

def normalize_component_name(raw_key):
    """
    Takes a raw string from Excel and returns a standardized component name if it's valid.
    Otherwise, returns None. This is the core of the cleaning logic.
    """
    if not isinstance(raw_key, str):
        return None

    # Standardize: remove whitespace and convert to uppercase
    clean_key = raw_key.strip().upper()
    if not clean_key:
        return None

    # --- Rule-based Normalization ---
    # More specific rules should come first
    if 'AGED' in clean_key and 'PVC' in clean_key:
        return 'AGED-PVC'
    if clean_key.startswith('MGCO'):  # Catches MgCO3, MgCO13, MGC03 etc.
        return 'MGCO3'
    if clean_key.startswith('TIO'):   # Catches TiO2, TIO, etc.
        return 'TIO2'
    if clean_key == 'PE WAX':
        return 'PE WAX'
    if clean_key == 'PE':
        return 'PE'
    if clean_key == 'CAST' or clean_key == 'CA ST': # Handles variations like CaSt
        return 'CAST'
    if clean_key == 'ZNST' or clean_key == 'ZN ST':
        return 'ZNST'
    if clean_key == 'BAST' or clean_key == 'BA ST':
        return 'BAST'
    
    # Check against a set of exact matches for other components
    exact_matches = {'PVC', 'TCPP', 'ATBC', 'PPA', 'ACR', 'ORGANOTIN', 'PLUMBAGO'}
    if clean_key in exact_matches:
        return clean_key
        
    # If no rule matches, it's not a recognized component
    return None


def process_raman_data(input_dir, output_path):
    """
    Traverses all Excel files in a directory, intelligently extracts and normalizes
    Raman spectral data, and saves it to a single JSON file.
    """
    all_samples_data = []

    if not os.path.isdir(input_dir):
        print(f"Error: Directory '{input_dir}' not found.")
        return

    excel_files = [f for f in os.listdir(input_dir) if f.endswith(('.xlsx', '.xls'))]
    if not excel_files:
        print(f"No Excel files found in '{input_dir}'.")
        return

    print(f"Found {len(excel_files)} Excel files. Starting processing...")

    for filename in excel_files:
        file_path = os.path.join(input_dir, filename)
        print(f"\n--- Processing file: {filename} ---")
        try:
            xls = pd.ExcelFile(file_path)
        except Exception as e:
            print(f"Could not read file {filename}. Error: {e}")
            continue

        for sheet_name in xls.sheet_names:
            try:
                df = pd.read_excel(xls, sheet_name=sheet_name, header=None)
                print(f"  - Reading sheet: '{sheet_name}'")
                num_samples_in_sheet = 0

                for i in range(0, df.shape[1], 2):
                    if pd.isna(df.iloc[0, i]):
                        continue
                    
                    components = {}
                    spectrum_start_row = -1

                    for row_idx in range(df.shape[0]):
                        key_raw = df.iloc[row_idx, i]
                        
                        # Use our new normalization function
                        normalized_key = normalize_component_name(key_raw)
                        
                        if normalized_key:
                            # It's a valid component, add it to our dictionary
                            value = df.iloc[row_idx, i+1]
                            try:
                                numeric_value = float(value)
                            except (ValueError, TypeError):
                                numeric_value = 0.0 # Assign 0 if content is not a number
                            
                            # Add content to existing key if it's a duplicate
                            components[normalized_key] = components.get(normalized_key, 0.0) + numeric_value
                        else:
                            # Not a valid component name, assume this is where the spectrum starts
                            spectrum_start_row = row_idx
                            break
                    
                    if spectrum_start_row == -1:
                        print(f"    -> Warning: Could not find the start of spectrum data for sample {(i//2) + 1}. Skipping.")
                        continue

                    spectrum_df = df.iloc[spectrum_start_row:, [i, i+1]].copy()
                    spectrum_df.columns = ['wavenumber', 'intensity']
                    
                    spectrum_df['wavenumber'] = pd.to_numeric(spectrum_df['wavenumber'], errors='coerce')
                    spectrum_df['intensity'] = pd.to_numeric(spectrum_df['intensity'], errors='coerce')
                    spectrum_df.dropna(how='any', inplace=True)

                    sample_id = f"{os.path.splitext(filename)[0]}_{sheet_name}_sample_{(i//2) + 1}"
                    sample_data = {
                        "sample_id": sample_id,
                        "components": components,
                        "raman_spectrum": {
                            "wavenumber": spectrum_df['wavenumber'].tolist(),
                            "intensity": spectrum_df['intensity'].tolist()
                        }
                    }
                    all_samples_data.append(sample_data)
                    num_samples_in_sheet += 1
                
                if num_samples_in_sheet > 0:
                    print(f"    -> Successfully processed {num_samples_in_sheet} samples from this sheet.")

            except Exception as e:
                print(f"    -> An error occurred while processing sheet '{sheet_name}': {e}")
                continue

    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(all_samples_data, f, ensure_ascii=False, indent=4)
        print(f"\n✅ Processing complete! All data has been saved to: {output_path}")
        print(f"Total samples processed: {len(all_samples_data)}")
    except Exception as e:
        print(f"\n❌ Error saving JSON file: {e}")

if __name__ == "__main__":
    input_directory = "/megadisk/fanghengyu/ai4science"
    output_json_file = "raman_dataset_cleaned_v2.json" # Use a new name
    
    # Instructions:
    # 1. Replace your old script with this one.
    # 2. Run this script. It will generate 'raman_dataset_cleaned_v2.json'.
    # 3. Use this new JSON file in your main training script.
    process_raman_data(input_directory, output_json_file)