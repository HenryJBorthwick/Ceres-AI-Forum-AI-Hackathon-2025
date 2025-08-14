import pathlib
import pandas as pd


def filter_first_projection(
    input_csv: pathlib.Path,
    output_csv: pathlib.Path,
    validity_column: str = "Validity period",
    projection_value: str = "first projection",
    phase_column: str = "Phase",
    all_phase_value: str = "all",
    three_plus_value: str = "3+",
) -> None:
    """Filter out rows whose validity period equals a projection value and phase equals 'all' or '3+'.

    Parameters
    ----------
    input_csv : pathlib.Path
        Path to the original IPC CSV file.
    output_csv : pathlib.Path
        Destination path for the filtered CSV.
    validity_column : str, optional
        Name of the column containing the validity period descriptor.
    projection_value : str, optional
        The value indicating a projection period to remove.
    phase_column : str, optional
        Name of the column containing the phase descriptor.
    all_phase_value : str, optional
        The phase value to remove (typically 'all').
    three_plus_value : str, optional
        The phase value to remove (typically '3+').
    """

    if not input_csv.exists():
        raise FileNotFoundError(f"Input file not found: {input_csv}")

    df = pd.read_csv(input_csv)

    before = len(df)
    
    # Filter out 'first projection' rows
    df_filtered = df[df[validity_column].str.lower() != projection_value.lower()]
    after_projection_filter = len(df_filtered)
    
    # Filter out 'all' phase rows
    df_filtered = df_filtered[df_filtered[phase_column].str.lower() != all_phase_value.lower()]
    after_all_filter = len(df_filtered)
    
    # Filter out '3+' phase rows
    df_filtered = df_filtered[df_filtered[phase_column] != three_plus_value]
    after_three_plus_filter = len(df_filtered)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df_filtered.to_csv(output_csv, index=False)

    projection_removed = before - after_projection_filter
    all_removed = after_projection_filter - after_all_filter
    three_plus_removed = after_all_filter - after_three_plus_filter
    total_removed = before - after_three_plus_filter

    print(
        f"Filtered {projection_removed} 'first projection' rows.\n"
        f"Filtered {all_removed} 'all' phase rows.\n"
        f"Filtered {three_plus_removed} '3+' phase rows.\n"
        f"Total: removed {total_removed} rows (kept {after_three_plus_filter} of {before}).\n"
        f"Filtered CSV written to {output_csv}"
    )


if __name__ == "__main__":
    src_dir = pathlib.Path(__file__).resolve().parent

    input_path = src_dir / "data" / "ipc_global_area_long.csv"
    output_path = src_dir / "data" / "ipc_global_area_long_current_only.csv"

    filter_first_projection(input_path, output_path) 