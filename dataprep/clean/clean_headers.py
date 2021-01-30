"""
Implement clean_headers function
"""
import re
from typing import Any, Optional, Union, List, Dict
from unicodedata import normalize

import dask
import dask.dataframe as dd
import pandas as pd

from ..eda.progress_bar import ProgressBar
from .utils import NULL_VALUES, to_dask


def clean_headers(
    df: Union[pd.DataFrame, dd.DataFrame],
    case: str = "snake",
    replace: Optional[Dict[str, str]] = None,
    remove_accents: bool = True,
    report: bool = True,
    progress: bool = True,
) -> pd.DataFrame:
    """
    Function to clean headers / column names.

    Parameters
    ----------
    df
        Dataframe from which column names are to be cleaned.

    case
        The desired case style of the column name.
            - 'snake': 'column_name'
            - 'kebab': 'column-name'
            - 'camel': 'columnName'
            - 'pascal': 'ColumnName'
            - 'const': 'COLUMN_NAME'
            - 'sentence': 'Column name'
            - 'title': 'Column Name'
            - 'lower': 'column name'
            - 'upper': 'COLUMN NAME'

        (default: 'snake')

    replace
        Values to replace in the column names.
            - {"old_value": "new_value"}

        (default: None)

    remove_accents
        If True, strip accents from the column names.

        (default: True)

    report
        If True, output the summary report. Otherwise, no report is outputted.

        (default: True)

    progress
        If True, enable the progress bar.

        (default: True)

    Examples
    --------
    Clean column names by converting the names to camel case style, removing accents,
    and correcting a mispelling.

    >>> df = pd.DataFrame({"FirstNom":["Philip", "Turanga"], "lastName": ["Fry", "Leela"],
                           "Téléphone":["555-234-5678", "(604) 111-2335"]})
    >>> clean_headers(df, case='camel', replace={"Nom": "Name"})
    Headers Cleaning Report:
        2 values cleaned (66.67%)
    Result contains 3 (100.0%) values in the correct format and 0 null values (0.0%)
      firstName lastName       telephone
    0    Philip      Fry    555-234-5678
    1   Turanga    Leela  (604) 111-2335
    """
    # pylint: disable=too-many-arguments

    case_styles = {
        "snake",
        "kebab",
        "camel",
        "pascal",
        "const",
        "sentence",
        "title",
        "lower",
        "upper",
    }

    if case not in case_styles:
        raise ValueError(
            f"case {case} is invalid, it needs to be one of {', '.join(c for c in case_styles)}"
        )

    # Convert to dask
    df = to_dask(df)

    # Store original column names for creating cleaning report
    orig_columns = df.columns

    if replace:
        df = df.rename(columns=lambda col: _replace_values(col, replace))

    if remove_accents:
        df = df.rename(columns=_remove_accents)

    df = df.rename(columns=lambda col: _convert_case(col, case))

    df.columns = _rename_duplicates(df.columns, case)

    stats = {}
    # Count the number of changed column names
    stats["cleaned"] = len(df.columns.astype(str).difference(orig_columns.astype(str)))
    # Count null column values
    stats["null"] = len([col for col in list(df.columns) if col in NULL_VALUES])

    with ProgressBar(minimum=1, disable=not progress):
        df = dask.compute(df)[0]

    # Output a report describing the result of clean_headers
    if report:
        _create_report(stats, len(df.columns))

    return df


def _convert_case(name: Any, case: str) -> Any:
    """
    Convert case style of a column name.

    Parameters
    ----------
    name
        Column name.
    case
        The desired case style of the column name.
    """
    if name in NULL_VALUES:
        return name

    if case in {"snake", "kebab", "camel", "pascal", "const"}:
        words = _split_strip_string(str(name))
    else:
        words = _split_string(str(name))

    if case == "snake":
        name = "_".join(words).lower()
    if case == "kebab":
        name = "-".join(words).lower()
    if case == "camel":
        name = words[0].lower() + "".join(w.capitalize() for w in words[1:])
    if case == "pascal":
        name = "".join(w.capitalize() for w in words)
    if case == "const":
        name = "_".join(words).upper()
    if case == "sentence":
        name = " ".join(words).capitalize()
    if case == "title":
        name = " ".join(w.capitalize() for w in words)
    if case == "lower":
        name = " ".join(words).lower()
    if case == "upper":
        name = " ".join(words).upper()

    return name


def _split_strip_string(string: str) -> List[str]:
    """
    Split the string into separate words and strip punctuation
    and special characters.
    """
    string = re.sub(r"[!()*+\,\-./:;<=>?[\]^_{|}~]", " ", string)
    string = re.sub(r"[\'\"\`]", "", string)

    return re.sub(r"([A-Z][a-z]+)", r" \1", re.sub(r"([A-Z]+|[0-9]+|\W+)", r" \1", string)).split()


def _split_string(string: str) -> List[str]:
    """
    Split the string into separate words.
    """
    string = re.sub(r"[\-_]", " ", string)

    return re.sub(r"([A-Z][a-z]+)", r" \1", re.sub(r"([A-Z]+)", r" \1", string)).split()


def _replace_values(name: Any, mapping: Dict[str, str]) -> Any:
    """
    Replace string values in the column name.

    Parameters
    ----------
    name
        Column name.
    mapping
        Maps old values in the column name to the new values.
    """
    if name in NULL_VALUES:
        return name

    name = str(name)
    for old_value, new_value in mapping.items():
        # If the old value or the new value is not alphanumeric, add underscores to the
        # beginning and end so the new value will be parsed correctly for _convert_case()
        new_val = (
            fr"{new_value}" if old_value.isalnum() and new_value.isalnum() else fr"_{new_value}_"
        )
        name = re.sub(fr"{old_value}", new_val, name, flags=re.IGNORECASE)

    return name


def _remove_accents(name: Any) -> Any:
    """
    Return the normal form for a Unicode string name using canonical
    decomposition.
    """
    if name in NULL_VALUES:
        return name

    return normalize("NFD", name).encode("ascii", "ignore").decode("ascii")


def _rename_duplicates(names: pd.Index, case: str) -> Any:
    """
    Rename duplicated column names to append a number at the end.
    """
    if case in {"snake", "camel", "pascal", "const"}:
        sep = "_"
    elif case == "kebab":
        sep = "-"
    else:
        sep = " "

    names = list(names)
    counts: Dict[str, int] = {}

    for i, col in enumerate(names):
        cur_count = counts.get(col, 0)
        if cur_count > 0 and names[i] not in NULL_VALUES:
            names[i] = f"{col}{sep}{cur_count}"
        counts[col] = cur_count + 1

    return names


def _create_report(stats: Dict[str, int], ncols: int) -> None:
    """
    Describe what was done in the cleaning process.
    """
    print("Headers Cleaning Report:")

    if stats["cleaned"] > 0:
        nclnd = stats["cleaned"]
        pclnd = round(nclnd / ncols * 100, 2)
        print(f"\t{nclnd} values cleaned ({pclnd}%)")

    nnull = stats["null"]
    pnull = round(nnull / ncols * 100, 2)
    ncorrect = ncols - nnull
    pcorrect = round(100 - pnull, 2)

    print(
        f"Result contains {ncorrect} ({pcorrect}%) values in the correct format "
        f"and {nnull} null values ({pnull}%)"
    )
