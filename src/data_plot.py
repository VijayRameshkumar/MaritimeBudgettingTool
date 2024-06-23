import pandas as pd
import streamlit as st
from st_aggrid import AgGrid, GridOptionsBuilder

simple_data = pd.DataFrame(
    {
        "name": ["a", "b"],
        "history": [
            [1, 2, 0, 0, 1, 0, 2],
            [1, 0, 0, 0, 2, 2, 2, 2],
        ],
    }
)

gb = GridOptionsBuilder.from_dataframe(simple_data)
gb.configure_side_bar()
gb.configure_column(
    "history",
    cellRenderer="agSparklineCellRenderer",
    cellRendererParams={
        "sparklineOptions": {
            "type": "line",
            "line": {"stroke": "#91cc75", "strokeWidth": 2},
        }
    },
)
gridOptions = gb.build()

g = AgGrid(
    simple_data,
    gridOptions=gridOptions,
)