import streamlit as st
from st_aggrid import AgGrid, GridOptionsBuilder, JsCode, GridUpdateMode


data = [
    {
        "Header": "Level 1 - Item 1",
        "median_50perc_population": 123,
        "optimal_63perc_population": 234,
        "top_75perc_population": 345,
        "Stats_Model_Optimal_Budget": 456,
        "records": [
            {
                "order": 1,
                "CATEGORIES": "Level 2 - Category 1",
                "median_50perc_population": 111,
                "optimal_63perc_population": 222,
                "top_75perc_population": 333,
                "Stats_Model_Optimal_Budget": 444,
                "records": [
                    {
                        "order": 1,
                        "SUBCATEGORIES": "Level 3 - Subcategory 1",
                        "median_50perc_population": 101,
                        "optimal_63perc_population": 202,
                        "top_75perc_population": 303,
                        "Stats_Model_Optimal_Budget": 404,
                    },
                    {
                        "order": 2,
                        "SUBCATEGORIES": "Level 3 - Subcategory 2",
                        "median_50perc_population": 102,
                        "optimal_63perc_population": 203,
                        "top_75perc_population": 304,
                        "Stats_Model_Optimal_Budget": 405,
                    }
                ]
            },
            {
                "order": 2,
                "CATEGORIES": "Level 2 - Category 2",
                "median_50perc_population": 121,
                "optimal_63perc_population": 232,
                "top_75perc_population": 343,
                "Stats_Model_Optimal_Budget": 454,
                "records": [
                    {
                        "order": 1,
                        "SUBCATEGORIES": "Level 3 - Subcategory 1",
                        "median_50perc_population": 105,
                        "optimal_63perc_population": 206,
                        "top_75perc_population": 307,
                        "Stats_Model_Optimal_Budget": 408,
                    }
                ]
            }
        ]
    }
]

gridOptions = {                
    "masterDetail": True,
    "rowSelection": "disabled",
    "columnDefs": [
        {
            "field": "Header", "minWidth": 400,
            "cellRenderer": "agGroupCellRenderer",
            "checkboxSelection": False,
        },
        {"field": "median_50perc_population", "minWidth": 150},
        {"field": "optimal_63perc_population", "minWidth": 150},
        {"field": "top_75perc_population", "minWidth": 150},
        {"field": "Stats_Model_Optimal_Budget", "minWidth": 150}
    ],
    "defaultColDef": {
        "flex": 1,
    },
    "detailCellRendererParams": {
        "detailGridOptions": {
            "rowSelection": "disabled",
            "suppressRowClickSelection": True,
            "enableRangeSelection": True,
            "pagination": False,
            "paginationAutoPageSize": True,
            "columnDefs": [
                {"field": "order", "minWidth": 50},
                {"field": "CATEGORIES", "minWidth": 400},
                {"field": "median_50perc_population", "minWidth": 150},
                {"field": "optimal_63perc_population", "minWidth": 150},
                {"field": "top_75perc_population", "minWidth": 150},
                {"field": "Stats_Model_Optimal_Budget"}
            ],
            "defaultColDef": {
                "sortable": True,
                "flex": 1,
            },
            "masterDetail": True,
            "detailCellRendererParams": {
                "detailGridOptions": {
                    "rowSelection": "disabled",
                    "suppressRowClickSelection": True,
                    "enableRangeSelection": True,
                    "pagination": False,
                    "paginationAutoPageSize": True,
                    "columnDefs": [
                        {"field": "order", "minWidth": 50},
                        {"field": "SUBCATEGORIES", "minWidth": 400},
                        {"field": "median_50perc_population", "minWidth": 150},
                        {"field": "optimal_63perc_population", "minWidth": 150},
                        {"field": "top_75perc_population", "minWidth": 150},
                        {"field": "Stats_Model_Optimal_Budget"}
                    ],
                    "defaultColDef": {
                        "sortable": True,
                        "flex": 1,
                    },
                },
                "getDetailRowData": JsCode(
                    """function (params) {
                        params.successCallback(params.data.records);
                    }"""
                ),
            }
        },
        "getDetailRowData": JsCode(
            """function (params) {
                params.successCallback(params.data.records);
            }"""
        ),
    },
    "rowData": data
}

r = AgGrid(None, height=250, gridOptions=gridOptions, allow_unsafe_jscode=True, enable_enterprise_modules=True, udate_mode=GridUpdateMode.SELECTION_CHANGED, fit_columns_on_grid_load=True, theme='balham')
