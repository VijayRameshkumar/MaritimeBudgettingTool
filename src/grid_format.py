from st_aggrid import AgGrid, GridOptionsBuilder, JsCode, GridUpdateMode

gridOptions = {
    
    # Enable Master / Detail
    "masterDetail": True,
    "rowSelection": "single",
    # Configure column definitions
    "columnDefs": [
                    {
                        "field": "Header",
                        "cellRenderer": "agGroupCellRenderer",
                        "checkboxSelection": True,
                    },
                    {"field": "median_50perc_population"},
                    {"field": "optimal_63perc_population"},
                    {"field": "top_75perc_population"}
                ],
    "defaultColDef": {
        "flex": 1,
    },
    # Provide Detail Cell Renderer Params
    "detailCellRendererParams": {
        # Provide the Grid Options to use on the Detail Grid
        "detailGridOptions": {
            "rowSelection": "multiple",
            "suppressRowClickSelection": True,
            "enableRangeSelection": True,
            "pagination": True,
            "paginationAutoPageSize": True,
            "columnDefs": [
                            {"field": "CATEGORIES"},
                            {"field": "median_50perc_population"},
                            {"field": "optimal_63perc_population"},
                            {"field": "top_75perc_population"}
                        ],
            "defaultColDef": {
                "sortable": True,
                "flex": 1,
            },
        },
        # Function to get the rows for each Detail Grid
        "getDetailRowData": JsCode(
            """function (params) {
                params.successCallback(params.data.records);
            }"""
        ),
    },
    # Provide the row data
    "rowData": data
}
