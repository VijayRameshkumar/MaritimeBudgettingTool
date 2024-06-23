import snowflake.connector
import config
import pandas as pd

def get_data(query, database=None, schema=None):
    # Establish the connection
    conn = snowflake.connector.connect(
        user=config.user,
        password= config.password,
        account= config.account,
        warehouse= config.warehouse,
        database= database if database else config.database,
        schema= schema if schema else config.schema
    )

    # Create a cursor object
    cursor = conn.cursor()

    cursor.execute(query)

    # Fetch and print the result
    result = cursor.fetchall()

    # Close the cursor and connection
    cursor.close()
    conn.close()
    return result


def get_expense_data():
    unique_cc = pd.read_excel(r'C:\Users\vijay.r\OneDrive - Synergy Maritime Private Limited\ML Projects\06_Vessel Accounts - Director\Budget Plan\VESSEL_PARTICULARS.xlsx',
                              engine='openpyxl')['VESSEL CODE'].unique().tolist()
    unique_cc_str = ', '.join([f"'{cc}'" for cc in unique_cc])

    expense_query = f'''SELECT
        TO_VARIANT(DATE_TRUNC('MONTH', MS.POSTING_DATE))::VARCHAR AS "Year-Month",
        MS.COST_CENTER,
        MS.NODECODE AS "Categories",
        MS.ACCOUNT_CODE,
        MS.ACCOUNT_CODE_DESC AS "SUB_CATEGORIES",
        SUM(MS.AMOUNT_USD) AS "Expense"
        FROM
        CURATED_DB.VESSEL_ACCOUNTS.MANAGER_STATEMENT MS
        WHERE
        MS.BRANCHCODE = 'Operating Expenses' AND MS.COST_CENTER IN ({unique_cc_str}) and MS.PERIOD >= 202101 and MS.PERIOD <= 202312
        GROUP BY "Year-Month", MS.COST_CENTER, MS.NODECODE, MS.ACCOUNT_CODE, MS.ACCOUNT_CODE_DESC
        ORDER BY "Year-Month", MS.COST_CENTER, MS.NODECODE ASC'''

    expense_data = pd.DataFrame(get_data(expense_query), columns=['DATE', 'COST_CENTER', 'CATEGORIES', "ACCOUNT_CODE", 'SUB_CATEGORIES', 'Expense'])
    expense_data['PERIOD'] = pd.to_datetime(expense_data['DATE']).dt.strftime('%Y%m')
    
    return expense_data