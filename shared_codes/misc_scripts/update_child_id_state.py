
import psycopg2
import psycopg2.extras as extras
import argparse

import sys; sys.path.append("../")
from utils.config_utils import connect_postgre

argparser = argparse.ArgumentParser()
argparser.add_argument("--state", help="Database name", required=True, type=str, choices=['start', 'stop'])
argparser.add_argument("--coin", help="Coin/asset name", type=str)
argparser.add_argument("--client_id", help="client_id in `clients` table", type=int)
argparser.add_argument("--parent_id", help="parent id as per `child_id_mapping` table", type=int, nargs='+')
argparser.add_argument("--child_id", help="child id as per `child_id_mapping` table", type=int, nargs='+')

args = argparser.parse_args()

"""
Update the state of the child_id in the `child_id_state` table:

POSSIBLE INPUTS WITH EACH INPUT TYPE:
    1. if coin:
        - state
        - coin
        - client_id : optional
    
    2. if client_id:
        - state
        - coin
        - parent_id : optional
    
    3. if parent_id:
        - state
        - client_id
    
    4. if child_id:
        - state

"""

##
conn, cur = connect_postgre()

## CHECK INPUTS
def check_inputs(args):
    if args.coin:
        assert not args.parent_id, "parent_id is not required with coin"
        assert not args.child_id, "child_id is not required with coin"
    
    elif args.client_id:
        assert not args.child_id, "child_id is not required with client_id"

check_inputs(args)
query_all = "select * from child_id_mapping;"
cur.execute(query_all)
all_mapping = cur.fetchall()

if args.coin:
    query = f"SELECT parent_trading_model from submodel_parameters where model_parameters->>'coin' = '{args.coin}'"
    cur.execute(query)
    coin_parent_ids = [x['parent_trading_model'] for x in cur.fetchall()]
    
    if not len(coin_parent_ids):
        raise ValueError(f"coin: {args.coin} not found in `submodel_parameters` table")
    
    all_mapping = [x for x in all_mapping if x['parent_trading_model'] in coin_parent_ids]
    
if args.client_id:
    
    ## Check if `client_id` exists in `all_mapping` list
    for x in all_mapping:
        if x['client_id'] == args.client_id:
            break
    
    else:
        raise ValueError(f"client_id: {args.client_id} not found in `child_id_mapping` table")
    
    ##
    all_mapping = [x for x in all_mapping if x['client_id'] == args.client_id]        

if args.parent_id:    
    ## Check if `parent_id` exists in `all_mapping` list
    for x in all_mapping:
        if x['parent_id'] in args.parent_id:
            break
    
    else:
        raise ValueError(f"parent_id: {args.parent_id} not found in `child_id_mapping` table")
    
    ##
    all_mapping = [x for x in all_mapping if x['parent_id'] in args.parent_id]

if args.child_id:
    ## Check if `child_id` exists in `all_mapping` list
    for x in all_mapping:
        if x['child_id'] in args.child_id:
            break
    
    else:
        raise ValueError(f"child_id: {args.child_id} not found in `child_id_mapping` table")
    
    ##
    all_mapping = [x for x in all_mapping if x['child_id'] in args.child_id]


## Upsert the state
# CREATE TABLE child_id_state(
#     child_trading_model bigint,
#     current_state status_domain NOT NULL,
#     reason text,
#     load_obj_db bool,
    # execution_state varchar(50),

for x in all_mapping:
    query = f"""
            INSERT INTO child_id_state (child_trading_model, current_state) 
            VALUES ({x['child_trading_model']}, '{args.state}') 
            ON CONFLICT (child_trading_model) 
                DO UPDATE SET current_state = '{args.state}';
            """
    
    ##
    cur.execute(query)

##
conn.commit()

conn.close()
print("Done!")

