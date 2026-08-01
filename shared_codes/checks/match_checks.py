


def get_common_checks():
    
    return {
        "git_check": 1, 
        "tz_check": "UTC", 
        "conda_dependencies_check": 1, 
        "db_writer_ping": 1,
        "logs_folder_check": 1,
        "soft_fd_check": 1
    }
    
def get_signal_broker_checks(input_dict):
    
    common_checks = get_common_checks()
    broker_checks = {
        "signal_broker_port_inuse": 0,
        "os_env_var_check": ['signal_broker_som', "signal_broker_mangal"], 
        "conda_env_name": "py_broker"
    }

    all_checks = {**common_checks, **broker_checks}
    
    ## commpare input_dict with all_checks
    for key, value in all_checks.items():
        if key not in input_dict:
            raise ValueError(f"Missing key: {key}")

        if isinstance(value, list):
            assert input_dict[key] in value, f"Error: {key} is not in {value}"
    
        else:
            assert input_dict[key] == value, f"Error: {key} is not equal to {value}"



def compare_dict(input_dict, all_checks):
    for key, value in all_checks.items():
        if key not in input_dict:
            raise ValueError(f"Missing key: {key}")

        if isinstance(value, list):
            assert input_dict[key] in value, f"Error: {key} is not in {value}. Input dict: {input_dict[key]}"
    
        else:
            assert input_dict[key] == value, f"Error: {key} is not equal to {value}. Input dict: {input_dict[key]}"
        
        

def signal_gen_checks(input_dict):
    common_checks = get_common_checks()
    signal_gen_checks = {
        "os_env_var_check": ['signal_gen_som', "signal_gen_mangal"], 
        "conda_env_name": "py_signal_gen",
        "data_check": 1,
        "broker_check": 1,
        "logs_folder_check": 1,
        "data_folder_check": 1,
        "pbsl_folder_check": 1
    }

    all_checks = {**common_checks, **signal_gen_checks}
    compare_dict(input_dict, all_checks)


def zmq_server_checks(input_dict):
    
    common_checks = get_common_checks()
    zmq_server_checks = {
        "os_env_var_check": ['magha_som_execution', 'phalguna_mangal_execution', 'chaitra_som_execution', 'vaisakha_som_execution'], 
        "conda_env_name": ["py_exec_no_gil", "py_exec"],
        "zmq_server_port_inuse": 0,
        "redis_running": 1,
        "dbn_shm_main_status": 1
    }
    
    all_checks = {**common_checks, **zmq_server_checks}
    compare_dict(input_dict, all_checks)
    

def binance_ws_checks(input_dict):
    
    common_checks = get_common_checks()
    zmq_server_checks = {
        "os_env_var_check": ['som_data', 'mangal_data'], 
        "conda_env_name": "py_exec",
        "binance_port_inuse": 0
    }
    
    all_checks = {**common_checks, **zmq_server_checks}
    compare_dict(input_dict, all_checks)

def dbn_live_shm_checks(input_dict):
    
    common_checks = get_common_checks()
    dbn_live_shm_checks = {
        "os_env_var_check": ['som_data', 'mangal_data'], 
        "conda_env_name": "py_exec",
        "dbn_live_shm_status": 0,
        "dbn_shm_main_status": 1
    }
    
    all_checks = {**common_checks, **dbn_live_shm_checks}
    compare_dict(input_dict, all_checks)

def shm_main_checks(input_dict):
    
    common_checks = get_common_checks()
    shm_main_checks = {
        "os_env_var_check": [('magha_som_execution', 'magha_som_data', 'magha_som_execution'),
                            ('phalguna_mangal_server', 'mangal_data', 'phalguna_mangal_execution'), 
                            ('vaisakha_som_execution', 'vaisakha_som_data', 'vaisakha_som_execution'),
                            ('chaitra_som_execution', 'chaitra_som_data', 'chaitra_som_execution')
                            ],
        "conda_env_name": "py_exec",
        "shm_main_status": 0
    }
    
    
    all_checks = {**common_checks, **shm_main_checks}
    compare_dict(input_dict, all_checks)


##
def worker_socket_checks(input_dict):
    
    common_checks = get_common_checks()
    worker_socket_checks = {
        "os_env_var_check": ['magha_som_execution', 'phalguna_mangal_execution', 'chaitra_som_execution', 'vaisakha_som_execution'], 
        "conda_env_name": "py_exec",
        "zmq_exec_server_ping": 1,
        "dbn_shm_main_status": 1
    }
    
    all_checks = {**common_checks, **worker_socket_checks}
    compare_dict(input_dict, all_checks)


def main_socket_checks(input_dict):
    
    common_checks = get_common_checks()
    main_socket_checks = {
        "os_env_var_check": ['magha_som_execution', 'phalguna_mangal_execution', 'vaisakha_som_execution', 'chaitra_som_execution'], 
        "conda_env_name": "py_exec",
        "ipc_trades_forwarder": 0,
        "ipc_data_pubsub": 0,
        "ipc_request_reply": 0,
        "ipc_push_pull": 0,
        "data_server_ping": 1,
        "signal_broker_ping": 1
    }
    
    all_checks = {**common_checks, **main_socket_checks}
    compare_dict(input_dict, all_checks)



def db_writer_checks(input_dict):
    
    common_checks = get_common_checks()
    db_writer_checks = {
        "os_env_var_check": ["magha_db_writer", 'vaisakha_db_writer', 'chaitra_db_writer', "phalguna_db_writer"], 
        "conda_env_name": "py_db_writer",
        "db_writer_port_inuse": 0
    }
    
    all_checks = {**common_checks, **db_writer_checks}
    compare_dict(input_dict, all_checks)


def system_watch_checks(input_dict):
    
    common_checks = get_common_checks()
    system_watch_checks = { 
        "conda_env_name": "py_system_watch",
        "json_folder_check": 1
    }
    
    all_checks = {**common_checks, **system_watch_checks}
    compare_dict(input_dict, all_checks)