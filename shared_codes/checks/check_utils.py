
import subprocess
from pathlib import Path
import os
import socket
import yaml
import time
import traceback
import zmq

from ..utils.config_utils import get_db_writer_socket

def db_writer_check(idenitity):
    try:
        socket, context = get_db_writer_socket(idenitity)
        
        ## Ping test
        socket.send_multipart([b'PING'])    

        ## Response
        response = socket.recv_multipart()
        ret_res = 1
    
    except Exception as e:
        tb_ = traceback.format_exc()
        print(f"Exception in db_writer_check: {e} \n {tb_}")
        ret_res = 0

    finally:
        socket.setsockopt(zmq.LINGER, 0)
        socket.close()
        context.term()
        return ret_res

def compare_dependencies(exported, current):
    return int(exported == current)

def compare_conda_envs(repo_env):
    # dump current env
    now_ts = str(time.time_ns())
    env_path = f'/tmp/base_env_{now_ts}.yml'
    subprocess.run(f"conda env export  | sed '/pythonclient/d' > {env_path}", shell=True, check=True)

    ######
    with open(env_path, 'r') as file:
        exported_env = yaml.safe_load(file)
    
    # current env
    with open(repo_env, 'r') as file:
        repo_env = yaml.safe_load(file)
    
    ##
    exporeted_pkgs = exported_env['dependencies']
    current_pkgs = repo_env['dependencies']
    
    return compare_dependencies(exporeted_pkgs, current_pkgs)


def check_addr_use(addr):
    ## Can be ipc connection as well
    if addr.startswith("ipc://"):
        return _check_ipc_usage(addr)

    port_ = int(addr.split(":")[-1])
    return _is_port_in_use(port_)

def _is_port_in_use(port):
    # Attempt to create a socket and bind to the given port
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind(("127.0.0.1", port))
        s.close()  # If successful, close the socket
        return 0 # Port is not in use
    except OSError:
        return 1  # Port is in use


def find_directory(target_dir, max_depth=3):
    root_dir = Path(os.getcwd())
    
    ## current dire check
    if target_dir in os.listdir(root_dir):
        return os.path.join(root_dir, target_dir)

    ## going backward
    for i in range(max_depth):
        current_dir = root_dir.parents[i]
        for x, dirs, files in os.walk(current_dir):
            if target_dir in dirs:
                return os.path.join(x, target_dir)

###
def _check_ipc_usage(socket_path):
    if socket_path.startswith("ipc://"):
        socket_path = socket_path.replace("ipc://", "")
        
    try:
        # Use lsof to check if any process is using the specified socket file
        output = subprocess.check_output(['lsof', socket_path], stderr=subprocess.STDOUT)
        if output:
            return 1
        else:
            return 0
    except subprocess.CalledProcessError as e:
        # If lsof returns a non-zero exit status, the socket is likely not in use or does not exist
        if e.returncode == 1:
            return 0
        else:
            raise e
    
    except Exception as e:
        raise e
