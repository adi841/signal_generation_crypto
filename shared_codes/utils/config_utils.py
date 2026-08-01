import logging
from logging.handlers import TimedRotatingFileHandler
import os
import sys
import datetime as dt
import threading
import subprocess
import boto3
import psycopg2
import psycopg2.extras as psycopg2_e
from collections import namedtuple
import hjson
import functools
from pathlib import Path
import zmq
import zmq.asyncio
import pickle
import time
import traceback
import atexit


from .dataclass_utils import MsgType, DumpErrorLog, DumpStrategyObject
 
 
# Create a global ZeroMQ context
CONTEXT = zmq.Context()
CONTEXT_ASYNC = zmq.asyncio.Context()

atexit.register(CONTEXT.term)
atexit.register(CONTEXT_ASYNC.term)

@functools.lru_cache(maxsize=20)
def get_current_branch_env(path=None):
    # `path` lets callers resolve the branch of a specific repo/dir; defaults to cwd.
    cmd = ["git"]
    if path is not None:
        cmd += ["-C", str(path)]
    cmd += ["rev-parse", "--abbrev-ref", "HEAD"]

    try:
        # Run the git command to get the current branch name
        branch_name = subprocess.check_output(cmd, stderr=subprocess.STDOUT).strip().decode('utf-8')

        ##
        branch_name_0 = branch_name.split("/")[0]
        branch_name_1 = branch_name.split("/")[1] if len(branch_name.split("/")) > 1 else None

        branch_set = {"main", "uat", "dev"}
        if branch_name_0 in branch_set:
            return branch_name_0
        
        elif not branch_name_1:
            raise ValueError(f"Branch name: {branch_name} not recognized")
        
        else:
            return branch_name_1
            
    except subprocess.CalledProcessError as e:
        # Handle cases where the current directory is not a git repository
        print(f"Error: {e.output.decode('utf-8')}") 
        return None

@functools.lru_cache(maxsize=20)
def get_s3_config(file_type, run_destination=None, use_branch=True):
    
    assert file_type in ['main_config', "binance_config", "cr_symbol_mapping", "global_settings", "som_config", "phalguna_mangal_config", "mangal_config", "tradefi_symbol_info", \
                    "rollover_info", "shm_tradefi_symbols_indx", "global_variables", "vaisakha_som_execution_params", "chaitra_som_execution_params", "magha_som_execution_params", \
                    "chaitra_som_config", "vaisakha_som_config", "magha_som_config", "kaiko", "coin_api"], f"File type not recognized: {file_type}"

    #
    s3_client = boto3.client('s3', aws_access_key_id=os.environ['AWS_ACCESS'], aws_secret_access_key=os.environ['AWS_SECRET'])

    #
    bucket_name = "suman-crypto-files"    
    mode = None
    
    current_branch_env = get_current_branch_env()   

    suffix = "hjson"
    if file_type == 'main_config':
        if run_destination == 'magha_som_execution':
            file_name = "magha_som_config"
        
        elif run_destination == 'phalguna_mangal_execution':
            file_name = "phalguna_mangal_config"
        
        elif run_destination == 'chaitra_som_execution':
            file_name = "chaitra_som_config"
        
        elif run_destination == 'vaisakha_som_execution':
            file_name = "vaisakha_som_config"
        
        else:
            raise Exception("Run destination not recognized")
    
    elif file_type == 'phalguna_mangal_config':
        file_name = "phalguna_mangal_config"
    
    elif file_type == "chaitra_som_config":
        file_name = "chaitra_som_config"
    
    elif file_type == "vaisakha_som_config":
        file_name = "vaisakha_som_config"

    elif file_type == "magha_som_config":
        file_name = "magha_som_config"

    elif file_type == 'cr_symbol_mapping':
        file_name = "cr_symbol_mapping"

    elif file_type == 'binance_config':
        file_name = "binance"
    
    elif file_type == 'global_settings':
        file_name = "global_settings"

    elif file_type == 'som_config':
        # Asset class wise
        file_name = "som_config"
    
    elif file_type == 'mangal_config':
        # Asset class wise
        file_name = "mangal_config"
    
    elif file_type == 'tradefi_symbol_info':
        file_name = "tradefi_symbol_info"
    
    elif file_type == "rollover_info":
        file_name = "rollover_info.csv"
    
    elif file_type == "shm_tradefi_symbols_indx":
        file_name = "shm_tradefi_symbols_indx"
        suffix = "csv"
    
    elif file_type == "global_variables":
        file_name = "global_variables"
    
    elif file_type == "vaisakha_som_execution_params":
        file_name = "vaisakha_som_execution_params"

    elif file_type == "chaitra_som_execution_params":
        file_name = "chaitra_som_execution_params"

    elif file_type == "magha_som_execution_params":
        file_name = "magha_som_execution_params"
    
    elif file_type == "kaiko":
        file_name = "kaiko"

    elif file_type == 'coin_api':
        file_name = "coin_api"

    else:
        raise Exception("File type not recognized")
    
    ##
    mode = None
    if use_branch:
        assert current_branch_env in ['dev', 'main', "uat"], f"Branch name: {current_branch_env} not recognized"

        object_key = f"{file_name}.{current_branch_env}.{suffix}"
        mode = current_branch_env
        print("File type: %s, Mode: %s" %(file_name, mode), object_key)

        s3_object = s3_client.get_object(Bucket=bucket_name, Key=object_key)
    
    else:
        print("File type: %s" %(file_name))

        s3_object = s3_client.get_object(Bucket=bucket_name, Key=file_name)

    ##
    data = s3_object['Body'].read()

    #
    return data, mode


##
class ResilientTimedRotatingFileHandler(TimedRotatingFileHandler):
    """
    A TimedRotatingFileHandler that recreates the log file if it's deleted externally.
    
    This handler uses time-based checking to minimize overhead while detecting deleted
    files. Only checks file existence after a specified time interval has elapsed.
    """
    
    def __init__(self, *args, check_interval_seconds=60, **kwargs):
        """
        Initialize the handler.
        
        Parameters
        ----------
        check_interval_seconds : int or float, optional
            Check file existence only after this many seconds have elapsed since last check.
            Default is 60 seconds. Lower values = faster detection but slightly more overhead.
            Set to 0 to disable time-based checking (always check).
        """
        super().__init__(*args, **kwargs)
        self._check_interval_seconds = check_interval_seconds
        self._last_check_time = time.time()
    
    def emit(self, record):
        """
        Emit a record, periodically checking if the log file was deleted.
        
        Checks file existence only after check_interval_seconds have passed since last check.
        This provides excellent performance with minimal overhead (just a time comparison).
        """
        try:
            # Check if enough time has passed since last check
            current_time = time.time()
            if current_time - self._last_check_time >= self._check_interval_seconds:
                self._last_check_time = current_time
                # Time-based check: verify file still exists
                if self.stream and hasattr(self.stream, 'name'):
                    # Use stat on the file descriptor (faster than path checking)
                    try:
                        fd_stat = os.fstat(self.stream.fileno())
                        # Check if the file path still links to the same inode
                        try:
                            path_stat = os.stat(self.stream.name)
                            # If inode numbers don't match, file was deleted and recreated
                            if fd_stat.st_ino != path_stat.st_ino:
                                self.stream.close()
                                self.stream = self._open()
                        except FileNotFoundError:
                            # File path doesn't exist anymore
                            self.stream.close()
                            self.stream = self._open()
                    except (OSError, ValueError):
                        # File descriptor invalid
                        try:
                            self.stream.close()
                        except:
                            pass
                        self.stream = self._open()
            
            # Perform the actual write
            super().emit(record)
        except Exception:
            self.handleError(record)


class ThreadedErrorHandler(logging.Handler):
    def __init__(self, error_origin, run_destination):
        super().__init__()
        self.error_origin = error_origin
        self.source = run_destination

        # Define the formatter inside the handler
        self.formatter = logging.Formatter(
            "%(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

    def emit(self, record):
        # Use the formatter to format the record
        formatted_message = self.formatter.format(record)

        loglevel_name = record.levelname.lower()
        if loglevel_name in ['warning', 'error', 'critical']:
            t1 = threading.Thread(target=dump_error_log, args=(self.error_origin, loglevel_name, formatted_message, self.source))
            t1.start()


def setup_logger(filename, error_origin, logger_name="log_testing", log_config={}, run_destination=None):
    """
    Setup a logger with specified filename, name, and level.

    This function sets up a logger with a file handler and a stream handler, both using the same formatter. 
    The file handler writes the logs to the specified file, while the stream handler writes the logs to the 
    standard output.

    Parameters
    ----------
    filename : str
        The name of the file to write logs to.
    logger_name : str, optional
        The name of the logger. Default is 'log_testing'.
    level : int, optional
        The level of the logger. Default is logging.INFO.

    Returns
    -------
    logging.Logger
        The setup logger object.

    Examples
    --------
    >>> logger = setup_logger('test.log', 'test_logger', self.logging_config)
    >>> logger.info('This is a test log message.')
    This will write 'This is a test log message.' to 'test.log' and print it to the standard output.
    """    
    
    assert run_destination is not None, "Run destination not provided"

    logger = logging.getLogger(logger_name)
    logger.setLevel(log_config['log_level'])

    # formatter = logging.Formatter("%(asctime)s: %(message)s", datefmt="%Y-%m-%d %I:%M:%S %p")
    # formatter = logging.Formatter("%(asctime)s, %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s", datefmt="%Y-%m-%d %I:%M:%S %p")
    formatter = logging.Formatter("%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    ## Settting File handler formatter to create log file
    # file_handler = logging.FileHandler(filename, mode="w")

    # file_handler = TimedRotatingFileHandler(filename, when=log_config['when_roll'], backupCount=log_config['backup_count'], utc=True)
    file_handler = ResilientTimedRotatingFileHandler(filename, when=log_config['when_roll'], backupCount=log_config['backup_count'], utc=True)
    file_handler.setLevel(log_config['file_level'])
    file_handler.setFormatter(formatter)

    ## Settting Stream handler formatter to print logs
    stream_handler = logging.StreamHandler(sys.stdout) # sys.stdout
    stream_handler.setLevel(log_config['print_level'])
    stream_handler.setFormatter(formatter)

    ##
    threaded_error_handler = ThreadedErrorHandler(error_origin=error_origin, run_destination=run_destination)

    
    ## Adding handler to logger
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.addHandler(threaded_error_handler)

    return logger

@functools.lru_cache(maxsize=2)
def get_logging_config():
    conf_ = get_s3_config("global_settings")[0]
    global_conf = hjson.loads(conf_.decode('utf-8'))
    return global_conf['logging_config']

# class OldmodelLogger(object):
#     """
#     A logger class for different models.

#     This class provides methods for creating and getting loggers for different models. Each model gets its 
#     own log file.

#     Parameters
#     ----------
#     log_file_path : str
#         The base path where the log files for each model will be saved.
#     """
#     def __init__(self, run_destination):
#         """
#         Initialize the modelLogger object.

#         Parameters
#         ----------
#         log_file_path : str
#             The base path where the log files for each model will be saved.
#         """
#         self.logging_config = get_logging_config()
#         self.error_origin_logging_dict = {}
#         self.run_destination = run_destination

#     def create_logger(self, error_origin):
#         """
#         Create a logger for a specific error_origin.

#         Parameters
#         ----------
#         error_origin : str
#             The name of the error_origin to create a logger for.

#         Returns
#         -------
#         logging.Logger
#             The logger object for the specified error_origin.

#         Examples
#         --------
#         >>> logger = modelLogger('logs')
#         >>> model_logger = logger.create_logger('my_model')
#         This will create a logger for 'my_model' and save the logs to 'logs/my_model.log'.
#         """
#         path_ = os.path.join(self.logging_config['logfile_path'], f"{error_origin}.log")
#         return setup_logger(path_, error_origin=error_origin, logger_name=f"{error_origin}_logger", log_config=self.logging_config, run_destination=self.run_destination)
    
#     def get_logger(self, error_origin, level: str):
#         """
#         Get the logger for a specific error_origin.

#         If a logger for the error_origin does not exist yet, it is created.

#         Parameters
#         ----------
#         error_origin : str
#             The name of the error_origin to get the logger for.
#         level : str, optional
#             The level of the logger. Default is 'info'. Possible values are 'debug', 'info', 'warning', 'error', and 'critical'.

#         Returns
#         -------
#         method
#             The logging method corresponding to the specified level for the error_origin's logger.

#         Examples
#         --------
#         >>> logger = modelLogger('logs')
#         >>> log_method = logger.get_logger('my_model', 'debug')
#         This will return the 'debug' method of the logger for 'my_model'.
#         """
#         if error_origin not in self.error_origin_logging_dict:
#             self.error_origin_logging_dict[error_origin] = self.create_logger(error_origin)
        
#         #
#         return getattr(self.error_origin_logging_dict[error_origin], level)



class modelLogger:
    def __init__(self, run_destination, idle_timeout=60, cleanup_interval=60):
        """
        Initialize CentralizedLoggerManager with an idle timeout and a centralized cleanup mechanism.

        Parameters
        ----------
        run_destination : str
            The path or destination where logs are directed.
        idle_timeout : int, optional
            Time in seconds after which unused loggers are closed. Default is 300 seconds.
        cleanup_interval : int, optional
            Interval in seconds at which the cleanup thread checks for idle loggers. Default is 60 seconds.
        """
        self.logging_config = get_logging_config()
        self.run_destination = run_destination
        self.idle_timeout = idle_timeout
        self.cleanup_interval = cleanup_interval
        self.next_cleanup = time.time() + self.cleanup_interval
        self.loggers = {}
        self.last_used = {}
        self.cleanup_logger = self.create_logger("logger_cleanup_task")

    def _cleanup_task(self):
        """
        Periodically check and close idle loggers.

        This function runs in a separate thread and wakes up at intervals defined
        by `cleanup_interval` to check the last usage time of each logger. If a logger
        has been idle longer than `idle_timeout`, it is closed and removed from the manager.
        """
        current_time = time.time()
        try:
            for error_origin, last_time in list(self.last_used.items()):
                if current_time - last_time > self.idle_timeout:
                    self._close_logger(error_origin)
        except Exception as e:
            tb_ = traceback.format_exc()
            msg_ = f"Error in _cleanup_task: {e}: {tb_}"
            self.cleanup_logger.critical(msg_)
    
    def close_all_loggers(self):
        """
        Close all loggers.
        """
        for error_origin in list(self.loggers.keys()):
            self._close_logger(error_origin)

            
    def _close_logger(self, error_origin):
        """
        Close and remove the specified logger.

        Parameters
        ----------
        error_origin : str
            The identifier of the logger to be closed.

        This method removes all handlers from the logger, closes them, and removes the
        logger from internal tracking to free up resources.
        """
        try:
            ##
            if error_origin in self.loggers:
                msg_ = f"Closing logger for {error_origin}"
                self.cleanup_logger.info(msg_)
                logger = self.loggers.pop(error_origin)
                for handler in logger.handlers[:]:
                    handler.close()
                    logger.removeHandler(handler)
                self.last_used.pop(error_origin, None)
        
        except Exception as e:
            tb_ = traceback.format_exc()
            msg_ = f"Error in _close_logger: {e}: {tb_}"
            self.cleanup_logger.critical(msg_)

    def create_logger(self, error_origin):
        """
        Create and configure a new logger for the specified error origin.

        Parameters
        ----------
        error_origin : str
            The identifier for the new logger.

        Returns
        -------
        logging.Logger
            A configured logger object for the specified error origin.

        This method sets up the logger with file and stream handlers based on the configuration
        settings and attaches it to the specified error origin.
        """
        if time.time() > self.next_cleanup:
            self._cleanup_task()
            self.next_cleanup = time.time() + self.cleanup_interval

        path_ = os.path.join(self.logging_config['logfile_path'], f"{error_origin}.log")
        logger = setup_logger(path_, error_origin=error_origin, logger_name=f"{error_origin}_logger", log_config=self.logging_config, run_destination=self.run_destination)
        return logger

    def get_logger(self, error_origin, level: str):
        """
        Retrieve or create a logger and reset its idle timer.

        Parameters
        ----------
        error_origin : str
            The identifier of the logger to be retrieved or created.
        level : str
            The logging level method to be accessed (e.g., 'debug', 'info', 'warning', 'error', 'critical').

        Returns
        -------
        method
            The specified logging method corresponding to the logger's level.

        This method checks if the logger exists, creates it if not, updates its last usage time,
        and then returns the specified logging level method.
        """
        self.last_used[error_origin] = time.time()
        if error_origin not in self.loggers:
            self.loggers[error_origin] = self.create_logger(error_origin)
        
        # Retrieve the desired logging method
        return getattr(self.loggers[error_origin], level)



def dump_error_log(error_origin, level, error_ms, source):
    
    curr_dt = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S:%f")
    identity = f"{source}_{error_origin}_{curr_dt}"
    
    socket, context = get_db_writer_socket(identity)

    error_msg = DumpErrorLog(log_level=level, source=source, error_origin=error_origin, error=error_ms)
    msg_type = MsgType.DUMP_ERROR_LOG
    
    pkl_error_msg = pickle.dumps(error_msg)
    pkl_msg_type = pickle.dumps(msg_type)
    
    try:
        socket.send_multipart([pkl_msg_type, pkl_error_msg])
        socket.recv()
    
    except zmq.error.Again:
        pass
        # write_error_log(error_msg)
        # error_msg_zmq = DumpErrorLog(log_level='error', source=source, error_origin=error_origin, error="Error in sending error log to db_writer")
        # write_error_log(error_msg_zmq)
        
    except Exception as e:
        pass
    
    finally:
        socket.setsockopt(zmq.LINGER, 0)
        socket.close()
        context.term()
    

async def dump_strategy_object(config_object, strat_obj: DumpStrategyObject):
    
    # config_object to dump error log
    ##
    context = zmq.asyncio.Context()
    identity = str(time.time_ns())

    socket = context.socket(zmq.REQ)
    socket.setsockopt(zmq.IDENTITY, identity.encode("utf-8"))
    socket.setsockopt(zmq.RCVTIMEO, 2*1000)
    socket.setsockopt(zmq.SNDTIMEO, 2*1000)

    socket.connect(get_dump_error_host())
    msg_type = MsgType.DUMP_STRATEGY_OBJECT
    
    ##
    pkl_error_msg = pickle.dumps(strat_obj)
    pkl_msg_type = pickle.dumps(msg_type)
    
    try:
        await socket.send_multipart([pkl_msg_type, pkl_error_msg])
        await socket.recv()
    
    except zmq.error.Again:
        msg_ = "Error in sending strategy object to db_writer"
        config_object.masterlog.get_logger("dump_strat_object_"+(str(strat_obj.child_trading_model)), "error")(msg_)
    
    except Exception as e:
        tb_ = traceback.format_exc()
        msg_ = f"Error in dump_strategy_object: {e}: {tb_}"
        config_object.masterlog.get_logger("dump_strat_object_"+(str(strat_obj.child_trading_model)), "error")(msg_)
    
    finally:
        socket.setsockopt(zmq.LINGER, 0)
        socket.close()
        context.term()

def write_error_log(error_msg: DumpErrorLog):
    
    ##
    query_insert = """
            INSERT INTO log_monitoring (timestamp, log_level, source, error_origin, error) 
            VALUES (%s, %s, %s, %s, %s);
            """
    
    current_timestamp = dt.datetime.utcnow()
    try:
        ##
        conn, cursor = connect_postgre(db_user="error_writer", direct_conn=True)
        cursor.execute(query_insert, (current_timestamp, error_msg.log_level, error_msg.source, error_msg.error_origin, error_msg.error))
        conn.commit()
            
    except Exception as e:
        pass
    
    finally:
        conn.close()
        cursor.close()


@functools.lru_cache(maxsize=2)
def get_dump_error_host():
    conf_ = get_s3_config("global_settings")[0]
    global_conf = hjson.loads(conf_.decode('utf-8'))
    client_ = os.environ["PG_CLIENT"]
    return global_conf['db_cluster'][client_]['db_cluster_host']


def get_db_writer_socket(identity):

    ##   
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.setsockopt(zmq.IDENTITY, identity.encode("utf-8"))
    socket.setsockopt(zmq.RCVTIMEO, 2*1000)
    socket.setsockopt(zmq.SNDTIMEO, 2*1000)

    socket.connect(get_dump_error_host())
    return socket, context

@functools.lru_cache(maxsize=2)
def get_db_params():
    tuple_ = namedtuple('db_params', ['host', 'database', 'user', 'password', 'port', 'min_conn', "max_conn", "max_inactive_connection_lifetime", "max_queries"])
    password = os.environ['POSTGRES_PASSWORD']

    conf_ = get_s3_config("global_settings")[0]
    global_conf = hjson.loads(conf_.decode('utf-8'))

    ##
    client = os.environ["PG_CLIENT"]
    db_host = global_conf['db_config'][client]['host']

    db_conf = global_conf['db_config']
    db_params = tuple_(db_host, db_conf['database'], db_conf['user'], password, db_conf['port'], db_conf['min_conn'], db_conf['max_conn'], db_conf['max_inactive_connection_lifetime'], db_conf['max_queries'])
    return db_params

@functools.lru_cache(maxsize=2)
def get_global_settings():
    conf_ = get_s3_config("global_settings")[0]
    global_conf = hjson.loads(conf_.decode('utf-8'))
    return global_conf


##
def connect_postgre(db_user, direct_conn=False, database=None):
    """
    Connect to the PostgreSQL database.
    
    Args:
    -----
    direct_conn: bool
        If True, then connect to the database directly. Otherwise, connect using the connection pool.
    """
    db_user_ls = get_db_users()
    db_params = get_db_params()
    port = 5432 if direct_conn else db_params.port
    
    if db_user not in db_user_ls:
        msg_ = f"User {db_user} not recognized in the database: {str(db_user_ls)}"
        error_msg_zmq = DumpErrorLog(log_level='error', source="config_utils", error_origin="config_utils", error=msg_)
        write_error_log(error_msg_zmq)
        return None, None
    
    # conn = psycopg2.connect(user=config_object.user, password=config_object.password, host=config_object.host, port=config_object.port, database=config_object.database)
    if database is None:
        database = db_params.database
    conn = psycopg2.connect(user=db_user, password=db_params.password, host=db_params.host, port=port, database=database)
    cursor = conn.cursor(cursor_factory=psycopg2_e.RealDictCursor)
    
    return conn, cursor


###
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
    

def get_clientID_info(client_id, retry_count=0):
    """
    Get client_id info from the `client_id_info` table
    
    Returns:
    --------
    dict
        client_id, client_name, client_type, client_email
    """
    
    try:
        if retry_count == 10:
            return {}
            
        conn, cursor = connect_postgre(db_user='tsdbadmin')
        
        query_ = f""" SELECT * FROM clients WHERE client_id = '{client_id}'; """
        cursor.execute(query_)
        data = cursor.fetchall()
        
        #
        conn.close()
        
        return data[0]
    
    except Exception as e:
        print(f"Error in get_clientID_info: {e}")
        import time; time.sleep(5) 
        return get_clientID_info(client_id, retry_count+1)


@functools.lru_cache(maxsize=2)
def get_db_users():
    """
    Get the list of users from the `users` table
    
    Returns:
    --------
    list
        List of users
    """
    db_params = get_db_params()
    
    conn = psycopg2.connect(user=db_params.user, password=db_params.password, host=db_params.host, port=5432, database=db_params.database)
    cursor = conn.cursor(cursor_factory=psycopg2_e.RealDictCursor)
    
    ##
    query_ = f""" SELECT rolname FROM pg_roles WHERE rolsuper; """
    cursor.execute(query_)
    data = cursor.fetchall()
    
    #
    conn.close()
    
    return [x['rolname'] for x in data]


@functools.lru_cache(maxsize=10)
def check_sync_with_origin():
    
    '''
    Check if the current branch is in sync with origin head.    
    '''
    def check_git_sync(path="."):
        # Fetch the latest information from origin
        subprocess.run(['git', 'fetch', 'origin'], cwd=path, check=True)
        
        # Get the current branch name
        branch_name = subprocess.getoutput(f'git -C {path} rev-parse --abbrev-ref HEAD')
        
        # List commits on local branch not in remote branch
        remote_not_in_local = subprocess.getoutput(f'git -C {path} rev-list {branch_name}..origin/{branch_name}')
        
        # List commits on remote branch not in local branch
        local_not_in_remote = subprocess.getoutput(f'git -C {path} rev-list origin/{branch_name}..{branch_name}')
        
        # Count the number of commit hashes in each list
        local_commits_count = len(local_not_in_remote.splitlines()) if local_not_in_remote else 0
        remote_commits_count = len(remote_not_in_local.splitlines()) if remote_not_in_local else 0    

        # Check if local branch is behind remote branch
        if remote_commits_count > 0:
            raise ValueError(f"The branch '{branch_name}' in {path} is not in sync with 'origin/{branch_name}'. Local branch is behind by {remote_commits_count} commits.")
        
        # if local_commits_count > 0:
        #     raise ValueError(f"The branch '{branch_name}' in {path} has local changes not pushed to 'origin/{branch_name}'. It is ahead by {local_commits_count} commits.")
    
        return branch_name

    curr_branch = check_git_sync(os.getcwd())
    curr_branch_env = get_current_branch_env(os.getcwd())
    shared_code_path = find_directory("shared_codes")

    shared_codes_branch_env = get_current_branch_env(shared_code_path)

    assert curr_branch_env == shared_codes_branch_env, "Current branch and shared_codes branch are not same environment."
    return 1

##
@functools.lru_cache(maxsize=10)
def check_db_host():
    ##
    client = os.environ["PG_CLIENT"]

    conf_ = get_s3_config("global_settings")[0]
    global_conf = hjson.loads(conf_.decode('utf-8'))

    ####
    assert client in global_conf['db_config'].keys(), f"Client {client} not recognized in the database config"
    assert client in global_conf['db_cluster'].keys(), f"Client {client} not recognized in the database cluster config"

@functools.lru_cache(maxsize=10)
def get_client_asset_class(client_id):
    conn, cursor = connect_postgre(db_user='tsdbadmin')

    query = f"""
        SELECT * from clients
        WHERE client_id = '{client_id}'
    """

    cursor.execute(query)
    result = cursor.fetchone()
    cursor.close()
    conn.close()

    return result["name"], result["asset_class"]

## Check if the current branch is in sync with origin head on every run
check_sync_with_origin()
check_db_host()


if __name__ == "__main__":
    get_db_users()
    # get_db_params()
    # print(get_current_branch_env())
    # print(get_s3_config('main_config', 'magha_som'))
    # print(get_s3_config('binance_config'))
    # print(get_s3_config('symbol_mapping'))
    # print(get_s3_config('global_settings'))
    # print(get_s3_config('cr_symbol_mapping'))
    # print(get_s3_config('binance_config
